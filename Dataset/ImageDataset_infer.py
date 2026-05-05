import os, time, cv2, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.utils import CFG
from utils.ui_draw import draw_yolo_only

class InfernanoDataset(Dataset):
    def __init__(self, root_path, mode='train',
                 transform=None, max_objects=3,
                 imgsz=256, yolo_conf=0.25, det_stride=1,  # ← 새 옵션
                 verbose_every=10):
        super().__init__()
        self.root_path = os.path.abspath(root_path)
        self.transform = transform or transforms.ToTensor()
        self.max_objects = max_objects
        self.mode = mode
        self.imgsz = imgsz
        self.yolo_conf = yolo_conf
        self.det_stride = max(1, int(det_stride))
        self.verbose_every = verbose_every

        self.device = CFG.device
        self.total_time = 0.0
        self.frame_count = 0

        # 이미지 목록
        self.samples = []
        for root, _, files in os.walk(self.root_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    self.samples.append({
                        'image': os.path.join(root, file),
                        'file_name': os.path.splitext(file)[0]
                    })

        # YOLO 초기화 (FP16 + fuse)
        self.yolo_model = YOLO(CFG.yolo_weight)
        self.yolo_model.to(self.device)
        try: self.yolo_model.fuse()
        except: pass
        if self.device.type == "cuda":
            try: self.yolo_model.model.half()
            except: pass

        # SAM2 predictor
        self.sam2_model = build_sam2(config_file=CFG.model_config,
                                     ckpt_path=CFG.model_checkpoint,
                                     device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        # 박스 캐시 (det_stride > 1 일 때 재사용)
        self._last_boxes = None
        self._last_points = None

    def __len__(self):
        return len(self.samples)

    def _read_rgb_uint8(self, path):
        # 빠르고 일관된 로딩: cv2로 BGR 읽고 RGB로 변환 → uint8 RGB
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb  # H,W,3 uint8

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t0 = time.perf_counter()

        # 1) 이미지 로드 (uint8 RGB)
        rgb = self._read_rgb_uint8(sample['image'])

        # 2) torch 텐서(overlay 등 위해)도 한번 생성
        img_tensor = self.transform(Image.fromarray(rgb))  # (C,H,W), float[0,1]

        rgb_for_model = (
            img_tensor.permute(1, 2, 0)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )
        # 3) YOLO 프롬프트 생성 (det_stride로 프레임 스킵)
        if (idx % self.det_stride) == 0 or self._last_boxes is None:
            # generate_prompts_yolo가 ndarray 입력도 받도록 수정 권장:
            # point_prompt, box_coords, vis_img = generate_prompts_yolo(rgb, self.device, self.yolo_model, imgsz=self.imgsz, conf=self.yolo_conf)
            # 경로만 받는 기존 함수라면 다음처럼 직접 호출:
            res = self.yolo_model.predict(
                source=rgb_for_model,
                imgsz=self.imgsz,
                conf=self.yolo_conf,
                iou=0.5,
                device=0 if self.device.type == 'cuda' else 'cpu',
                verbose=False
            )[0]

            boxes = res.boxes.xyxy  # (N,4) tensor on device
            clss  = res.boxes.cls
            confs = res.boxes.conf
            # 클래스/신뢰도 정렬, 상위 2~3개만 선택 (예시)
            # 필요하면 여기서 artery/vein 클래스별 1개씩 선택 로직 추가
            if boxes is None or len(boxes)==0:
                # 미검출: 이전 박스 재사용 or 더미 박스
                box_coords = None
                point_prompt = None
            else:
                box_coords = boxes.unsqueeze(0)  # (1,B,4)
                # 박스 중심 포인트를 foreground로 (1=FG)
                centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
                point_prompt = (
                    centers.unsqueeze(0).to(self.device),                   # (1,B,2)
                    torch.ones((1, centers.shape[0]), dtype=torch.int64, device=self.device)  # (1,B)
                )
            self._last_boxes = box_coords
            self._last_points = point_prompt
        else:
            box_coords = self._last_boxes
            point_prompt = self._last_points

        # 4) SAM2 임베딩 추출 (uint8 RGB 바로 전달)
        with torch.no_grad():
            self.predictor.set_image(rgb_for_model)
            print("\n===== SAM INPUT DEBUG =====")
            print("box_coords:", None if box_coords is None else box_coords.shape)
            print("point_prompt:", None if point_prompt is None else point_prompt[0].shape)
            print("point_labels:", None if point_prompt is None else point_prompt[1].shape)

            image_embeddings = self.predictor.get_image_embedding()
            image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe()
            high_res_features = [feat for feat in self.predictor._features["high_res_feats"]]

            # prompt encoder (항상 계산: train/eval 동일 구조 반환)
            if point_prompt is None and box_coords is None:
                # 완전 미검출 시 더미 처리 (모델이 이를 핸들하도록)
                sparse_embeddings = torch.zeros((1, 0, 256), device=self.device)   # 샘 값은 모델 구조에 맞게
                dense_embeddings = torch.zeros_like(image_embeddings[0:1])          # 모양만 일치
            else:
                sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                    points=point_prompt,
                    boxes=box_coords,
                    masks=None
                )
            dense_embeddings = dense_embeddings.squeeze(0)
        

        for i, feat in enumerate(high_res_features):
            print(f"high_res_features[{i}]:", feat.shape)

        # 5) 속도 로그 (간격 낮춤)
        dt = time.perf_counter() - t0
        self.total_time += dt
        self.frame_count += 1
        
        # 6) 반환 (상위 코드가 .to(device) 하므로 여기선 그대로 반환; 이미 device 상에 있음)
        return (
            img_tensor,                 # (C,H,W) float[0,1] (CPU)
            sparse_embeddings,          # (1,P,C) (GPU or CPU depending on predictor device)
            dense_embeddings,           # (P',C,...) squeezed (GPU)
            image_embeddings,           # (GPU)
            image_pe,                   # (GPU)
            high_res_features,          # list[GPU tensor]
            sample['image'],            # str
            box_coords,                 # (1,B,4) or None
            rgb
        )
