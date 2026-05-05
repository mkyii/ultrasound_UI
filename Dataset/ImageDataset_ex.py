import os
import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.utils import CFG, generate_prompts_yolo

# img - png / mask - numpy 시 데이터셋

# 탐지가 없을 때 사용될 빈 텐서 반환 함수
def get_empty_prompt_tensors(device):
    sparse = torch.zeros((1, 0, 256), dtype=torch.float32, device=device)        # 빈 sparse 임베딩
    dense  = torch.zeros((1, 256, 64, 64), dtype=torch.float32, device=device)   # dense 임베딩
    box    = torch.zeros((1, 0, 4), dtype=torch.float32, device=device)          # 빈 box 좌표
    return sparse, dense, box


class exVisualDataset(Dataset):  # PyTorch Dataset 상속
    def __init__(self, root_path, mask_path, label_path, mode='train',
                 transform=None, mask_transform=None, max_objects=3):

        super().__init__()

        self.root_path = os.path.abspath(root_path)   # 이미지 경로
        self.mask_path = os.path.abspath(mask_path)   # 마스크 경로

        self.labels_df = pd.read_csv(label_path)      # CSV 파일 읽어와서 DataFrame으로 저장

        self.transform = transform or transforms.ToTensor()       # 이미지 전처리
        self.mask_transform = mask_transform or transforms.ToTensor()  # 마스크 전처리
        self.max_objects = max_objects
        self.mode = mode       # 'train' 또는 'eval' 모드

        # 샘플 리스트 생성 (이미지, 마스크, 라벨 등 포함)
        self.samples = []
        for _, row in self.labels_df.iterrows():

            file_name = str(row['file_name']).zfill(5)
                        
            image_file = os.path.join(self.root_path, row['folder_name'], file_name + '.png')
            mask_file  = os.path.join(self.mask_path, row['folder_name'], file_name + '.npy')
            label = int(row.get('category', None))

            if os.path.exists(image_file):  # 이미지가 실제로 존재하면 샘플 추가
                self.samples.append({
                    'image': image_file,
                    'mask': mask_file if os.path.exists(mask_file) else None,
                    'label': label,
                    'file_name': row['file_name'],
                    'patient': row['folder_name']
                })

        # 모델 초기화 (YOLO + SAM2)
        self.device = CFG.device
        self.yolo_model = YOLO("./checkpoint/emr_yolo.pt")  # YOLO 객체 탐지 모델
        self.sam2_model = build_sam2(
            config_file=CFG.model_config,
            ckpt_path=CFG.model_checkpoint,
            device=self.device
        )
        self.predictor = SAM2ImagePredictor(self.sam2_model)  # 이미지 예측기

        self.total_time = 0   # 시간 측정용
        self.frame_count = 0  # 프레임 수

    def __len__(self):
        return len(self.samples)  # 전체 샘플 수

    def __getitem__(self, idx):
        sample = self.samples[idx]  # 해당 인덱스의 샘플 가져오기

        # 이미지 로드 및 전처리
        img = Image.open(sample['image']).convert("RGB")
        img_tensor = self.transform(img)

        # 마스크 로드 (있으면), 없으면 0으로 채운 마스크 생성
        if sample['mask'] and os.path.exists(sample['mask']):

            mask_np = np.load(sample['mask'])

            mask = Image.fromarray(mask_np.astype(np.uint8))

            mask = mask.resize((img_tensor.shape[2], img_tensor.shape[1]), Image.NEAREST)

            mask_tensor = torch.from_numpy(np.array(mask)).long()

        else:

            mask_tensor = torch.zeros((img_tensor.shape[1], img_tensor.shape[2])).long()

        # YOLO로 프롬프트 (박스/포인트) 생성
        ret = generate_prompts_yolo(sample['image'], self.device, self.yolo_model)
        point_prompt, box_coords, vis_img, has_det = ret

        # SAM2에서 임베딩 추출
        image_np = img_tensor.permute(1, 2, 0).numpy()  # HWC 형태로 변환
        with torch.no_grad():
            self.predictor.set_image(image_np)
            image_embeddings = self.predictor.get_image_embedding()
            image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe()
            high_res_features = [feat for feat in self.predictor._features["high_res_feats"]]

        # 프롬프트가 없을 경우 → dummy 텐서 사용
        if not has_det:
            sparse_embeddings, dense_embeddings, box_out = get_empty_prompt_tensors(self.device)
        else:
            sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                points=point_prompt, boxes=box_coords, masks=None
            )
            dense_embeddings = dense_embeddings.squeeze(0)
            box_out = box_coords

        # 학습 모드일 때는 모든 정보 반환
        if self.mode == "train":
            return (
                img_tensor,                    # 0: 전처리된 이미지
                mask_tensor,                   # 1: 정답 마스크
                sparse_embeddings.cpu(),       # 2: sparse prompt embedding
                dense_embeddings.cpu(),        # 3: dense prompt embedding
                image_embeddings.cpu(),        # 4: 이미지 임베딩
                image_pe.cpu(),                # 5: positional encoding
                high_res_features,             # 6: 고해상도 특징맵
                sample['image'],               # 7: 이미지 경로
                box_out.cpu(),                 # 8: bounding box
                sample['label'],               # 9: 클래스 라벨
                sample['patient']              # 10: 환자 ID
            )
        else:  # 평가 모드
            return (
                img_tensor,                    # 0
                mask_tensor,                   # 1
                image_embeddings.cpu(),        # 2
                image_pe.cpu(),                # 3
                high_res_features,             # 4
                sample['image'],               # 5
                box_out.cpu(),                 # 6
                sample['label'],               # 7
                sample['patient']              # 8
            )