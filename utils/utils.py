import os
import numpy as np   
import random     
import torch            
import cv2              
from PIL import Image   
import torch.nn as nn          
import torch.nn.functional as F  
from colorama import Fore, Style # 콘솔 출력 시 색상 적용

# 콘솔 출력 시 색상 설정
c_ = Fore.GREEN      # 초록색 텍스트
sr_ = Style.RESET_ALL # 스타일 리셋

# 설정 클래스 정의
class CFG:
    SEED = 26 
    seed = 101 

    yolo_weight = "./checkpoint/best.pt"   # YOLO 모델 가중치 경로
    model_checkpoint = "./sam2/checkpoints/sam2.1_hiera_small.pt"  # SAM 모델 가중치
    model_config = "sam2.1_hiera_s"            # SAM 모델 구성 이름
    decoder_checkpoint = "./checkpoint/best_sam2_self_supervised.bin"       # 디코더 가중치
    xgb_model = "./checkpoint/xgb_models/xgb_model_fold1.pkl"
    xgb_scaler = "./checkpoint/xgb_models/xgb_scaler_fold1.pkl"
    # 이미지 관련 설정
    train_bs = 1               # 학습 배치 사이즈
    valid_bs = 1               # 검증 배치 사이즈
    img_size = [256, 256]      # 입력 이미지 크기

    # 학습 설정
    epochs = 1                 # 에폭 수
    lr = 1e-4                  # 학습률
    scheduler = 'CosineAnnealingLR'  # 학습률 스케줄러
    min_lr = 1e-6              # 최소 학습률
    T_max = int(30000 / train_bs * epochs) + 50  # CosineAnnealing에서 사용할 max step 수
    T_0 = 25                   # CosineAnnealingWarmRestarts 주기
    warmup_epochs = 5          # 워밍업 에폭 수
    wd = 1e-6                  # weight decay 

    # Gradient Accumulation 설정
    n_accumulate = max(1, 32 // train_bs)  # 배치 사이즈에 따른 누적 횟수
    num_classes = 3                        # 클래스 수

    # 디바이스 설정 (GPU 사용 가능하면 CUDA, 아니면 CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 랜덤 시드 고정 함수
def set_seed(seed=42):
    np.random.seed(seed)                       # numpy 시드 고정
    random.seed(seed)                          # 파이썬 random 시드 고정
    torch.manual_seed(seed)                    # CPU용 torch 시드 고정
    torch.cuda.manual_seed(seed)               # GPU용 torch 시드 고정
    torch.backends.cudnn.deterministic = True  # CUDNN 연산을 결정적으로 고정
    torch.backends.cudnn.benchmark = False     # 성능 최적화 비활성화
    os.environ['PYTHONHASHSEED'] = str(seed)   # 해시 함수의 시드 고정
    print('> SEEDING DONE')                    # 완료 메시지 출력

# 초음파 이미지 전처리 함수
def preprocess_ultrasound_image(image: Image.Image):
    image = np.array(image)  # PIL 이미지를 numpy 배열로 변환
    if len(image.shape) == 3:  # 컬러 이미지인 경우
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백으로 변환
    image_eq = cv2.equalizeHist(image)  # 히스토그램 균등화로 명암 대비 개선
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE 생성
    image_clahe = clahe.apply(image_eq)  # CLAHE 적용
    image_blur = cv2.GaussianBlur(image_clahe, (5, 5), 0)  # 가우시안 블러로 노이즈 제거
    image_norm = (image_blur - image_blur.min()) / (image_blur.max() - image_blur.min())  # 정규화 (0~1)
    return Image.fromarray((image_norm * 255).astype(np.uint8))  # 다시 PIL 이미지로 변환

# 채널 확장 함수 (흑백 이미지를 RGB로 확장)
def expand_channels(image: torch.Tensor):
    if image.shape[0] == 1:               # 채널 수가 1이면
        return image.repeat(3, 1, 1)      # 3채널로 복제
    return image                          # 이미 3채널이면 그대로 반환

# 마스크를 텐서로 변환하는 함수
def mask_to_tensor(mask):
    if isinstance(mask, Image.Image):         # PIL 이미지인 경우
        mask = mask.convert("L")              # 흑백으로 변환
    elif isinstance(mask, torch.Tensor):      # 이미 텐서면 그대로 반환
        return mask
    return torch.tensor(np.array(mask), dtype=torch.long)  # numpy 배열로 바꿔서 텐서로 변환


# YOLO 기반 프롬프트 생성 함수
def generate_prompts_yolo(image_path, device, yolo_model, padding=10):
    image_np = cv2.imread(image_path)                            # 이미지 로드
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)         # BGR → RGB 변환

    results = yolo_model(image_np)                               # YOLO 모델 추론
    vis_img = results[0].plot()                                  # 시각화 이미지 생성

    # 감지된 바운딩 박스 추출
    bboxes = [box.xyxy[0].tolist() for result in results for box in result.boxes]

    if len(bboxes) == 0:  # 탐지된 객체가 없을 경우
        empty_point = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
        empty_label = torch.zeros((1, 1), dtype=torch.int32, device=device)
        empty_box   = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)
        return (empty_point, empty_label), empty_box, vis_img, False

    # 탐지된 객체가 있는 경우: 프롬프트 좌표 및 박스 생성
    point_coords = []
    point_labels = []
    box_coords = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # 중심점 계산

        point_coords.append([cx, cy])  # 포인트 좌표 저장
        point_labels.append(1)         # 라벨 1 (foreground)
        box_coords.append([x1-padding, y1-padding, x2+padding, y2+padding])  # 패딩 추가한 박스

    # 텐서로 변환하여 반환
    point_coords = torch.tensor([point_coords], dtype=torch.float32, device=device)
    point_labels = torch.tensor([point_labels], dtype=torch.int32, device=device)
    box_coords   = torch.tensor([box_coords], dtype=torch.float32, device=device)

    return (point_coords, point_labels), box_coords, vis_img, True
