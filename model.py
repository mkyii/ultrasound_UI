import torch.nn as nn
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.utils import CFG
    
class SAM2FineTune(nn.Module):
    def __init__(self, checkpoint, device="cuda"):
        super(SAM2FineTune, self).__init__()
        self.device = device
        self.checkpoint = checkpoint

        # SAM2 모델 생성
        self.sam2_model = build_sam2(config_file=CFG.model_config, ckpt_path=self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        #  모델을 명시적으로 학습 모드로 설정
        self.sam2_model.train()
        self.sam2_model.requires_grad_(True)  # 모든 레이어의 requires_grad 활성화

        # Train mask decoder
        self.predictor.model.sam_mask_decoder.train(True)
        for param in self.predictor.model.sam_mask_decoder.parameters():
            param.requires_grad = True

        # Train prompt encoder
        self.predictor.model.sam_prompt_encoder.train(True)
        for param in self.predictor.model.sam_prompt_encoder.parameters():
            param.requires_grad = True

        self.to(self.device)  # 모델을 디바이스로 이동

    def forward(self, images, masks=None):
        images = images.to(self.device)
        
        if masks is not None:
            masks = masks.to(self.device)

        self.predictor.set_image(images)
        outputs = self.predictor.predict()

        return outputs

def build_model():
    model = SAM2FineTune(checkpoint=CFG.model_checkpoint)
    model.to(CFG.device)
    return model

def load_model(path, device):
    # 모델 초기화 (체크포인트 없이 생성)
    model = SAM2FineTune()  
    model.to(device)

    # 저장된 state_dict 로드
    checkpoint = torch.load(path, map_location=device)  #  모델 가중치 로드
    model.load_state_dict(checkpoint)  #  state_dict 적용

    model.eval()  # 평가 모드로 전환
    return model
