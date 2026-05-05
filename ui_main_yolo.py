import torch
from utils.utils import CFG
from model import build_model
from Dataset.dataloader import nano_streaming_loader
from utils.ui_epoch import ui_yolo_epoch

import warnings
warnings.filterwarnings("ignore")


def main():
    intra_loader = nano_streaming_loader()

    model = build_model()

    mask_decoder = model.sam2_model.sam_mask_decoder.to(CFG.device)
    decoder_path = CFG.decoder_checkpoint
    mask_decoder.load_state_dict(torch.load(decoder_path, map_location=CFG.device))

    ui_yolo_epoch(
            mask_decoder, intra_loader, CFG.device, epoch=0,
            show_window=True,
            video_out_path="./results/016_4_30.mp4",
            video_fps=30,
            use_fp16=True,
            out_size=(1210, 643),   # 최종 프레임 고정
            panel_ratio=0.24,       # 패널 폭 비율
            left_ratio=0.76,        # 또는 왼쪽 비율 고정
            ui_density=0.80,        # UI 전체 축소
        )

if __name__ == "__main__":
    main()
