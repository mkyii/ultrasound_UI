import os
import cv2
import torch
import joblib
import numpy as np

import torch
import torch.nn.functional as F

from typing import Optional, Tuple
from collections import deque
from tqdm import tqdm

from utils.infer import infer_masks, postprocess_masks
from utils.classifier import compute_rosc_probability
from utils.renderer import render_segmentation

from utils.text_renderer import TextRenderer
from utils.ui_draw import draw_badge, draw_meter, draw_sparkline, draw_yolo_only


from tqdm import tqdm
from utils.utils import CFG

@torch.inference_mode()
def ui_epoch(
    model,dataloader,device,
    epoch: int,
    show_window: bool = True,
    video_out_path: Optional[str] = None,
    video_fps: int = 25,
    alpha: float = 0.6,
    use_fp16: bool = False,
    prob_threshold: float = 0.5,
    out_size: Optional[Tuple[int, int]] = None,
    ui_scale: float = 1.6,
    vis_stride: int = 1,
    panel_width: int = 260,
    panel_ratio: Optional[float] = 0.24,
    left_ratio: Optional[float] = 0.76,
    ui_density: float = 0.80,):
    """UI visualization + inference loop for IntraCPR-Net"""

    # ------------------ basic setup ------------------
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True

    model.eval()
    if use_fp16 and device.type == "cuda":
        model.half()

    # ------------------ load classifier ------------------
    xgb = joblib.load(CFG.xgb_model)
    scaler = joblib.load(CFG.xgb_scaler)

    MIN_FRAMES = 30
    frame_prob_buffer = []
    current_patient_id = None
    save_frame_dir = './visible'

    # history for graph
    e_hist_ca = deque(maxlen=180)

    text = TextRenderer()
    win_name = f"IntraCPR-Net (epoch {epoch})"

    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    writer = None
    resolved_size = None

    # ------------------ main loop ------------------
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Vis {epoch}",
    )

    for step, batch in pbar:

        import time

        start_time = time.time()
        frame_count = 0

        # ---------- inference ----------
        low_res_masks = infer_masks(
            model=model,
            batch=batch,
            device=device,
            use_fp16=use_fp16,
        )

        prd_masks = postprocess_masks(low_res_masks)

        images = batch[0]
        paths = batch[6]
        B = prd_masks.size(0)

        prd_masks = torch.argmax(F.softmax(low_res_masks, dim=1), dim=1)


        # ---------- per-frame ----------
        for i in range(B):

            filename = os.path.splitext(os.path.basename(paths[i]))[0]
            patient_id = "_".join(filename.split("_")[:3])

            # reset per patient
            if current_patient_id is None:
                current_patient_id = patient_id
            elif patient_id != current_patient_id:
                frame_prob_buffer.clear()
                e_hist_ca.clear()
                current_patient_id = patient_id

            # ---------- base image ----------
            img_tensor = images[i]
            mask = prd_masks[i].detach().cpu().numpy().astype(np.uint8)

            if save_frame_dir is not None:
                os.makedirs(save_frame_dir, exist_ok=True)

                save_name = f"{patient_id}_{filename}_{step}_{i}.png"
                save_path = os.path.join(save_frame_dir, save_name)

                cv2.imwrite(save_path, mask)

            vis = render_segmentation(
                image=img_tensor,
                mask=mask,
                alpha=alpha,
            )

            H, W = vis.shape[:2]

            resize_mask = cv2.resize(
                mask.astype(np.uint8),
                (480, 640),
                interpolation=cv2.INTER_NEAREST
            )

            # ---------- feature & ROSC prob ----------
            prob, feats = compute_rosc_probability(
                mask=resize_mask,
                xgb=xgb,
                scaler=scaler,
            )

            frame_prob_buffer.append(prob)

            if not np.isnan(feats["art_ecc"]):
                e_hist_ca.append(float(feats["art_ecc"]))

            # ---------- temporal decision ----------
            label = "WARMUP"
            p_rosc = None

            if len(frame_prob_buffer) >= MIN_FRAMES:
                p_rosc = float(np.mean(frame_prob_buffer))
                label = "ROSC" if p_rosc >= prob_threshold else "Arrest"

            # ---------- output size resolve ----------
            if resolved_size is None:
                base_w = W + panel_width
                base_h = H

                if out_size is not None:
                    resolved_size = out_size
                else:
                    resolved_size = (
                        int(base_w * ui_scale),
                        int(base_h * ui_scale),
                    )

                if show_window:
                    cv2.resizeWindow(win_name, *resolved_size)

                if video_out_path:
                    os.makedirs(os.path.dirname(video_out_path), exist_ok=True)
                    writer = cv2.VideoWriter(
                        video_out_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        video_fps,
                        resolved_size,
                    )

            total_w, total_h = resolved_size
            left_w = int(total_w * left_ratio)
            panel_w = total_w - left_w

            # ---------- compose ----------
            vis_r = cv2.resize(vis, (left_w, total_h))
            panel = np.full((total_h, panel_w, 3), (18, 18, 18), np.uint8)

            panel = draw_badge(
                panel,
                label,
                12,
                12,
                w=panel_w - 24,
                h=40,
                text=text,
            )

            panel = draw_meter(
                panel,
                12,
                60,
                panel_w - 24,
                14,
                p_rosc or 0.0,
                label,
            )

            # art_ecc sparkline
            graph_h = int(100 * ui_density)
            graph_y = total_h - graph_h - 20

            panel = draw_sparkline(
                panel,
                list(e_hist_ca),
                None,
                x=12,
                y=graph_y,
                w=panel_w - 24,
                h=graph_h,
                pad=8,
                text=text,
            )

            comp = np.zeros((total_h, total_w, 3), np.uint8)
            comp[:, :left_w] = vis_r
            comp[:, left_w:] = panel
            cv2.line(
                comp,
                (left_w, 0),
                (left_w, total_h),
                (90, 90, 90),
                1,
            )

            # ---------- output ----------
            if writer:
                writer.write(comp)

            if show_window:
                cv2.imshow(win_name, comp)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    show_window = False

        frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print(f"FPS: {fps:.2f}", end="\r")
    # ------------------ cleanup ------------------
    if writer:
        writer.release()

    if show_window:
        cv2.destroyWindow(win_name)

    print("Inference finished.")





@torch.inference_mode()
def ui_yolo_epoch(
    model,dataloader,device,
    epoch: int,
    show_window: bool = True,
    video_out_path: Optional[str] = None,
    video_fps: int = 25,
    alpha: float = 0.6,
    use_fp16: bool = False,
    prob_threshold: float = 0.5,
    out_size: Optional[Tuple[int, int]] = None,
    ui_scale: float = 1.6,
    vis_stride: int = 1,
    panel_width: int = 260,
    panel_ratio: Optional[float] = 0.24,
    left_ratio: Optional[float] = 0.76,
    ui_density: float = 0.80,):
    """UI visualization + inference loop for IntraCPR-Net"""

    # ------------------ basic setup ------------------
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True

    model.eval()
    if use_fp16 and device.type == "cuda":
        model.half()

    # ------------------ load classifier ------------------
    xgb = joblib.load(CFG.xgb_model)
    scaler = joblib.load(CFG.xgb_scaler)

    MIN_FRAMES = 30
    frame_prob_buffer = []
    current_patient_id = None
    save_frame_dir = './visible'

    # history for graph
    e_hist_ca = deque(maxlen=180)

    text = TextRenderer()
    win_name = f"IntraCPR-Net (epoch {epoch})"

    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    writer = None
    resolved_size = None

    # ------------------ main loop ------------------
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Vis {epoch}",
    )

    for step, batch in pbar:

        import time

        start_time = time.time()
        frame_count = 0

        # ---------- inference ----------
        low_res_masks = infer_masks(
            model=model,
            batch=batch,
            device=device,
            use_fp16=use_fp16,
        )

        prd_masks = postprocess_masks(low_res_masks)

        images = batch[0]
        paths = batch[6]
        box_coords = batch[7]
        raw_img = batch[8]
        B = prd_masks.size(0)

        if isinstance(raw_img, list):
            raw_img = raw_img[0]

        H, W = raw_img.shape[:2]
        print(f"image shape: {H,W}")

        vis = draw_yolo_only(raw_img, box_coords)

        prd_masks = torch.argmax(F.softmax(low_res_masks, dim=1), dim=1)

        # ---------- per-frame ----------
        for i in range(B):

            filename = os.path.splitext(os.path.basename(paths[i]))[0]
            patient_id = "_".join(filename.split("_")[:3])

            # reset per patient
            if current_patient_id is None:
                current_patient_id = patient_id
            elif patient_id != current_patient_id:
                frame_prob_buffer.clear()
                e_hist_ca.clear()
                current_patient_id = patient_id

            # ---------- base image ----------
            img_tensor = images[i]
            mask = prd_masks[i].detach().cpu().numpy().astype(np.uint8)

            if save_frame_dir is not None:
                os.makedirs(save_frame_dir, exist_ok=True)

                save_name = f"{patient_id}_{filename}_{step}_{i}.png"
                save_path = os.path.join(save_frame_dir, save_name)

                cv2.imwrite(save_path, mask)


            H, W = vis.shape[:2]

            resize_mask = cv2.resize(
                mask.astype(np.uint8),
                (480, 640),
                interpolation=cv2.INTER_NEAREST
            )

            # ---------- feature & ROSC prob ----------
            prob, feats = compute_rosc_probability(
                mask=resize_mask,
                xgb=xgb,
                scaler=scaler,
            )

            frame_prob_buffer.append(prob)

            if not np.isnan(feats["art_ecc"]):
                e_hist_ca.append(float(feats["art_ecc"]))

            # ---------- temporal decision ----------
            label = "WARMUP"
            p_rosc = None

            if len(frame_prob_buffer) >= MIN_FRAMES:
                p_rosc = float(np.mean(frame_prob_buffer))
                label = "ROSC" if p_rosc >= prob_threshold else "Arrest"

            # ---------- output size resolve ----------
            if resolved_size is None:
                base_w = W + panel_width
                base_h = H

                if out_size is not None:
                    resolved_size = out_size
                else:
                    resolved_size = (
                        int(base_w * ui_scale),
                        int(base_h * ui_scale),
                    )

                if show_window:
                    cv2.resizeWindow(win_name, *resolved_size)

                if video_out_path:
                    os.makedirs(os.path.dirname(video_out_path), exist_ok=True)
                    writer = cv2.VideoWriter(
                        video_out_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        video_fps,
                        resolved_size,
                    )

            total_w, total_h = resolved_size
            left_w = int(total_w * left_ratio)
            panel_w = total_w - left_w

            # ---------- compose ----------
            vis_r = cv2.resize(vis, (left_w, total_h))
            panel = np.full((total_h, panel_w, 3), (18, 18, 18), np.uint8)

            panel = draw_badge(
                panel,
                label,
                12,
                12,
                w=panel_w - 24,
                h=40,
                text=text,
            )

            panel = draw_meter(
                panel,
                12,
                60,
                panel_w - 24,
                14,
                p_rosc or 0.0,
                label,
            )

            # art_ecc sparkline
            graph_h = int(100 * ui_density)
            graph_y = total_h - graph_h - 20

            panel = draw_sparkline(
                panel,
                list(e_hist_ca),
                None,
                x=12,
                y=graph_y,
                w=panel_w - 24,
                h=graph_h,
                pad=8,
                text=text,
            )

            comp = np.zeros((total_h, total_w, 3), np.uint8)
            comp[:, :left_w] = vis_r
            comp[:, left_w:] = panel
            cv2.line(
                comp,
                (left_w, 0),
                (left_w, total_h),
                (90, 90, 90),
                1,
            )

            # ---------- output ----------
            if writer:
                writer.write(comp)

            if show_window:
                cv2.imshow(win_name, comp)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    show_window = False

        frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print(f"FPS: {fps:.2f}", end="\r")
    # ------------------ cleanup ------------------
    if writer:
        writer.release()

    if show_window:
        cv2.destroyWindow(win_name)

    print("Inference finished.")

    