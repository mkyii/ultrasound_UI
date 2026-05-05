import os
import cv2
import torch

import numpy as np
from typing import Optional
from utils.text_renderer import TextRenderer

from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import CFG

LABEL_COLOR = {"ROSC": (0, 255, 0), "Arrest": (0, 0, 255)}  # BGR
ID_TO_NAME  = {1: "CA", 2: "IJV"}
ID_TO_COLOR = {1: (0, 0, 255), 2: (255, 0, 0)}              # CA=red(BGR), IJV=blue(BGR)

# 둥글/부드러움 관련
SMOOTH_OPEN_PX        = 3
SMOOTH_CLOSE_PX       = 5
SMOOTH_BLUR_SIGMA     = 2.5
SMOOTH_CHAIKIN_ITERS  = 2
SMOOTH_THRESH         = 0.5

# 채움/외곽선
FILL_ALPHA_DEFAULT    = 0.45   # 내부 채움 투명도(0~1)
STROKE_PX_DEFAULT     = 1      # 외곽선 두께(정수)

# UI 레이아웃
RPW = 260  # panel min width

def draw_rounded_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color, radius: int = 12, alpha: float = 1.0):
    x = int(x); y = int(y); w = int(w); h = int(h)
    r = int(max(0, min(radius, min(w, h) // 2)))
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.circle(overlay, (x + r,     y + r),     r, color, -1)
    cv2.circle(overlay, (x + w - r, y + r),     r, color, -1)
    cv2.circle(overlay, (x + r,     y + h - r), r, color, -1)
    cv2.circle(overlay, (x + w - r, y + h - r), r, color, -1)
    if alpha >= 1.0:
        return overlay
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

def draw_badge(panel, label, x, y, w, h, scale: float = 1.0, text=None):
    color  = LABEL_COLOR.get(label, (40, 40, 40))
    radius = max(8, int(12 * scale))
    panel  = draw_rounded_rect(panel, x, y, w, h, color, radius=radius, alpha=0.95)
    fs_img = max(18, int(26 * scale))
    if text is None:
        text = TextRenderer()
    text.put(panel, label, (x + int(18 * scale), y + h - int(10 * scale)),
             font_height=fs_img, color=(255, 255, 255), thickness=-1)
    return panel

def draw_meter(panel, x, y, w, h, value=None, label=None, fg=(60, 200, 80), bg=(70, 70, 70), alpha: float = 0.6):
    value = 0.0 if value is None else float(np.clip(value, 0.0, 1.0))
    if label is not None:
        fg = (0, 0, 255) if label == "Arrest" else (0, 255, 0) if label == "ROSC" else fg
    panel = draw_rounded_rect(panel, x, y, w, h, bg, radius=h // 2, alpha=alpha)
    fw = int(round(w * value))
    if fw > 0:
        panel = draw_rounded_rect(panel, x, y, fw, h, fg, radius=h // 2, alpha=min(alpha + 0.2, 1.0))
    return panel

def draw_pill(img, text_s, x, y, w=140, h=40, color=(180, 220, 235), font_scale=None, text_color=(30, 30, 30), text=None):
    img = draw_rounded_rect(img, x, y, w, h, color, radius=h // 2, alpha=0.9)
    if text is None:
        text = TextRenderer()
    font_h = max(14, int(22 * (h / 40.0))) if font_scale is None else int(22 * font_scale * (h / 40.0))
    tx = x + max(10, int(14 * (h / 40.0)))
    ty = y + int(h * 0.72)
    text.put(img, text_s, (tx, ty), font_height=font_h, color=text_color, thickness=-1)
    return img

def draw_legend(img, items, x=12, y=12, scale=1.0, text=None):
    if text is None:
        text = TextRenderer()
    yy = y
    for name, color in items:
        img = draw_rounded_rect(img, x, yy, int(120 * scale), int(32 * scale), (25, 25, 25), radius=int(14 * scale), alpha=0.55)
        cv2.circle(img, (x + int(18 * scale), yy + int(16 * scale)), int(7 * scale), color, -1, cv2.LINE_AA)
        text.put(img, name, (x + int(34 * scale), yy + int(22 * scale)), font_height=max(14, int(20 * scale)),
                 color=(230, 230, 230), thickness=-1)
        yy += int(36 * scale)
    return img

def draw_sparkline(img, series_r, series_b, x, y, w, h, pad=12, text=None):
    img = draw_rounded_rect(img, x, y, w, h, (30, 30, 30), radius=10, alpha=1.0)
    x0 = x + pad; y0 = y + h - pad; x1 = x + w - pad; y1 = y + pad
    cv2.arrowedLine(img, (x0, y0), (x1, y0), (220, 220, 220), 1, tipLength=0.02)
    cv2.arrowedLine(img, (x0, y0), (x0, y1), (220, 220, 220), 1, tipLength=0.02)

    def to_pts(arr):
        if len(arr) < 2:
            return []
        v = np.clip(np.asarray(arr, np.float32), 0.0, 1.0)
        xs = np.linspace(x0, x1, num=len(v)).astype(np.int32)
        ys = (y0 - (y0 - y1) * v).astype(np.int32)
        return list(zip(xs, ys))

    pts_r = to_pts(series_r)
    if len(pts_r) >= 2:
        cv2.polylines(img, [np.int32(pts_r)], False, (0, 0, 255), 2, cv2.LINE_AA)
    return img

def draw_progress(img, cur, total, margin=12, h=10, fg=(200, 200, 200), bg=(60, 60, 60), alpha=0.5):
    H, W = img.shape[:2]
    x, y = margin, H - margin - h
    w = W - 2 * margin
    img = draw_rounded_rect(img, x, y, w, h, bg, radius=h // 2, alpha=alpha)
    frac = 0 if total <= 0 else float(cur) / float(total)
    img = draw_rounded_rect(img, x, y, int(w * frac), h, fg, radius=h // 2, alpha=alpha)
    return img

def largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(m)
    if num <= 1:
        return m
    areas = np.bincount(labels.ravel()); areas[0] = 0
    return (labels == np.argmax(areas)).astype(np.uint8)

def calculate_eccentricity(binary: np.ndarray) -> float:
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float("nan")
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return float("nan")
    (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
    a = max(MA, ma) / 2.0
    b = min(MA, ma) / 2.0
    if a <= 1e-6:
        return 0.0
    return float(np.sqrt(1.0 - (b * b) / (a * a)))

def _elliptic(k: int) -> np.ndarray:
    k = int(max(1, k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def _chaikin(cnt: np.ndarray, iters: int = 1) -> np.ndarray:
    pts = cnt[:, 0, :].astype(np.float32) if cnt.ndim == 3 else cnt.astype(np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32).reshape(-1, 1, 2)
    for _ in range(max(0, iters)):
        new = []
        for i in range(len(pts)):
            p0, p1 = pts[i], pts[(i + 1) % len(pts)]
            new += [0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1]
        pts = np.asarray(new, np.float32)
    return pts.astype(np.int32).reshape(-1, 1, 2)

def make_roundish(binary: np.ndarray,
                  open_px: int = SMOOTH_OPEN_PX,
                  close_px: int = SMOOTH_CLOSE_PX,
                  blur_sigma: float = SMOOTH_BLUR_SIGMA,
                  chaikin_iters: int = SMOOTH_CHAIKIN_ITERS,
                  thresh: float = SMOOTH_THRESH):
    """
    binary: 0/1 or 0/255 (H,W)
    return: mask_bin(0/1), alpha_soft(0~1), smooth_contours(list of int32 Nx1x2)
    """
    m = (binary > 0).astype(np.uint8) * 255
    if open_px  > 0: m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  _elliptic(open_px))
    if close_px > 0: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _elliptic(close_px))

    soft = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), blur_sigma)
    soft = np.clip(soft, 0.0, 1.0)

    mask_bin = (soft > float(thresh)).astype(np.uint8)

    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    smooth_cnts = [_chaikin(c, iters=chaikin_iters) for c in cnts]
    return mask_bin, soft, smooth_cnts

def draw_yolo_only(image, box_coords, input_size=(256, 256)):
    img = image.copy()

    if box_coords is None:
        return img

    boxes = box_coords.squeeze(0).detach().cpu().numpy()

    H, W = img.shape[:2]
    in_h, in_w = input_size

    scale_x = W / in_w
    scale_y = H / in_h

    for box in boxes:
        x1, y1, x2, y2 = box

        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

def overlay_round(img: np.ndarray,
                  mask_bin: np.ndarray,
                  alpha_soft: np.ndarray,
                  color,
                  fill_alpha: float = FILL_ALPHA_DEFAULT,
                  stroke_px: int = STROKE_PX_DEFAULT,
                  smooth_cnts=None):
    """
    mask_bin: 0/1, alpha_soft: 0~1
    fill_alpha: 내부 투명도(0~1), stroke_px: 외곽선 두께(정수)
    """
    out = img.copy()
    col = np.empty_like(out); col[:] = np.array(color, np.uint8)

    a = (np.clip(alpha_soft, 0, 1) * float(np.clip(fill_alpha, 0, 1)))[..., None].astype(np.float32)
    out = (out * (1 - a) + col * a).astype(np.uint8)

    if smooth_cnts is None:
        cnts, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        smooth_cnts = cnts
    th = max(1, int(round(stroke_px)))
    if len(smooth_cnts):
        cv2.polylines(out, smooth_cnts, isClosed=True, color=color, thickness=th, lineType=cv2.LINE_AA)
    return out

def _ensure_palette_768(pal):
    if pal is None:
        pal = [0, 0, 0, 255, 0, 0, 0, 0, 255] + [0, 0, 0] * 253
    elif isinstance(pal, (bytes, bytearray)):
        pal = list(pal)
    if len(pal) < 768:
        pal = (pal + [0] * 768)[:768]
    return pal

def save_ann_png(path, mask, palette=None):
    assert mask.dtype == np.uint8 and mask.ndim == 2
    output_mask = Image.fromarray(mask, mode='P')
    if palette is None:
        palette = [
            0, 0, 0,      # 0: 배경
            255, 0, 0,    # 1: 빨강
            0, 0, 255,    # 2: 파랑
            0, 255, 0     # 3: 초록
        ] + [0, 0, 0] * 252
    output_mask.putpalette(_ensure_palette_768(palette))
    output_mask.save(path)

def get_per_obj_mask(mask):
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    return {object_id: (mask == object_id) for object_id in object_ids}

def put_per_obj_mask(per_obj_mask, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for object_id in sorted(per_obj_mask)[::-1]:
        object_mask = per_obj_mask[object_id]
        if object_mask.ndim > 2:
            object_mask = object_mask[0]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask.astype(bool)] = object_id
    return mask

def draw_fps_hud(img: np.ndarray, e2e_fps: float, model_fps: float,
                 x: int = 12, y: int = 12) -> np.ndarray:
    """왼쪽 상단에 작은 반투명 박스로 FPS 표기"""
    w, h = 200, 46
    img = draw_rounded_rect(img, x, y, w, h, (0,0,0), radius=12, alpha=0.45)
    s1 = f"E2E  : {e2e_fps:5.1f} fps"
    s2 = f"Model: {model_fps:5.1f} fps"
    cv2.putText(img, s1, (x+12, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, s2, (x+12, y+36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
    return img


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        target_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        logpt = (logpt * target_one_hot).sum(dim=1)
        pt = (pt * target_one_hot).sum(dim=1)
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def predict_single_sequence(model_class, weight_path, input_dim, X_single, return_prob=False):
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(weight_path, map_location=CFG.device))
    model.to(CFG.device)
    model.eval()
    with torch.no_grad():
        logits = model(X_single)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        prob_value = probs[0][pred].item()
    return (pred, prob_value) if return_prob else pred

class SessionAggregator:
    def __init__(self, num_classes=2, eps=1e-8):
        self.store = defaultdict(list)
        self.num_classes = num_classes
        self.eps = eps

    def add_logits(self, pid, logits_1c: torch.Tensor):
        self.store[pid].append(logits_1c.detach().float().cpu().numpy())

    def finalize(self):
        out_probs, out_preds = {}, {}
        for pid, L in self.store.items():
            logit_mean = np.mean(np.stack(L, axis=0), axis=0)
            p = F.softmax(torch.tensor(logit_mean), dim=0).numpy()
            out_probs[pid] = p
            out_preds[pid] = int(np.argmax(p))
        return out_probs, out_preds
    

class TextRenderer:
    def __init__(self, ttf_path: Optional[str] = None, fallback_font=cv2.FONT_HERSHEY_SIMPLEX):
        self.use_ft = False
        self.fallback_font = fallback_font
        try:
            self.ft = cv2.freetype.createFreeType2()  # type: ignore[attr-defined]
            if ttf_path is None:
                candidates = [
                    "C:/Windows/Fonts/malgun.ttf",
                    "C:/Windows/Fonts/malgunbd.ttf",
                    "C:/Windows/Fonts/NanumGothic.ttf",
                    "C:/Windows/Fonts/NanumSquareR.ttf",
                ]
                for p in candidates:
                    if os.path.exists(p):
                        ttf_path = p
                        break
            if ttf_path and os.path.exists(ttf_path):
                self.ft.loadFontData(fontFileName=ttf_path, id=0)
                self.use_ft = True
        except Exception:
            self.use_ft = False

    def put(self, img, text, org, font_height=28, color=(255, 255, 255), thickness=-1):
        if self.use_ft:
            self.ft.putText(img, text, org, fontHeight=int(font_height),
                            color=color, thickness=thickness, line_type=cv2.LINE_AA,
                            bottomLeftOrigin=False)
        else:
            fs = max(0.4, font_height / 32.0)
            th = max(1, int(round(max(1, thickness if thickness > 0 else 2) * fs)))
            x, y = org
            cv2.putText(img, text, (x, y), self.fallback_font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), self.fallback_font, fs, color,   th,   cv2.LINE_AA)

