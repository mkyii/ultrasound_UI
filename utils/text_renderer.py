import os
import cv2

from typing import Optional

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