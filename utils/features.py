import cv2
import numpy as np

def extract_ellipse(mask):
    feats = []
    for label in [1, 2]:  # artery=1, IJV=2
        comp = (mask == label).astype(np.uint8)
        area = comp.sum()

        if area < 5:
            feats += [0]*6
            continue

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)

        if len(c) < 5:
            feats += [0]*6
            continue

        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        major = max(MA, ma)
        minor = min(MA, ma)

        eccentricity = np.sqrt(1 - (minor/major)**2)
        axis_ratio = major / (minor + 1e-8)

        peri = cv2.arcLength(c, True)
        roundness = 4*np.pi*area / (peri**2 + 1e-8)

        feats += [area, major, minor, eccentricity, axis_ratio, roundness]

    return feats

def extract_B2_features(mask):
    feats = extract_ellipse(mask)

    B2 = {
        "art_area": feats[0],
        "art_major": feats[1],
        "art_minor": feats[2],
        "art_ecc": feats[3],
        "art_axisratio":feats[4],
        "art_round": feats[5],
        "ijv_area": feats[6],
        "ijv_major": feats[7],
        "ijv_minor": feats[8],
        "ijv_ecc": feats[9],
        "ijv_axisratio":feats[10],
        "ijv_round": feats[11],
    }

    return B2