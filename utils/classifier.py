import numpy as np

def compute_rosc_probability(
    mask: np.ndarray,
    xgb,
    scaler,
):
    """
    Binary mask → feature → ROSC probability
    """
    from utils.features import extract_B2_features

    feats = extract_B2_features(mask)

    X = np.array([[ 
        feats["art_area"], feats["art_major"], feats["art_minor"],
        feats["art_ecc"], feats["art_axisratio"], feats["art_round"],
        feats["ijv_area"], feats["ijv_major"], feats["ijv_minor"],
        feats["ijv_ecc"], feats["ijv_axisratio"], feats["ijv_round"],
    ]], dtype=np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    prob = float(xgb.predict_proba(scaler.transform(X))[0, 1])

    return prob, feats



