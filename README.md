# AI_IntraCPR

checkpoint
https://drive.google.com/file/d/17ORhT4tBFoeonKs-MkVlhOK1kRsxUEXi/view?usp=sharing

🔬 IntraCPR-Net
Real-time carotid ultrasound–based AI system for ROSC detection during CPR

🧠 Overview
IntraCPR-Net is an end-to-end deep learning framework designed to assess return of spontaneous circulation (ROSC) during ongoing cardiopulmonary resuscitation (CPR) using carotid ultrasound imaging.

Current CPR guidelines require periodic interruption of chest compressions (~every 2 minutes) to perform pulse checks. However, these interruptions can reduce coronary perfusion pressure and negatively impact patient survival.

IntraCPR-Net addresses this limitation by enabling:

Continuous monitoring without interrupting chest compressions
Real-time vascular analysis using ultrasound
Objective ROSC classification based on carotid artery dynamics
⚙️ Pipeline Architecture

The system consists of three sequential modules:

1) Detection (YOLOv12-n)
Detects:
Carotid artery (CA)
Internal jugular vein (IJV)
Provides ROI for downstream processing
Optimized for real-time inference with low computational cost

2) Segmentation (SAM2 with auto-prompt)
Converts detection outputs into prompts
Performs instance-level vessel segmentation
Produces:
CA mask (red)
IJV mask (blue)
Robust to:
motion artifacts
compression-induced deformation

3) Classification (MLP-based temporal model)
Input features:
Carotid compressibility (CAC)
Jugular compressibility
Outputs:
ROSC vs Arrest classification
Supports:
frame-wise prediction
patient-level aggregation

📊 Datasets

🔹 RealCAC Dataset
Acquired during pulse-check (compression paused)
Relatively stable imaging conditions
Used for model development

🔹 IntraCPR-CAC Dataset
Acquired during ongoing chest compressions
Includes:
motion artifacts
tissue deformation
Used for evaluation under real clinical conditions

🔹 External Validation
Devices:
GE Healthcare
Mindray
Evaluates cross-device generalization
