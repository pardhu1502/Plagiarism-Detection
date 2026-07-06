"""
Global configuration for the plagiarism detection system.
Modify these values according to your requirements.
"""


PDF_DPI = 300



PATCH_SIZE = 224

STRIDE = 112


MAX_PAGE_SIZE = 2480

USE_GRAYSCALE = False

HIST_EQUALIZATION = True

BLUR_KERNEL = (3, 3)



MODEL_NAME = "facebook/dinov2-base"

BATCH_SIZE = 64

DEVICE = "cuda"



SIMILARITY_THRESHOLD = 0.85

TOP_K = 5




ALPHA = 0.45

  
HEATMAP_THRESHOLD = 0.75




REPORT_NAME = "Plagiarism_Report.pdf"



RANDOM_SEED = 42