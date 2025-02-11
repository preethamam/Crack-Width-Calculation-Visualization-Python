IMAGE_PATH = "images/crack.bmp" # Image path and filename
METHOD = "afmm"  # "zhang" or "lee" or "medial_axis" or "afmm"
PRUNE_THRESHOLD = 0.1 # % Prune threshold for skeletonization
PIXEL_SCALE = 1 # Scale of the pixels (float or int)
MOV_WINDOW_SIZE = 5 # Moving window size for smoothing
MOV_WINDOW_TYPE = "mean" # Moving window type for smoothing ("mean" or "median")
SKEL_ORIENT_BLOCK_SIZE = 5 # Block size for skeleton orientation calculation
PRINT_RESULTS = True # Flag to print results