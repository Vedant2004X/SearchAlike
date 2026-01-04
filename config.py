import sys
import os
from pathlib import Path
import torch
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# BASE DIRECTORY - AUTO-DETECT FOR EXE & SCRIPT
# ============================================================================

if getattr(sys, 'frozen', False):
    # Running as compiled executable (PyInstaller)
    BASE_DIR = Path(sys._MEIPASS)
    print(f"[CONFIG] Running from PyInstaller bundle: {BASE_DIR}")
else:
    # Running as script
    BASE_DIR = Path(__file__).resolve().parent
    print(f"[CONFIG] Running as script from: {BASE_DIR}")

# ============================================================================
# DATA DIRECTORIES
# ============================================================================

DATA_FOLDER = BASE_DIR / 'data'
IMAGES_FOLDER = DATA_FOLDER / 'images'
EMBEDDINGS_FOLDER = DATA_FOLDER / 'embeddings'
MODEL_FOLDER = DATA_FOLDER / 'model'

# Static directories (for web)
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'

# Checkpoint and logging directories
CHECKPOINT_FOLDER = BASE_DIR / 'checkpoints'
LOG_DIR = BASE_DIR / 'runs'
VIZ_OUTPUT_DIR = BASE_DIR / 'visualizations'

# ============================================================================
# ENSURE ALL DIRECTORIES EXIST
# ============================================================================

_REQUIRED_DIRS = [
    DATA_FOLDER,
    IMAGES_FOLDER,
    EMBEDDINGS_FOLDER,
    MODEL_FOLDER,
    STATIC_DIR,
    UPLOAD_FOLDER,
    CHECKPOINT_FOLDER,
    LOG_DIR,
    VIZ_OUTPUT_DIR
]

for folder in _REQUIRED_DIRS:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[CONFIG] Warning: Could not create {folder}: {e}")

print(f"[CONFIG] Data folder: {DATA_FOLDER}")
print(f"[CONFIG] Images folder: {IMAGES_FOLDER}")
print(f"[CONFIG] Embeddings folder: {EMBEDDINGS_FOLDER}")

# ============================================================================
# FEATURE FILES & INDICES
# ============================================================================

# Pretrained model files
EMBEDDINGS_FILE = EMBEDDINGS_FOLDER / 'features.npy'
IMAGE_PATHS_FILE = EMBEDDINGS_FOLDER / 'image_paths.pkl'
FAISS_INDEX_FILE = EMBEDDINGS_FOLDER / 'faiss_index.bin'

# Triplet model files
TRIPLET_EMBEDDINGS_FILE = EMBEDDINGS_FOLDER / 'triplet_features.npy'
TRIPLET_IMAGE_PATHS_FILE = EMBEDDINGS_FOLDER / 'triplet_image_paths.pkl'
TRIPLET_FAISS_INDEX = EMBEDDINGS_FOLDER / 'triplet_faiss_index.bin'

# PCA model
PCA_COMPONENTS = 128
PCA_MODEL_FILE = EMBEDDINGS_FOLDER / 'pca_model.pkl'

# ============================================================================
# FLASK CONFIGURATION
# ============================================================================

SECRET_KEY = os.environ.get('SECRET_KEY', 'triplet-network-secret-key-2024')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Flask deployment
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = False

# ============================================================================
# MODEL SELECTION - AUTO-DETECTION
# ============================================================================

TRIPLET_MODEL_PATH = CHECKPOINT_FOLDER / 'best_model.pth'


def should_use_triplet_model():
    """Check if trained triplet model exists and is valid"""
    if TRIPLET_MODEL_PATH.exists():
        try:
            # Try to load checkpoint to verify it's valid
            checkpoint = torch.load(TRIPLET_MODEL_PATH, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                print(f"[CONFIG] Found trained triplet model: {TRIPLET_MODEL_PATH}")
                return True
        except Exception as e:
            print(f"[CONFIG] Triplet model exists but is invalid: {e}")
            return False
    return False


USE_TRIPLET_MODEL = should_use_triplet_model()

# ============================================================================
# HARDWARE DETECTION
# ============================================================================

USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

print(f"[CONFIG] Using device: {DEVICE.upper()}")
if USE_GPU:
    print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[CONFIG] CUDA Version: {torch.version.cuda}")
else:
    print(f"[CONFIG] No GPU available - using CPU (slower)")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Pretrained Model Settings (fallback when no triplet model)
MODEL_NAME = 'resnet50'
FEATURE_DIM = 2048

# Triplet Network Settings
TRIPLET_BACKBONE = 'resnet50'
TRIPLET_EMBEDDING_DIM = 128
TRIPLET_MARGIN = 1.0

# Image Preprocessing
IMAGE_SIZE = (224, 224)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

TOP_K = 10
SIMILARITY_THRESHOLD = 0.0

# FAISS settings
USE_FAISS = True
FAISS_INDEX_TYPE = 'IndexFlatIP'

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training hyperparameters
TRAIN_BATCH_SIZE = 32 if USE_GPU else 16
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 0.0001
TRAIN_WEIGHT_DECAY = 1e-4

# Triplet mining settings
TRIPLET_MINING_MODE = 'online'  # 'online' or 'offline'
TRIPLET_LOSS_TYPE = 'hardest'  # 'hardest', 'semi-hard', or 'all'

# Data split
VAL_SPLIT = 0.2

# Training optimization
EARLY_STOPPING_PATIENCE = 15
SAVE_CHECKPOINT_EVERY = 5

# Backbone training
FREEZE_BACKBONE_INITIALLY = False
UNFREEZE_AFTER_EPOCHS = 10

# Projection head
PROJECTION_HIDDEN_DIM = 512
PROJECTION_DROPOUT = 0.5

# Data loader settings
NUM_WORKERS = 4 if USE_GPU else 2
PIN_MEMORY = True if USE_GPU else False

# Triplet dataset
TRIPLETS_PER_ANCHOR = 5

# Balanced batch sampler
BALANCED_BATCH_P = 8
BALANCED_BATCH_K = 4

# Optimization features
ENABLE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_VALUE = 1.0
ENABLE_AUGMENTATION = True
ENABLE_MIXED_PRECISION = False

# ============================================================================
# LABEL GENERATION SETTINGS
# ============================================================================

FILENAME_DELIMITER = '_'
FILENAME_LABEL_POSITION = 0
SYNTHETIC_N_CLUSTERS = 20

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOG_LEVEL = 'INFO'
ENABLE_TENSORBOARD = True
VERBOSE_LOGGING = False

# Visualization settings
VIZ_MAX_CLASSES = 20

# ============================================================================
# UPLOAD & CLEANUP
# ============================================================================

UPLOAD_CLEANUP_HOURS = 24

# ============================================================================
# AUTO-TRAINING SETTINGS
# ============================================================================

# Minimum number of images required for training
MIN_IMAGES_FOR_TRAINING = 50

# Auto-train on first run if model doesn't exist
AUTO_TRAIN_ON_FIRST_RUN = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_config():
    """Get current model configuration as dictionary"""
    return {
        'use_triplet': USE_TRIPLET_MODEL,
        'model_name': MODEL_NAME if not USE_TRIPLET_MODEL else 'triplet',
        'backbone': TRIPLET_BACKBONE if USE_TRIPLET_MODEL else MODEL_NAME,
        'embedding_dim': TRIPLET_EMBEDDING_DIM if USE_TRIPLET_MODEL else FEATURE_DIM,
        'device': DEVICE,
        'use_gpu': USE_GPU,
        'model_path': str(TRIPLET_MODEL_PATH) if USE_TRIPLET_MODEL else 'pretrained'
    }


def check_training_requirements():
    """
    Check if system is ready for training

    Returns:
        tuple: (ready: bool, messages: list)
    """
    issues = []

    # Check if images exist
    if not IMAGES_FOLDER.exists():
        issues.append(f"   ❌ Images folder not found: {IMAGES_FOLDER}")
        return False, issues

    # Count images in all subfolders
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    for ext in valid_extensions:
        image_files.extend(list(IMAGES_FOLDER.rglob(f'*{ext}')))
        image_files.extend(list(IMAGES_FOLDER.rglob(f'*{ext.upper()}')))

    # Remove duplicates
    image_files = list(set(image_files))
    num_images = len(image_files)

    if num_images == 0:
        issues.append(f"   ❌ No images found in {IMAGES_FOLDER}")
        issues.append(f"      Please organize images in subfolders by category")
        return False, issues

    if num_images < MIN_IMAGES_FOR_TRAINING:
        issues.append(
            f"   ⚠️  Only {num_images} images found. "
            f"Recommend {MIN_IMAGES_FOR_TRAINING}+ for good results"
        )
        return False, issues

    # Check for multiple classes
    subfolders = [f for f in IMAGES_FOLDER.iterdir() if f.is_dir()]
    if len(subfolders) < 2:
        issues.append(f"   ⚠️  Only 1 category found. Need at least 2 for training")
        return False, issues

    return True, [f"   ✅ Found {num_images} images in {len(subfolders)} categories"]


def print_startup_info():
    """Print detailed startup information"""
    print("\n" + "=" * 70)
    print(" 🔍 IMAGE SIMILARITY SEARCH - SEARCHALIKE")
    print("=" * 70)

    print(f"\n📊 Model Status:")
    if USE_TRIPLET_MODEL:
        print(f"   ✅ Using Trained Triplet Model")
        print(f"      Model: {TRIPLET_MODEL_PATH}")
        print(f"      Embedding Dim: {TRIPLET_EMBEDDING_DIM}")
    else:
        print(f"   ℹ️  No trained model found")
        print(f"      Using pretrained {MODEL_NAME} (fallback)")
        if AUTO_TRAIN_ON_FIRST_RUN:
            ready, messages = check_training_requirements()
            if ready:
                print(f"      Will auto-train on startup")
            else:
                print(f"      Cannot auto-train - requirements not met:")
                for msg in messages:
                    print(f"      {msg}")

    print(f"\n💻 Hardware:")
    print(f"   Device: {DEVICE.upper()}")
    if USE_GPU:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
    else:
        print(f"   ⚠️  CPU Mode (slow) - GPU recommended for faster processing")

    print(f"\n📁 Paths:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Images: {IMAGES_FOLDER}")
    print(f"   Embeddings: {EMBEDDINGS_FOLDER}")
    print(f"   Checkpoints: {CHECKPOINT_FOLDER}")
    print(f"   Uploads: {UPLOAD_FOLDER}")

    print(f"\n⚙️  Configuration:")
    print(f"   Max file size: {MAX_CONTENT_LENGTH / (1024 * 1024):.0f}MB")
    print(f"   Top-K results: {TOP_K}")
    print(f"   Auto-train: {'Enabled' if AUTO_TRAIN_ON_FIRST_RUN else 'Disabled'}")
    print(f"   Min images for training: {MIN_IMAGES_FOR_TRAINING}")

    print("=" * 70 + "\n")


def validate_paths():
    """Validate all required paths are accessible"""
    print("\n[CONFIG] Validating paths...")

    paths_to_check = {
        'Base': BASE_DIR,
        'Data': DATA_FOLDER,
        'Images': IMAGES_FOLDER,
        'Embeddings': EMBEDDINGS_FOLDER,
        'Uploads': UPLOAD_FOLDER,
        'Checkpoints': CHECKPOINT_FOLDER,
    }

    all_valid = True
    for name, path in paths_to_check.items():
        if path.exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (does not exist)")
            all_valid = False

    return all_valid


def get_image_count():
    """Get total number of images in dataset"""
    if not IMAGES_FOLDER.exists():
        return 0

    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    count = 0

    for ext in valid_extensions:
        count += len(list(IMAGES_FOLDER.rglob(f'*{ext}')))
        count += len(list(IMAGES_FOLDER.rglob(f'*{ext.upper()}')))

    return len(set(list(IMAGES_FOLDER.rglob('*.*'))))  # Return unique files


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Paths
    'BASE_DIR',
    'DATA_FOLDER',
    'IMAGES_FOLDER',
    'EMBEDDINGS_FOLDER',
    'MODEL_FOLDER',
    'STATIC_DIR',
    'UPLOAD_FOLDER',
    'CHECKPOINT_FOLDER',
    'LOG_DIR',
    'VIZ_OUTPUT_DIR',

    # Files
    'EMBEDDINGS_FILE',
    'IMAGE_PATHS_FILE',
    'FAISS_INDEX_FILE',
    'TRIPLET_EMBEDDINGS_FILE',
    'TRIPLET_IMAGE_PATHS_FILE',
    'TRIPLET_FAISS_INDEX',
    'PCA_MODEL_FILE',
    'TRIPLET_MODEL_PATH',

    # Flask
    'SECRET_KEY',
    'MAX_CONTENT_LENGTH',
    'ALLOWED_EXTENSIONS',
    'FLASK_HOST',
    'FLASK_PORT',
    'FLASK_DEBUG',

    # Models
    'USE_TRIPLET_MODEL',
    'MODEL_NAME',
    'FEATURE_DIM',
    'TRIPLET_BACKBONE',
    'TRIPLET_EMBEDDING_DIM',
    'TRIPLET_MARGIN',
    'IMAGE_SIZE',
    'IMAGENET_MEAN',
    'IMAGENET_STD',

    # Search
    'TOP_K',
    'SIMILARITY_THRESHOLD',
    'USE_FAISS',
    'FAISS_INDEX_TYPE',

    # Training
    'TRAIN_BATCH_SIZE',
    'TRAIN_EPOCHS',
    'TRAIN_LEARNING_RATE',
    'TRAIN_WEIGHT_DECAY',
    'TRIPLET_MINING_MODE',
    'TRIPLET_LOSS_TYPE',
    'VAL_SPLIT',
    'EARLY_STOPPING_PATIENCE',
    'SAVE_CHECKPOINT_EVERY',
    'FREEZE_BACKBONE_INITIALLY',
    'NUM_WORKERS',
    'PIN_MEMORY',
    'TRIPLETS_PER_ANCHOR',
    'PCA_COMPONENTS',

    # Hardware
    'USE_GPU',
    'DEVICE',

    # Settings
    'AUTO_TRAIN_ON_FIRST_RUN',
    'MIN_IMAGES_FOR_TRAINING',
    'UPLOAD_CLEANUP_HOURS',
    'LOG_LEVEL',
    'ENABLE_TENSORBOARD',

    # Functions
    'get_model_config',
    'check_training_requirements',
    'print_startup_info',
    'validate_paths',
    'get_image_count',
    'should_use_triplet_model',
]
