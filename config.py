"""
Configuration file for Image Captioning project
All hyperparameters and paths in one place
"""

import os

# ============================================
# DATASET CONFIGURATION
# ============================================

# Number of training examples to use
NUM_EXAMPLES = 50000

# Dataset paths - UPDATE THESE TO MATCH YOUR SETUP
ANNOTATION_FILE = r"Dataset/annotations_trainval2017/annotations/captions_train2017.json"
IMAGE_FOLDER = r"Dataset/train2017/train2017/"

# Verify paths exist
if not os.path.exists(ANNOTATION_FILE):
    print(f"⚠️  WARNING: Annotation file not found: {ANNOTATION_FILE}")

if not os.path.exists(IMAGE_FOLDER):
    print(f"⚠️  WARNING: Image folder not found: {IMAGE_FOLDER}")

# ============================================
# MODEL HYPERPARAMETERS
# ============================================

# Transformer architecture
EMBED_DIM = 256        # Embedding dimension
DENSE_DIM = 512        # Feed-forward network dimension
NUM_HEADS = 6          # Number of attention heads

# Vocabulary settings
TOP_K = 10225           # Top K most frequent words
VOCAB_SIZE = TOP_K + 1  # Vocabulary size (10226) - auto-calculated
MAX_LENGTH = 51         # Maximum caption length

# CNN Feature Extractor
CNN_FEATURE_DIM = 2048  # InceptionV3 output dimension

# ============================================
# TRAINING CONFIGURATION
# ============================================

# Training parameters
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 15
LEARNING_RATE = 0.0005

# Data split
TRAIN_TEST_SPLIT = 0.2  # 20% for validation

# ============================================
# CHECKPOINT CONFIGURATION
# ============================================

# Checkpoint directory
CHECKPOINT_DIR = './checkpoints/train'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

# Save checkpoint every N epochs
CHECKPOINT_FREQUENCY = 1

# ============================================
# OUTPUT CONFIGURATION
# ============================================

# Where to save tokenizer
TOKENIZER_PATH = 'tokenizer.pkl'

# Where to save features (optional caching)
FEATURES_DIR = './features'

# ============================================
# INFERENCE CONFIGURATION
# ============================================

# Image preprocessing
IMAGE_SIZE = (299, 299)  # InceptionV3 input size

# Text-to-speech settings
TTS_RATE = 150    # Speech rate
TTS_VOLUME = 0.9  # Volume (0-1)

# ============================================
# DISPLAY CONFIGURATION
# ============================================

# Number of sample predictions to show during training
NUM_SAMPLES_TO_DISPLAY = 3

# ============================================
# HELPER FUNCTIONS
# ============================================


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("IMAGE CAPTIONING CONFIGURATION")
    print("=" * 60)
    print(f"\nDataset:")
    print(f"  Annotations: {ANNOTATION_FILE}")
    print(f"  Images: {IMAGE_FOLDER}")
    print(f"  Training examples: {NUM_EXAMPLES}")

    print(f"\nModel Architecture:")
    print(f"  Embedding dim: {EMBED_DIM}")
    print(f"  Dense dim: {DENSE_DIM}")
    print(f"  Attention heads: {NUM_HEADS}")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    print(f"  Max caption length: {MAX_LENGTH}")

    print(f"\nTraining:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()
