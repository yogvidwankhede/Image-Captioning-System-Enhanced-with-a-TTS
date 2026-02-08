# Image Captioning with Transformer

A complete implementation of an image captioning system using Transformer architecture with InceptionV3 features and text-to-speech capabilities.

## 📁 Project Structure

```
Image_Captioning_Project/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Data loading and tokenization
├── feature_extractor.py   # CNN feature extraction (InceptionV3)
├── model.py               # Transformer model architecture
├── train.py               # Training pipeline
├── inference.py           # Caption generation
├── main.py                # Streamlit web interface
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── Dataset/               # Dataset folder (you need to download)
│   ├── train2017/
│   └── annotations_trainval2017/
│
├── checkpoints/           # Model checkpoints (created during training)
│   └── train/
│
└── tokenizer.pkl          # Saved tokenizer (created during training)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download MS COCO 2017:
- **Training images**: http://images.cocodataset.org/zips/train2017.zip
- **Annotations**: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Extract to:
```
Dataset/
├── train2017/train2017/
└── annotations_trainval2017/annotations/
```

### 3. Configure Paths

Edit `config.py` to match your setup:

```python
ANNOTATION_FILE = r"Dataset/annotations_trainval2017/annotations/captions_train2017.json"
IMAGE_FOLDER = r"Dataset/train2017/train2017/"
NUM_EXAMPLES = 30000  # Adjust based on your resources
```

### 4. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess captions
- Create tokenizer
- Extract image features
- Train the model
- Save checkpoints

**Training time:** 4-12 hours depending on GPU and dataset size

### 5. Generate Captions

```bash
python inference.py
```

Or use the Streamlit web interface:

```bash
streamlit run main.py
```

## 📚 Module Overview

### `config.py`
Central configuration file containing:
- Dataset paths
- Model hyperparameters (EMBED_DIM, DENSE_DIM, NUM_HEADS, etc.)
- Training settings (BATCH_SIZE, EPOCHS, LEARNING_RATE)
- File paths for checkpoints and tokenizer

### `data_loader.py`
Handles data loading and preprocessing:
- `load_captions_data()`: Load COCO captions
- `create_tokenizer()`: Build vocabulary and tokenize captions
- `save_tokenizer()`: Save tokenizer for inference
- `preprocess_image()`: Prepare images for InceptionV3

### `feature_extractor.py`
CNN feature extraction:
- `create_feature_extractor()`: Load InceptionV3 model
- `extract_features()`: Extract and cache image features as .npy files
- `load_cached_features()`: Load pre-extracted features

### `model.py`
Transformer architecture:
- `TransformerEncoderBlock`: Processes image features
- `TransformerDecoderBlock`: Generates captions
- `PositionalEmbedding`: Adds position information
- `ImageCaptioningModel`: Complete end-to-end model

### `train.py`
Training pipeline:
- Complete training loop
- Checkpoint management
- Loss tracking
- Automatic feature extraction

### `inference.py`
Caption generation:
- Load trained model
- Generate captions for new images
- Display results

## 🎯 Usage Examples

### Training

```python
# Basic training with default config
python train.py

# To modify settings, edit config.py first
```

### Inference (Python)

```python
from inference import setup_inference, generate_caption
from config import MAX_LENGTH

# Setup
feature_extractor, caption_model, tokenizer = setup_inference()

# Generate caption
caption = generate_caption(
    "path/to/image.jpg",
    feature_extractor,
    caption_model,
    tokenizer,
    MAX_LENGTH,
    display=True
)

print(f"Caption: {caption}")
```

### Streamlit Interface

```bash
streamlit run main.py
```

Features:
- Upload images via drag-and-drop
- Generate captions with one click
- Adjust temperature and sampling parameters
- Text-to-speech output
- Generate multiple caption variations

## ⚙️ Hyperparameters

Key hyperparameters in `config.py`:

```python
# Model Architecture
EMBED_DIM = 256        # Embedding dimension
DENSE_DIM = 512        # Feed-forward dimension
NUM_HEADS = 6          # Attention heads

# Vocabulary
TOP_K = 5000          # Top K most frequent words
VOCAB_SIZE = 5001     # Vocabulary size

# Training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
```

## 🔧 Customization

### Change Dataset Size

Edit in `config.py`:
```python
NUM_EXAMPLES = 10000  # Use 10k images instead of 30k
```

### Adjust Model Size

```python
EMBED_DIM = 512       # Larger model
DENSE_DIM = 1024
NUM_HEADS = 8
```

### Modify Vocabulary

```python
TOP_K = 10000        # Larger vocabulary
```

**Important:** After changing vocabulary size, you must retrain from scratch!

## 🐛 Troubleshooting

### Vocabulary Mismatch Error

**Problem:** Getting gibberish captions like "hits hits hits"

**Cause:** Model's VOCAB_SIZE doesn't match tokenizer

**Solution:**
```python
# In config.py, ensure:
VOCAB_SIZE = TOP_K + 1  # Must match tokenizer

# If you changed TOP_K, retrain the model completely
```

### Out of Memory

**Problem:** GPU runs out of memory during training

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 32  # or 16

# Reduce dataset size
NUM_EXAMPLES = 10000  # instead of 30000

# Reduce model size
EMBED_DIM = 128
DENSE_DIM = 256
```

### Slow Training

**Solutions:**
1. Use GPU (10x faster than CPU)
2. Reduce NUM_EXAMPLES
3. Features are cached as .npy files - subsequent runs are faster
4. Use mixed precision training (advanced)

### Checkpoint Not Loading

**Problem:** Model doesn't improve, checkpoint not loading

**Check:**
```python
# Verify checkpoint directory exists
ls checkpoints/train/

# Check latest checkpoint
python -c "import tensorflow as tf; print(tf.train.latest_checkpoint('./checkpoints/train'))"
```

## 📊 Expected Results

### Good Caption Examples:
- "a woman standing in a kitchen with yellow walls"
- "a dog running through a field of grass"
- "a person sitting on a bench near the ocean"

### Training Metrics:
- Initial loss: ~4.0
- Final loss: ~1.5-2.0 (after 20 epochs)
- Training time: ~20-30 seconds per epoch (GPU)

## 🚨 Critical Notes

### 1. Vocabulary Size Must Match

The **most common error** is vocabulary mismatch:

```python
# These MUST match:
VOCAB_SIZE in config.py == len(tokenizer.word_index) + 1
```

If they don't match, you'll get random/repeated words.

### 2. Retrain After Vocabulary Changes

If you change `TOP_K` or `VOCAB_SIZE`:
1. Delete old checkpoints
2. Delete tokenizer.pkl
3. Retrain from scratch

### 3. Feature Caching

Features are cached as `.npy` files next to images:
- Saves time on subsequent runs
- But takes up disk space
- Can delete if needed (will re-extract)

## 🎓 Architecture Details

### Encoder-Decoder with Attention

```
Input Image (299x299x3)
    ↓
InceptionV3 (pretrained)
    ↓
Features (64x2048)
    ↓
Transformer Encoder
    ↓
Attended Features
    ↓ (cross-attention)
Transformer Decoder ← Previous Words
    ↓
Predicted Next Word
```

### Key Components:

1. **InceptionV3**: Extract visual features
2. **Transformer Encoder**: Process image features with self-attention
3. **Transformer Decoder**: Generate captions word-by-word
4. **Cross-Attention**: Decoder attends to image features
5. **Positional Encoding**: Add position information to tokens

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)
- [MS COCO Dataset](https://cocodataset.org/)

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- MS COCO dataset creators
- TensorFlow team
- Transformer architecture researchers

---

## 💡 Tips

1. **Start small**: Train on 5-10k images first to verify everything works
2. **Monitor training**: Watch loss - should decrease steadily
3. **Test during training**: Generate sample captions after each epoch
4. **Save tokenizer**: Don't forget to save tokenizer.pkl!
5. **Backup checkpoints**: Keep backups of good checkpoints

## 🔗 Quick Links

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MS COCO Download](https://cocodataset.org/#download)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Need help?** Check the troubleshooting section or review the code comments in each module.