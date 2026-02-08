"""
Inference script for Image Captioning
Generates captions from images using trained model
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import *
from data_loader import load_tokenizer
from model import create_model


def load_and_preprocess_image(image_path):
    """
    Load and preprocess image for InceptionV3
    
    Args:
        image_path: Path to image file
        
    Returns:
        img: Preprocessed image tensor
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def evaluate_caption(image_tensor, feature_extractor, caption_model, tokenizer, max_length):
    """
    Generate caption for a single image
    
    Args:
        image_tensor: Preprocessed image tensor
        feature_extractor: InceptionV3 feature extractor
        caption_model: Trained caption model
        tokenizer: Fitted tokenizer
        max_length: Maximum caption length
        
    Returns:
        result: List of caption words
    """
    # Extract features
    temp_input = tf.expand_dims(image_tensor, 0)
    img_tensor_val = feature_extractor(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, 2048)
    )

    # Initialize with start token
    start_token = tokenizer.word_index.get('<start>', 3)
    dec_input = tf.expand_dims([start_token], 0)
    result = []
    last_word = ""

    # Generate caption word by word
    for i in range(max_length):
        predictions = caption_model(
            (img_tensor_val, dec_input), training=False)
        predictions = predictions[:, -1, :]

        predicted_id = tf.argmax(predictions, axis=-1)[0].numpy()

        # Check if token exists in vocabulary
        if predicted_id not in tokenizer.index_word:
            break

        predicted_word = tokenizer.index_word[predicted_id]

        # Stop if we hit end token
        if predicted_word == '<end>':
            return result

        # Stop if word repeats (simple loop detection)
        if predicted_word == last_word:
            break

        result.append(predicted_word)
        last_word = predicted_word

        # Update decoder input
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def generate_caption(
    image_path,
    feature_extractor,
    caption_model,
    tokenizer,
    max_length,
    display=True
):
    """
    Generate and display caption for an image
    
    Args:
        image_path: Path to image
        feature_extractor: InceptionV3 feature extractor
        caption_model: Trained caption model
        tokenizer: Fitted tokenizer
        max_length: Maximum caption length
        display: Whether to display image with caption
        
    Returns:
        caption: Generated caption string
    """
    # Load and preprocess image
    image_tensor = load_and_preprocess_image(image_path)

    # Generate caption
    result = evaluate_caption(
        image_tensor,
        feature_extractor,
        caption_model,
        tokenizer,
        max_length
    )

    # Clean up caption
    caption = ' '.join(result).replace(
        '<start>', '').replace('<end>', '').strip()

    if display:
        # Display image with caption
        image = Image.open(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(caption, fontsize=14, wrap=True)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return caption


def setup_inference(checkpoint_path=None):
    """
    Setup inference pipeline
    
    Args:
        checkpoint_path: Path to specific checkpoint (optional)
        
    Returns:
        Tuple of (feature_extractor, caption_model, tokenizer)
    """
    print("🔧 Setting up inference pipeline...")

    # Load tokenizer
    print("\n1️⃣ Loading tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # Update vocab size from tokenizer
    vocab_size = len(tokenizer.word_index) + 1
    print(f"   Vocabulary size: {vocab_size}")

    # Load feature extractor
    print("\n2️⃣ Loading InceptionV3 feature extractor...")
    image_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet'
    )
    feature_extractor = tf.keras.Model(
        image_model.input,
        image_model.layers[-1].output
    )
    print("   ✅ Feature extractor loaded")

    # Create caption model
    print("\n3️⃣ Creating caption model...")
    caption_model = create_model(
        vocab_size=vocab_size,
        max_length=MAX_LENGTH,
        embed_dim=EMBED_DIM,
        dense_dim=DENSE_DIM,
        num_heads=NUM_HEADS
    )

    # Initialize model with dummy data
    dummy_img = tf.zeros((1, 64, 2048))
    dummy_cap = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    _ = caption_model((dummy_img, dummy_cap), training=False)

    # Load checkpoint
    print("\n4️⃣ Loading checkpoint...")
    ckpt = tf.train.Checkpoint(caption_model=caption_model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        CHECKPOINT_DIR,
        max_to_keep=5
    )

    if checkpoint_path:
        ckpt.restore(checkpoint_path)
        print(f"   ✅ Restored from: {checkpoint_path}")
    elif ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"   ✅ Restored from: {ckpt_manager.latest_checkpoint}")
    else:
        print("   ⚠️  No checkpoint found! Model will use random weights.")

    print("\n✅ Inference setup complete!\n")

    return feature_extractor, caption_model, tokenizer


def main():
    """
    Main inference function - generate captions for sample images
    """
    # Setup
    feature_extractor, caption_model, tokenizer = setup_inference()

    # Example usage
    print("=" * 60)
    print("CAPTION GENERATION")
    print("=" * 60)

    # You can specify image paths here
    test_images = [
        # Add your test image paths here
        # Example: "path/to/image1.jpg"
    ]

    if not test_images:
        print("No test images specified.")
        print("To use this script:")
        print("1. Edit the test_images list in main()")
        print("2. Add paths to your test images")
        print("\nAlternatively, use the functions directly:")
        print("  caption = generate_caption(image_path, feature_extractor, caption_model, tokenizer, MAX_LENGTH)")
        return

    # Generate captions
    for i, image_path in enumerate(test_images, 1):
        print(f"\n{i}. Processing: {image_path}")
        try:
            caption = generate_caption(
                image_path,
                feature_extractor,
                caption_model,
                tokenizer,
                MAX_LENGTH,
                display=True
            )
            print(f"   Caption: {caption}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Caption generation complete!")


if __name__ == "__main__":
    main()
