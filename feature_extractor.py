"""
Feature extraction using InceptionV3
Extracts image features and saves them as .npy files for efficient training
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os


def create_feature_extractor():
    """
    Create InceptionV3 feature extraction model
    
    Returns:
        feature_extractor: Keras model for feature extraction
    """
    print("Loading InceptionV3 model...")

    # Load pretrained InceptionV3 (without top classification layer)
    image_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet'
    )

    # Extract features from last convolutional layer
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    feature_extractor = tf.keras.Model(new_input, hidden_layer)

    print("✅ InceptionV3 feature extractor loaded")
    print(f"   Input shape: {feature_extractor.input_shape}")
    print(f"   Output shape: {feature_extractor.output_shape}")

    return feature_extractor


def load_image(image_path):
    """
    Load and preprocess single image for InceptionV3
    
    Args:
        image_path: Path to image
        
    Returns:
        img: Preprocessed image tensor
        image_path: Original image path
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    # InceptionV3 expects inputs in range [-1, 1]
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def extract_features(img_name_vector, feature_extractor, batch_size=16):
    """
    Extract features from all images and save as .npy files
    
    Args:
        img_name_vector: List of image paths
        feature_extractor: InceptionV3 model
        batch_size: Batch size for processing
    """
    # Get unique images (one image may have multiple captions)
    unique_images = sorted(set(img_name_vector))

    print(f"Total images: {len(img_name_vector)}")
    print(f"Unique images: {len(unique_images)}")
    print(f"Starting feature extraction with batch_size={batch_size}...")

    # Create dataset for efficient processing
    image_dataset = tf.data.Dataset.from_tensor_slices(unique_images)
    image_dataset = image_dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size)

    # Process batches with progress bar
    for img, path in tqdm(image_dataset, desc="Extracting features"):
        # Extract features
        batch_features = feature_extractor(img)
        batch_features = tf.reshape(
            batch_features,
            (batch_features.shape[0], -1, 2048)
        )

        # Save each feature as .npy file (same name as image)
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    print("✅ Feature extraction complete!")
    print(f"   Features saved as .npy files alongside images")


def load_cached_features(image_path):
    """
    Load pre-extracted features from .npy file
    
    Args:
        image_path: Path to image (will look for .npy file)
        
    Returns:
        features: Loaded feature tensor
    """
    npy_path = image_path + '.npy'
    if os.path.exists(npy_path):
        return np.load(npy_path)
    else:
        raise FileNotFoundError(f"Feature file not found: {npy_path}")


def check_features_exist(img_name_vector):
    """
    Check if features have already been extracted
    
    Args:
        img_name_vector: List of image paths
        
    Returns:
        bool: True if all features exist, False otherwise
    """
    unique_images = set(img_name_vector)

    for img_path in unique_images:
        npy_path = img_path + '.npy'
        if not os.path.exists(npy_path):
            return False

    return True


def map_func(img_name, cap):
    """
    Map function to load cached features for TensorFlow dataset
    
    Args:
        img_name: Image path tensor
        cap: Caption tensor
        
    Returns:
        img_tensor: Loaded feature tensor
        cap: Caption tensor
    """
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


if __name__ == "__main__":
    """
    Test feature extraction
    """
    from config import *
    from data_loader import load_captions_data

    print("Testing feature extraction...")

    # Load a small sample
    print(f"\nLoading {min(100, NUM_EXAMPLES)} images for testing...")
    train_captions, img_name_vector = load_captions_data(
        ANNOTATION_FILE, IMAGE_FOLDER, min(100, NUM_EXAMPLES)
    )

    # Create feature extractor
    feature_extractor = create_feature_extractor()

    # Check if features already exist
    if check_features_exist(img_name_vector):
        print("\n✅ Features already exist! Skipping extraction.")

        # Test loading
        sample_img = img_name_vector[0]
        features = load_cached_features(sample_img)
        print(f"\nSample feature shape: {features.shape}")
        print(f"Expected shape: (64, 2048) or similar")
    else:
        print("\n📦 Extracting features (this may take a while)...")
        extract_features(img_name_vector, feature_extractor, batch_size=16)

        # Test loading
        sample_img = img_name_vector[0]
        features = load_cached_features(sample_img)
        print(f"\n✅ Feature extraction test complete!")
        print(f"Sample feature shape: {features.shape}")
