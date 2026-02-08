"""
Data loading and preprocessing for Image Captioning
Handles COCO dataset loading, caption processing, and tokenization
"""

import json
import os
import tensorflow as tf
from sklearn.utils import shuffle
import pickle


def load_captions_data(annotation_file, image_folder, num_examples):
    """
    Load captions from COCO format JSON file
    
    Args:
        annotation_file: Path to captions JSON file
        image_folder: Path to folder containing images
        num_examples: Number of examples to load
        
    Returns:
        train_captions: List of caption strings
        img_name_vector: List of image paths
    """
    print(f"Loading captions from: {annotation_file}")

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_name_vector = []

    # Process each annotation
    for annot in annotations['annotations']:
        # Add start and end tokens
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']

        # Build full image path
        full_coco_image_path = os.path.join(
            image_folder, '%012d.jpg' % (image_id))

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    print(f"Total captions loaded: {len(all_captions)}")

    # Shuffle for randomness
    all_captions, all_img_name_vector = shuffle(
        all_captions, all_img_name_vector, random_state=1)

    # Select subset
    train_captions = all_captions[:num_examples]
    img_name_vector = all_img_name_vector[:num_examples]

    print(f"Using {len(train_captions)} captions for training")

    return train_captions, img_name_vector


def create_tokenizer(train_captions, top_k=5000):
    """
    Create and fit tokenizer on training captions
    
    Args:
        train_captions: List of caption strings
        top_k: Keep only top K most frequent words
        
    Returns:
        tokenizer: Fitted Keras tokenizer
        train_seqs: Tokenized sequences
        cap_vector: Padded caption vectors
        max_length: Maximum caption length
    """
    print(f"Creating tokenizer with top {top_k} words...")

    # Create tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=top_k,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ '
    )

    # Fit on captions
    tokenizer.fit_on_texts(train_captions)

    # Convert captions to sequences
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Add padding token
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Pad sequences to same length
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    # Calculate max length
    max_length = cap_vector.shape[1]

    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Maximum caption length: {max_length}")
    print(f"Number of sequences: {len(train_seqs)}")

    # Show some statistics
    print(f"\nTokenizer statistics:")
    print(f"  Total unique words: {len(tokenizer.word_index)}")
    print(f"  Words kept (top_k): {top_k}")
    print(
        f"  OOV token: <unk> (index: {tokenizer.word_index.get('<unk>', 'N/A')})")
    print(
        f"  Start token: <start> (index: {tokenizer.word_index.get('<start>', 'N/A')})")
    print(
        f"  End token: <end> (index: {tokenizer.word_index.get('<end>', 'N/A')})")

    return tokenizer, train_seqs, cap_vector, max_length


def save_tokenizer(tokenizer, filepath='tokenizer.pkl'):
    """
    Save tokenizer to pickle file
    
    Args:
        tokenizer: Keras tokenizer object
        filepath: Where to save the tokenizer
    """
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved to: {filepath}")


def load_tokenizer(filepath='tokenizer.pkl'):
    """
    Load tokenizer from pickle file
    
    Args:
        filepath: Path to tokenizer pickle file
        
    Returns:
        tokenizer: Loaded tokenizer object
    """
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✅ Tokenizer loaded from: {filepath}")
    return tokenizer


def preprocess_image(image_path):
    """
    Load and preprocess image for InceptionV3
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image tensor
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def create_dataset(img_name_vector, cap_vector, batch_size, buffer_size):
    """
    Create TensorFlow dataset for training
    
    Args:
        img_name_vector: List of image paths
        cap_vector: Padded caption vectors
        batch_size: Batch size
        buffer_size: Shuffle buffer size
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((img_name_vector, cap_vector))

    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    """
    Test the data loading functions
    """
    from config import *

    print("Testing data loading...")
    print(f"Annotation file: {ANNOTATION_FILE}")
    print(f"Image folder: {IMAGE_FOLDER}")

    # Load captions
    train_captions, img_name_vector = load_captions_data(
        ANNOTATION_FILE, IMAGE_FOLDER, NUM_EXAMPLES
    )

    print(f"\nSample captions:")
    for i in range(3):
        print(f"  {i+1}. {train_captions[i]}")
        print(f"     Image: {img_name_vector[i]}")

    # Create tokenizer
    tokenizer, train_seqs, cap_vector, max_length = create_tokenizer(
        train_captions, top_k=TOP_K
    )

    print(f"\nSample tokenized sequence:")
    print(f"  Caption: {train_captions[0]}")
    print(f"  Sequence: {train_seqs[0][:10]}...")
    print(f"  Padded: {cap_vector[0][:10]}...")

    # Save tokenizer
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    print("\n✅ Data loading test complete!")
