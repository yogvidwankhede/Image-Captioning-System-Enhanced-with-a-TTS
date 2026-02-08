"""
Training script for Image Captioning model
Handles full training loop with checkpoints and progress tracking
"""

import tensorflow as tf
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split

# Import project modules
from config import *
from data_loader import (
    load_captions_data,
    create_tokenizer,
    save_tokenizer,
)
from feature_extractor import (
    create_feature_extractor,
    extract_features,
    check_features_exist
)
from model import create_model


def loss_function(real, pred):
    """
    Masked sparse categorical crossentropy loss
    Ignores padding tokens in loss calculation
    
    Args:
        real: True token IDs
        pred: Predicted logits
        
    Returns:
        loss: Computed loss value
    """
    # Create mask to ignore padding tokens (0)
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    # Compute loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    loss_ = loss_object(real, pred)

    # Apply mask
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target, model, optimizer):
    """
    Single training step
    
    Args:
        img_tensor: Batch of image features
        target: Batch of target captions
        model: Caption model
        optimizer: Optimizer
        
    Returns:
        loss: Batch loss value
    """
    # Teacher forcing: feed correct previous word as input
    tar_inp = target[:, :-1]   # Input: "<start> A cat is..."
    tar_real = target[:, 1:]   # Target: "A cat is... <end>"

    with tf.GradientTape() as tape:
        predictions = model((img_tensor, tar_inp), training=True)
        loss = loss_function(tar_real, predictions)

    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_model(
    model,
    dataset,
    num_steps,
    epochs,
    optimizer,
    ckpt_manager,
    start_epoch=0
):
    """
    Main training loop
    
    Args:
        model: Caption model
        dataset: Training dataset
        num_steps: Steps per epoch
        epochs: Total epochs
        optimizer: Optimizer
        ckpt_manager: Checkpoint manager
        start_epoch: Starting epoch (for resuming)
        
    Returns:
        history: Training history
    """
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Total epochs: {epochs}")
    print(f"Steps per epoch: {num_steps}")
    print(f"Starting from epoch: {start_epoch + 1}")
    print("=" * 60)

    history = {
        'loss': [],
        'epoch_time': []
    }

    for epoch in range(start_epoch, epochs):
        start = time.time()
        total_loss = 0

        print(f"\n📊 Epoch {epoch + 1}/{epochs}")
        print("-" * 60)

        # Iterate through batches
        for batch, (img_tensor, target) in enumerate(dataset):
            batch_loss = train_step(img_tensor, target, model, optimizer)
            total_loss += batch_loss

            # Print progress every 100 batches
            if batch % 100 == 0:
                avg_loss = total_loss / (batch + 1)
                print(
                    f'  Batch {batch:4d}/{num_steps} | Loss: {batch_loss.numpy():.4f} | Avg: {avg_loss.numpy():.4f}')

        # Calculate epoch metrics
        avg_epoch_loss = total_loss / num_steps
        epoch_time = time.time() - start

        # Save checkpoint
        ckpt_path = ckpt_manager.save()

        # Store history
        history['loss'].append(avg_epoch_loss.numpy())
        history['epoch_time'].append(epoch_time)

        # Print epoch summary
        print("-" * 60)
        print(f'✅ Epoch {epoch + 1} Complete')
        print(f'   Average Loss: {avg_epoch_loss:.4f}')
        print(f'   Time: {epoch_time:.2f} sec ({epoch_time/60:.2f} min)')
        print(f'   Checkpoint: {ckpt_path}')
        print("=" * 60)

    print("\n🎉 Training Complete!")

    return history


def setup_training():
    """
    Setup complete training pipeline
    
    Returns:
        Tuple of (model, dataset, num_steps, optimizer, ckpt_manager, tokenizer)
    """
    print("🔧 Setting up training pipeline...\n")

    # Step 1: Load data
    print("1️⃣ Loading captions...")
    train_captions, img_name_vector = load_captions_data(
        ANNOTATION_FILE,
        IMAGE_FOLDER,
        NUM_EXAMPLES
    )

    # Step 2: Create tokenizer
    print("\n2️⃣ Creating tokenizer...")
    tokenizer, train_seqs, cap_vector, max_length = create_tokenizer(
        train_captions,
        top_k=TOP_K
    )

    # Save tokenizer
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    # Update config with actual values
    global MAX_LENGTH, VOCAB_SIZE
    MAX_LENGTH = max_length
    VOCAB_SIZE = len(tokenizer.word_index) + 1

    print(f"\n⚠️  IMPORTANT: Vocabulary size from tokenizer: {VOCAB_SIZE}")
    print(f"   Your config has: {TOP_K + 1}")
    if VOCAB_SIZE != (TOP_K + 1):
        print(f"   ⚠️  MISMATCH DETECTED! Using tokenizer value: {VOCAB_SIZE}")

    # Step 3: Extract features
    print("\n3️⃣ Checking image features...")
    if check_features_exist(img_name_vector):
        print("✅ Features already extracted!")
    else:
        print("📦 Extracting features (this will take a while)...")
        feature_extractor = create_feature_extractor()
        extract_features(img_name_vector, feature_extractor, batch_size=16)

    # Step 4: Train/val split
    print("\n4️⃣ Splitting data...")
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(
        img_name_vector,
        cap_vector,
        test_size=TRAIN_TEST_SPLIT,
        random_state=0
    )

    print(f"   Training: {len(img_name_train)} images")
    print(f"   Validation: {len(img_name_val)} images")

    # Step 5: Create dataset
    print("\n5️⃣ Creating TensorFlow dataset...")

    def load_data(img_name, cap):
        """Load features from .npy files"""
        img_tensor = np.load(img_name.decode('utf-8') + '.npy')
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load cached features
    dataset = dataset.map(
        lambda item1, item2: tf.numpy_function(
            load_data, [item1, item2], [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    num_steps = len(img_name_train) // BATCH_SIZE

    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps per epoch: {num_steps}")

    # Step 6: Create model
    print("\n6️⃣ Creating model...")
    model = create_model(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        embed_dim=EMBED_DIM,
        dense_dim=DENSE_DIM,
        num_heads=NUM_HEADS
    )

    # Step 7: Setup optimizer
    print("\n7️⃣ Setting up optimizer...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    print(f"   Optimizer: Adam")
    print(f"   Learning rate: {LEARNING_RATE}")

    # Step 8: Setup checkpoints
    print("\n8️⃣ Setting up checkpoints...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    ckpt = tf.train.Checkpoint(
        caption_model=model,
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        CHECKPOINT_DIR,
        max_to_keep=5
    )

    # Restore if checkpoint exists
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"   ✅ Restored from: {ckpt_manager.latest_checkpoint}")
        # Try to determine epoch from checkpoint name
        try:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        except:
            pass
    else:
        print(f"   Starting from scratch")

    print(f"   Checkpoint dir: {CHECKPOINT_DIR}")

    print("\n✅ Setup complete!\n")

    return model, dataset, num_steps, optimizer, ckpt_manager, tokenizer, start_epoch


def main():
    """
    Main training function
    """
    # Print configuration
    print_config()

    # Setup
    model, dataset, num_steps, optimizer, ckpt_manager, tokenizer, start_epoch = setup_training()

    # Train
    history = train_model(
        model=model,
        dataset=dataset,
        num_steps=num_steps,
        epochs=EPOCHS,
        optimizer=optimizer,
        ckpt_manager=ckpt_manager,
        start_epoch=start_epoch
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs: {len(history['loss'])}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Average epoch time: {np.mean(history['epoch_time']):.2f} sec")
    print(f"Total training time: {sum(history['epoch_time'])/3600:.2f} hours")
    print("=" * 60)

    print("\n✅ All done! Your model is ready for inference.")
    print(f"📁 Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"📁 Tokenizer saved as: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
