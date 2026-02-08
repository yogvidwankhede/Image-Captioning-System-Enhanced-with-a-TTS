"""
Diagnostic script to check your image captioning model and tokenizer
Run this to identify issues with your setup
"""

import os
import tensorflow as tf
import pickle
import numpy as np

print("=" * 60)
print("IMAGE CAPTIONING MODEL DIAGNOSTICS")
print("=" * 60)

# 1. Check tokenizer
print("\n1. CHECKING TOKENIZER...")
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("✓ Tokenizer loaded successfully")
    print(f"   Vocabulary size: {len(tokenizer.word_index)}")
    print(f"   Word index size: {len(tokenizer.word_index)}")
    print(f"   Index word size: {len(tokenizer.index_word)}")

    # Check special tokens
    print("\n   Special tokens:")
    special_tokens = ['<start>', '<end>', '<pad>', '<unk>']
    for token in special_tokens:
        if token in tokenizer.word_index:
            idx = tokenizer.word_index[token]
            print(f"   ✓ {token:8s} -> index {idx}")
        else:
            print(f"   ✗ {token:8s} NOT FOUND")

    # Show some example words
    print("\n   Sample vocabulary (first 20 words):")
    for word, idx in list(tokenizer.word_index.items())[:20]:
        print(f"   {word:15s} -> {idx}")

except FileNotFoundError:
    print("✗ tokenizer.pkl not found!")
    print("  Make sure you've saved the tokenizer from your training notebook")
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")

# 2. Check checkpoints
print("\n2. CHECKING MODEL CHECKPOINTS...")
checkpoint_dir = "./checkpoints/train"
if os.path.exists(checkpoint_dir):
    print(f"✓ Checkpoint directory found: {checkpoint_dir}")
    files = os.listdir(checkpoint_dir)
    print(f"   Files in checkpoint directory:")
    for f in files:
        print(f"   - {f}")

    # Try to load checkpoint
    try:
        ckpt_manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(), checkpoint_dir, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            print(f"\n✓ Latest checkpoint: {ckpt_manager.latest_checkpoint}")
        else:
            print("\n✗ No checkpoint files found")
    except Exception as e:
        print(f"\n✗ Error checking checkpoints: {e}")
else:
    print(f"✗ Checkpoint directory not found: {checkpoint_dir}")

# 3. Test model initialization
print("\n3. TESTING MODEL INITIALIZATION...")
try:
    from tensorflow.keras import layers

    class TransformerEncoderBlock(layers.Layer):
        def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.dense_dim = dense_dim
            self.num_heads = num_heads
            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim)
            self.dense_proj = tf.keras.Sequential([
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim)
            ])
            self.layernorm_1 = layers.LayerNormalization()
            self.layernorm_2 = layers.LayerNormalization()

        def call(self, inputs, training, mask=None):
            inputs = self.layernorm_1(inputs)
            inputs = self.dense_proj(inputs)
            attention_output = self.attention(
                query=inputs, value=inputs, key=inputs, training=training)
            proj_input = self.layernorm_2(inputs + attention_output)
            return proj_input

    class PositionalEmbedding(layers.Layer):
        def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.token_embeddings = layers.Embedding(
                input_dim=vocab_size, output_dim=embed_dim)
            self.position_embeddings = layers.Embedding(
                input_dim=sequence_length, output_dim=embed_dim)
            self.sequence_length = sequence_length
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim

        def call(self, inputs):
            length = tf.shape(inputs)[-1]
            positions = tf.range(start=0, limit=length, delta=1)
            embedded_tokens = self.token_embeddings(inputs)
            embedded_positions = self.position_embeddings(positions)
            return embedded_tokens + embedded_positions

    class TransformerDecoderBlock(layers.Layer):
        def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.ff_dim = ff_dim
            self.num_heads = num_heads
            self.attention_1 = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim)
            self.attention_2 = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim)
            self.dense_proj = tf.keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim)
            ])
            self.layernorm_1 = layers.LayerNormalization()
            self.layernorm_2 = layers.LayerNormalization()
            self.layernorm_3 = layers.LayerNormalization()

        def call(self, inputs, encoder_outputs, training, mask=None):
            attention_output_1 = self.attention_1(
                query=inputs, value=inputs, key=inputs, training=training)
            out_1 = self.layernorm_1(inputs + attention_output_1)
            attention_output_2 = self.attention_2(
                query=out_1, value=encoder_outputs, key=encoder_outputs, training=training)
            out_2 = self.layernorm_2(out_1 + attention_output_2)
            proj_output = self.dense_proj(out_2)
            return self.layernorm_3(out_2 + proj_output)

    class ImageCaptioningModel(tf.keras.Model):
        def __init__(self, cnn_feature_dim, embed_dim, dense_dim, num_heads, vocab_size, max_len):
            super().__init__()
            self.encoder = TransformerEncoderBlock(
                embed_dim, dense_dim, num_heads)
            self.pos_embedding = PositionalEmbedding(
                max_len, vocab_size, embed_dim)
            self.decoder = TransformerDecoderBlock(
                embed_dim, dense_dim, num_heads)
            self.score = layers.Dense(vocab_size)

        def call(self, inputs, training=False):
            img_features, captions = inputs
            encoded_img = self.encoder(img_features, training)
            embedded_captions = self.pos_embedding(captions)
            decoded_output = self.decoder(
                embedded_captions, encoded_img, training)
            preds = self.score(decoded_output)
            return preds

    # Try to create model
    EMBED_DIM = 256
    DENSE_DIM = 512
    NUM_HEADS = 6
    VOCAB_SIZE = 8256
    MAX_LENGTH = 51

    model = ImageCaptioningModel(
        cnn_feature_dim=2048,
        embed_dim=EMBED_DIM,
        dense_dim=DENSE_DIM,
        num_heads=NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LENGTH
    )

    # Initialize with dummy data
    dummy_img = tf.zeros((1, 64, 2048))
    dummy_cap = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    output = model((dummy_img, dummy_cap), training=False)

    print("✓ Model created successfully")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected shape: (1, {MAX_LENGTH}, {VOCAB_SIZE})")

    # Check if shapes match
    if output.shape == (1, MAX_LENGTH, VOCAB_SIZE):
        print("   ✓ Output shape is correct!")
    else:
        print("   ✗ Output shape mismatch!")

except Exception as e:
    print(f"✗ Error creating model: {e}")
    import traceback
    traceback.print_exc()

# 4. Test prediction logic
print("\n4. TESTING PREDICTION LOGIC...")
try:
    # Simulate a prediction
    dummy_predictions = tf.random.normal((1, VOCAB_SIZE))
    predicted_id = tf.argmax(dummy_predictions, axis=-1)[0].numpy()

    print(f"✓ Prediction logic works")
    print(f"   Sample predicted ID: {predicted_id}")

    if tokenizer and predicted_id in tokenizer.index_word:
        print(f"   Mapped to word: '{tokenizer.index_word[predicted_id]}'")
    elif tokenizer:
        print(f"   ✗ Predicted ID {predicted_id} not in tokenizer!")
        print(
            f"   Tokenizer index_word range: {min(tokenizer.index_word.keys())} to {max(tokenizer.index_word.keys())}")

except Exception as e:
    print(f"✗ Error in prediction logic: {e}")

# 5. Recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

print("""
Common issues and solutions:

1. SINGLE WORD CAPTIONS:
   - Check that your decoder input is being updated correctly
   - Ensure you're not breaking the loop too early
   - Verify temperature and sampling parameters
   - Check for repetition detection being too aggressive

2. VOCABULARY MISMATCH:
   - Ensure tokenizer vocab_size matches model VOCAB_SIZE
   - Check that special tokens exist in tokenizer
   - Verify index_word mapping is complete

3. MODEL NOT LEARNING:
   - Check if checkpoint exists and is loading correctly
   - Verify training loss was decreasing
   - Ensure sufficient training data and epochs

4. RANDOM OUTPUTS:
   - Model might not be trained or checkpoint not loading
   - Check model architecture matches training code
   - Verify weights are being restored

Try running the fixed version with:
   python main_fixed.py
   
Or in Streamlit:
   streamlit run main_fixed.py
""")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
