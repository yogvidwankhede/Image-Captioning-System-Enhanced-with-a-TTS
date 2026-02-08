"""
Transformer-based Image Captioning Model
Architecture: InceptionV3 features → Transformer Encoder → Transformer Decoder → Caption
"""

import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoderBlock(layers.Layer):
    """
    Transformer Encoder Block
    Processes image features with self-attention
    """

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        # Feed-forward network
        self.dense_proj = tf.keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        # Layer normalization
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        # Normalize and project
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_proj(inputs)

        # Self-attention on image features
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training
        )

        # Residual connection + normalization
        proj_input = self.layernorm_2(inputs + attention_output)
        return proj_input


class PositionalEmbedding(layers.Layer):
    """
    Positional Embedding for text sequences
    Combines token embeddings with position embeddings
    """

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)

        # Token embeddings
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        # Position embeddings
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)

        # Get embeddings
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)

        # Combine
        return embedded_tokens + embedded_positions


class TransformerDecoderBlock(layers.Layer):
    """
    Transformer Decoder Block
    Generates text conditioned on image features
    """

    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        # Self-attention (looks at previous words)
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        # Cross-attention (looks at image features)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        # Feed-forward network
        self.dense_proj = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        # Layer normalization
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def call(self, inputs, encoder_outputs, training, mask=None):
        # 1. Self-attention (looking at previous words)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # 2. Cross-attention (looking at the image)
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=None,
            training=training
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        # 3. Feed-forward network
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)


class ImageCaptioningModel(tf.keras.Model):
    """
    Complete Image Captioning Model
    Combines Encoder and Decoder for end-to-end training
    """

    def __init__(self, cnn_feature_dim, embed_dim, dense_dim, num_heads, vocab_size, max_len):
        super().__init__()

        # Encoder (processes image features)
        self.encoder = TransformerEncoderBlock(embed_dim, dense_dim, num_heads)

        # Positional embedding for captions
        self.pos_embedding = PositionalEmbedding(
            max_len, vocab_size, embed_dim)

        # Decoder (generates text)
        self.decoder = TransformerDecoderBlock(embed_dim, dense_dim, num_heads)

        # Final prediction layer
        self.score = layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        """
        Forward pass
        
        Args:
            inputs: Tuple of (img_features, captions)
            training: Whether in training mode
            
        Returns:
            predictions: Predicted word probabilities
        """
        img_features, captions = inputs

        # Encode image features
        encoded_img = self.encoder(img_features, training)

        # Embed captions with positional encoding
        embedded_captions = self.pos_embedding(captions)

        # Decode to generate predictions
        decoded_output = self.decoder(embedded_captions, encoded_img, training)

        # Predict next words
        preds = self.score(decoded_output)
        return preds


def create_model(vocab_size, max_length, embed_dim=256, dense_dim=512, num_heads=6):
    """
    Helper function to create model with standard parameters
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum caption length
        embed_dim: Embedding dimension (default: 256)
        dense_dim: Feed-forward dimension (default: 512)
        num_heads: Number of attention heads (default: 6)
        
    Returns:
        model: ImageCaptioningModel instance
    """
    model = ImageCaptioningModel(
        cnn_feature_dim=2048,  # InceptionV3 output
        embed_dim=embed_dim,
        dense_dim=dense_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_len=max_length
    )

    print("✅ Model created successfully")
    print(f"   Architecture: Transformer Encoder-Decoder")
    print(f"   Embedding dim: {embed_dim}")
    print(f"   Dense dim: {dense_dim}")
    print(f"   Attention heads: {num_heads}")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Max caption length: {max_length}")

    return model


if __name__ == "__main__":
    """
    Test model creation
    """
    from config import *

    print("Testing model creation...")

    # Create model
    model = create_model(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        embed_dim=EMBED_DIM,
        dense_dim=DENSE_DIM,
        num_heads=NUM_HEADS
    )

    # Test with dummy data
    print("\nTesting forward pass with dummy data...")
    dummy_img_features = tf.zeros((2, 64, 2048))  # Batch of 2
    dummy_captions = tf.zeros((2, MAX_LENGTH), dtype=tf.int32)

    output = model((dummy_img_features, dummy_captions), training=False)

    print(f"✅ Forward pass successful!")
    print(
        f"   Input shape: img={dummy_img_features.shape}, cap={dummy_captions.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: (2, {MAX_LENGTH}, {VOCAB_SIZE})")

    # Count parameters
    total_params = sum([tf.size(var).numpy()
                       for var in model.trainable_variables])
    print(f"\n📊 Model Statistics:")
    print(f"   Trainable parameters: {total_params:,}")
    print(f"   Model size (approx): {total_params * 4 / (1024**2):.2f} MB")
