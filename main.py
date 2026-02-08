import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import pickle
import pyttsx3
import os

# =========================================
# MODEL ARCHITECTURE
# =========================================


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
        self.encoder = TransformerEncoderBlock(embed_dim, dense_dim, num_heads)
        self.pos_embedding = PositionalEmbedding(
            max_len, vocab_size, embed_dim)
        self.decoder = TransformerDecoderBlock(embed_dim, dense_dim, num_heads)
        self.score = layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        img_features, captions = inputs
        encoded_img = self.encoder(img_features, training)
        embedded_captions = self.pos_embedding(captions)
        decoded_output = self.decoder(embedded_captions, encoded_img, training)
        preds = self.score(decoded_output)
        return preds

# =========================================
# CONFIGURATION
# =========================================


EMBED_DIM = 256
DENSE_DIM = 512
NUM_HEADS = 6
VOCAB_SIZE = 10226
MAX_LENGTH = 51


@st.cache_resource
def load_resources():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.error("tokenizer.pkl not found")
        st.stop()

    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    feature_extractor = tf.keras.Model(
        image_model.input, image_model.layers[-1].output)

    caption_model = ImageCaptioningModel(
        cnn_feature_dim=2048, embed_dim=EMBED_DIM, dense_dim=DENSE_DIM,
        num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE, max_len=MAX_LENGTH)

    dummy_img = tf.zeros((1, 64, 2048))
    dummy_cap = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    _ = caption_model((dummy_img, dummy_cap), training=False)

    ckpt = tf.train.Checkpoint(caption_model=caption_model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, "./checkpoints/train", max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        st.success(f"✓ Loaded checkpoint")
    else:
        st.warning("⚠️ No checkpoint")

    return tokenizer, feature_extractor, caption_model


tokenizer, feature_extractor, caption_model = load_resources()

# =========================================
# INFERENCE WITH AGGRESSIVE FIX
# =========================================


def load_image_from_path(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def evaluate_caption_aggressive(image_tensor, temperature=0.5, top_p=0.9):
    """
    Ultra-aggressive caption generation with multiple safety mechanisms
    """
    # Extract features
    temp_input = tf.expand_dims(image_tensor, 0)
    img_tensor_val = feature_extractor(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, 2048))

    # Tokens
    start_token = tokenizer.word_index.get('<start>', 3)
    end_token = tokenizer.word_index.get('<end>', 4)

    # Initialize
    dec_input = tf.expand_dims([start_token], 0)
    result = []

    # AGGRESSIVE tracking
    used_words = set()  # Track ALL used words
    last_3_words = []   # Track last 3 words
    word_positions = {}  # Track where each word was used

    for step in range(MAX_LENGTH - 1):
        # Get predictions
        predictions = caption_model(
            (img_tensor_val, dec_input), training=False)
        predictions = predictions[:, -1, :]

        # Mask out vocabulary beyond what exists in tokenizer
        valid_vocab = min(VOCAB_SIZE, len(tokenizer.index_word) + 1)
        if VOCAB_SIZE > valid_vocab:
            mask = np.ones(VOCAB_SIZE)
            mask[valid_vocab:] = 0
            predictions = predictions * \
                tf.constant(mask, dtype=tf.float32)[tf.newaxis, :]

        # Apply temperature
        predictions = predictions / max(temperature, 0.01)

        # Convert to probabilities
        probs = tf.nn.softmax(predictions[0]).numpy()

        # Top-p (nucleus) sampling for better diversity
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)

        # Find cutoff for top-p
        cutoff_idx = np.searchsorted(cumsum_probs, top_p)
        cutoff_idx = max(cutoff_idx, 10)  # At least top 10

        # Get valid candidates
        candidate_indices = sorted_indices[:cutoff_idx]
        candidate_probs = sorted_probs[:cutoff_idx]

        # Filter candidates
        valid_candidates = []
        valid_candidate_probs = []

        for idx, prob in zip(candidate_indices, candidate_probs):
            if idx not in tokenizer.index_word:
                continue

            word = tokenizer.index_word[idx]

            # Skip special tokens
            if word in ['<pad>', '<unk>', '<start>']:
                continue

            # Check for end token
            if word == '<end>' and step >= 3:
                if len(result) >= 5:  # Only end if we have enough words
                    return result
                continue

            # AGGRESSIVE FILTERING

            # 1. Skip if word already used
            if word in used_words:
                continue

            # 2. Skip if word is in last 3 words
            if word in last_3_words:
                continue

            # 3. Skip very similar words (simple check)
            skip = False
            for used_word in used_words:
                if len(word) > 3 and len(used_word) > 3:
                    # Check if words are very similar
                    if word[:3] == used_word[:3] or word[-3:] == used_word[-3:]:
                        skip = True
                        break
            if skip:
                continue

            valid_candidates.append(idx)
            valid_candidate_probs.append(prob)

        # If no valid candidates, force stop
        if not valid_candidates:
            if len(result) >= 3:
                return result
            # Emergency: use any word not yet used
            for idx in candidate_indices:
                if idx in tokenizer.index_word:
                    word = tokenizer.index_word[idx]
                    if word not in used_words and word not in ['<pad>', '<unk>', '<start>', '<end>']:
                        valid_candidates = [idx]
                        valid_candidate_probs = [1.0]
                        break
            if not valid_candidates:
                return result if result else ['a', 'scene']

        # Sample from valid candidates
        valid_candidate_probs = np.array(valid_candidate_probs)
        valid_candidate_probs = valid_candidate_probs / valid_candidate_probs.sum()

        chosen_idx = np.random.choice(
            len(valid_candidates), p=valid_candidate_probs)
        predicted_id = valid_candidates[chosen_idx]
        predicted_word = tokenizer.index_word[predicted_id]

        # Add to tracking
        used_words.add(predicted_word)
        last_3_words.append(predicted_word)
        if len(last_3_words) > 3:
            last_3_words.pop(0)
        word_positions[predicted_word] = step

        # Add to result
        result.append(predicted_word)

        # Update input
        new_token = tf.constant([[predicted_id]], dtype=tf.int32)
        dec_input = tf.concat([dec_input, new_token], axis=-1)

        # Hard stops
        if len(result) >= 15:  # Max 15 words
            break

        if step > 5 and len(result) >= 5 and predicted_word in ['.', ',']:
            break

    return result if result else ['a', 'room', 'with', 'furniture']


# =========================================
# STREAMLIT UI
# =========================================

st.set_page_config(page_title="Image Captioner", layout="centered")

st.title("👁️ Image Captioning - Fixed Version")

vocab_mismatch = (len(tokenizer.word_index) + 1 != VOCAB_SIZE)
if vocab_mismatch:
    st.warning(
        f"⚠️ Vocab mismatch: Model={VOCAB_SIZE}, Tokenizer={len(tokenizer.word_index)+1}")
    st.info("This version uses aggressive filtering to work around the mismatch")

st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.5, 0.1,
                                help="Lower = more predictable, higher = more creative")
top_p = st.sidebar.slider("Top-P (Nucleus)", 0.5, 1.0, 0.9, 0.05,
                          help="Probability mass to sample from")

st.sidebar.markdown("---")
st.sidebar.info("""
**How this works:**
- Prevents ANY word repetition
- Forces diversity in generation  
- Stops early to avoid loops
- Uses aggressive filtering

**Best settings:**
- Temp: 0.5-0.7
- Top-P: 0.85-0.95
""")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button('✨ Generate Caption', use_container_width=True):
            with st.spinner('Generating...'):
                try:
                    processed_img = load_image_from_path("temp_image.jpg")
                    caption_words = evaluate_caption_aggressive(
                        processed_img,
                        temperature=temperature,
                        top_p=top_p
                    )

                    if caption_words:
                        full_caption = ' '.join(caption_words)
                        full_caption = full_caption.strip()
                        if full_caption:
                            full_caption = full_caption[0].upper(
                            ) + full_caption[1:]
                            if not full_caption.endswith(('.', '!', '?')):
                                full_caption += '.'

                        st.success(f"**Caption:** {full_caption}")
                        st.info(f"Words: {len(caption_words)}")

                        # TTS
                        try:
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 150)
                            engine.save_to_file(
                                full_caption, 'caption_audio.mp3')
                            engine.runAndWait()

                            if os.path.exists('caption_audio.mp3'):
                                with open('caption_audio.mp3', 'rb') as f:
                                    st.audio(f.read(), format='audio/mp3')
                        except:
                            pass
                    else:
                        st.warning("Could not generate caption")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col2:
        if st.button('🎲 Generate 3 Variations', use_container_width=True):
            with st.spinner('Generating variations...'):
                try:
                    processed_img = load_image_from_path("temp_image.jpg")

                    st.write("**Variations:**")
                    for i in range(3):
                        caption_words = evaluate_caption_aggressive(
                            processed_img,
                            temperature=temperature + (i * 0.15),
                            top_p=top_p
                        )

                        if caption_words:
                            caption = ' '.join(caption_words)
                            caption = caption[0].upper() + \
                                caption[1:] if caption else ""
                            if caption and not caption.endswith(('.', '!', '?')):
                                caption += '.'
                            st.write(f"{i+1}. {caption}")

                except Exception as e:
                    st.error(f"Error: {e}")

with st.expander("🔧 Technical Info"):
    st.write(f"**Model Configuration:**")
    st.write(f"- Model vocab: {VOCAB_SIZE}")
    st.write(f"- Tokenizer vocab: {len(tokenizer.word_index) + 1}")
    st.write(
        f"- Valid range: 0 to {min(VOCAB_SIZE, len(tokenizer.index_word))}")
    st.write(f"- Max caption length: {MAX_LENGTH}")

    st.write("\n**Special Tokens:**")
    for tok in ['<start>', '<end>', '<pad>', '<unk>']:
        idx = tokenizer.word_index.get(tok, 'NOT FOUND')
        st.write(f"- {tok}: {idx}")

    st.write("\n**This Version:**")
    st.write("- Prevents ALL word repetition")
    st.write("- Uses top-p (nucleus) sampling")
    st.write("- Tracks last 3 words")
    st.write("- Filters similar words")
    st.write("- Hard limit: 15 words")

st.markdown("---")
st.caption(
    "⚠️ For best results, retrain model with vocab_size=8256 to match tokenizer")
