import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import pickle
import pyttsx3
import os

# =========================================
# 1. MODEL ARCHITECTURE (Must match training)
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
        self.embedding = PositionalEmbedding(50, 5001, embed_dim)
        self.out = layers.Dense(5001)

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
# 2. CONFIGURATION & RESOURCE LOADING
# =========================================


# Hyperparameters (Must match training exactly)
EMBED_DIM = 256
DENSE_DIM = 512
NUM_HEADS = 6
VOCAB_SIZE = 5001
MAX_LENGTH = 51


@st.cache_resource
def load_resources():
    # A. Load Tokenizer
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.error("Error: 'tokenizer.pkl' not found. Did you run the save cell?")
        st.stop()

    # B. Load Feature Extractor (InceptionV3)
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    feature_extractor = tf.keras.Model(new_input, hidden_layer)

    # C. Load & Restore Captioning Model
    caption_model = ImageCaptioningModel(
        cnn_feature_dim=2048, embed_dim=EMBED_DIM, dense_dim=DENSE_DIM,
        num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE, max_len=MAX_LENGTH
    )

    # We need to call the model once on dummy data to initialize weights
    dummy_img = tf.zeros((1, 64, 2048))
    dummy_cap = tf.zeros((1, MAX_LENGTH))
    _ = caption_model((dummy_img, dummy_cap), training=False)

    # Restore weights
    ckpt = tf.train.Checkpoint(caption_model=caption_model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, "./checkpoints/train", max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Restored from {ckpt_manager.latest_checkpoint}")
    else:
        st.warning(
            "No checkpoints found! Model is initializing with random weights.")

    return tokenizer, feature_extractor, caption_model


# Load everything
tokenizer, feature_extractor, caption_model = load_resources()

# =========================================
# 3. INFERENCE LOGIC (The "Brain")
# =========================================


def load_image_from_path(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def evaluate_caption(image_tensor):
    # 1. Extract Features
    temp_input = tf.expand_dims(image_tensor, 0)
    img_tensor_val = feature_extractor(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, 2048))

    # 2. Generate Caption
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    last_word = ""

    for i in range(MAX_LENGTH):
        predictions = caption_model(
            (img_tensor_val, dec_input), training=False)
        predictions = predictions[:, -1, :]

        predicted_id = tf.argmax(predictions, axis=-1)[0].numpy()

        if tokenizer.index_word.get(predicted_id) is None:
            break

        predicted_word = tokenizer.index_word[predicted_id]

        # Stop loops
        if predicted_word == last_word:
            break

        if predicted_word == '<end>':
            return result

        result.append(predicted_word)
        last_word = predicted_word
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

# =========================================
# 4. STREAMLIT UI
# =========================================


st.set_page_config(page_title="Image Captioner", layout="centered")

st.title("👁️ AI Image Captioning System")
st.markdown("Upload an image, and the AI will describe what it sees.")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily so TensorFlow can read it
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('✨ Generate Caption'):
        with st.spinner('Thinking...'):
            try:
                # Preprocess and Predict
                processed_img = load_image_from_path("temp_image.jpg")
                caption_words = evaluate_caption(processed_img)

                # Format text
                full_caption = ' '.join(caption_words)
                st.success(f"**Predicted Caption:** {full_caption}")

                # Text to Speech
                if full_caption:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.save_to_file(full_caption, 'caption_audio.mp3')
                    engine.runAndWait()

                    # Streamlit Audio Player
                    audio_file = open('caption_audio.mp3', 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')

            except Exception as e:
                st.error(f"An error occurred: {e}")
