import os, json, numpy as np, pickle
BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "data", "intents.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = []
for intent in data["intents"]:
    for p in intent.get("patterns", []):
        # pick a random response for that pattern as target (simple way to build pairs)
        resp = intent.get("responses", [intent.get("tag","")])[0]
        pairs.append((p, resp))

# Prepare text data: inputs and target sentences with start/end tokens
input_texts = [p for p,_ in pairs]
target_texts = ['\t ' + r + ' \n' for _,r in pairs]  # \t start, \n end

# Tokenizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_tokenizer = Tokenizer(filters='', lower=True, oov_token='<OOV>')
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
max_encoder_seq_length = max(len(s) for s in input_sequences)

target_tokenizer = Tokenizer(filters='', lower=True, char_level=False, oov_token='<OOV>')
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
max_decoder_seq_length = max(len(s) for s in target_sequences)

num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences([s[:-1] for s in target_sequences], maxlen=max_decoder_seq_length, padding='post')
decoder_target_data = pad_sequences([s[1:] for s in target_sequences], maxlen=max_decoder_seq_length, padding='post')

# One-hot encode decoder target for categorical crossentropy
import tensorflow as tf
decoder_target_onehot = tf.keras.utils.to_categorical(decoder_target_data, num_decoder_tokens)

# Build stronger seq2seq model: encoder with stacked BiLSTM, decoder with LSTM + attention-like (simple)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, TimeDistributed

# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
enc_emb = Embedding(input_dim=num_encoder_tokens, output_dim=128, mask_zero=True, name='enc_emb')(encoder_inputs)
enc_l1 = Bidirectional(LSTM(128, return_sequences=True, return_state=True, recurrent_regularizer=tf.keras.regularizers.l2(1e-4)), name='enc_bi_l1')
enc_out1, forward_h1, forward_c1, backward_h1, backward_c1 = enc_l1(enc_emb)
state_h1 = tf.keras.layers.Concatenate()([forward_h1, backward_h1])
state_c1 = tf.keras.layers.Concatenate()([forward_c1, backward_c1])
enc_l2 = Bidirectional(LSTM(128, return_sequences=False, return_state=True, recurrent_regularizer=tf.keras.regularizers.l2(1e-4)), name='enc_bi_l2')
enc_out2, forward_h2, forward_c2, backward_h2, backward_c2 = enc_l2(enc_out1)
state_h2 = tf.keras.layers.Concatenate()([forward_h2, backward_h2])
state_c2 = tf.keras.layers.Concatenate()([forward_c2, backward_c2])

# Use the second layer states as initial state for decoder
encoder_states = [state_h2, state_c2]

# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
dec_emb_layer = Embedding(input_dim=num_decoder_tokens, output_dim=128, mask_zero=True, name='dec_emb')
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True, recurrent_regularizer=tf.keras.regularizers.l2(1e-4), name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'), name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit([encoder_input_data, decoder_input_data], decoder_target_onehot, batch_size=16, epochs=120, validation_split=0.1)

# Save full model (optional)
model.save(os.path.join(BASE, "model_tf", "seq2seq_full.h5"))

# Build inference encoder model
import os
os.makedirs(os.path.join(BASE, "model_tf"), exist_ok=True)
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save(os.path.join(BASE, "model_tf", "encoder_model.h5"))

# Build inference decoder model
from tensorflow.keras.layers import Input as KInput
decoder_state_input_h = KInput(shape=(256,), name='decoder_state_input_h')
decoder_state_input_c = KInput(shape=(256,), name='decoder_state_input_c')
dec_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2d, state_c2d = decoder_lstm(dec_emb2, initial_state=dec_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + dec_states_inputs, [decoder_outputs2, state_h2d, state_c2d])
decoder_model.save(os.path.join(BASE, "model_tf", "decoder_model.h5"))

# Save tokenizers and params
target_index_word = {i:w for w,i in target_tokenizer.word_index.items()}
target_index_word = {i: w for w,i in enumerate([''] + list(target_tokenizer.word_index.keys()))}
data_to_save = {
    "input_tokenizer": input_tokenizer,
    "target_tokenizer": target_tokenizer,
    "max_encoder_seq_length": max_encoder_seq_length,
    "max_decoder_seq_length": max_decoder_seq_length,
    "target_index_word": target_index_word,
    "target_word_index": target_tokenizer.word_index
}
with open(os.path.join(BASE, "model_tf", "tokenizers.pkl"), "wb") as f:
    pickle.dump(data_to_save, f)

print("Training complete. Models saved to model_tf/")