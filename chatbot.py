import json, random, numpy as np, os, tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope

class Chatbot:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.intents_path = os.path.join(base_dir, "data", "intents.json")

        with open(self.intents_path, "r", encoding="utf-8") as f:
            self.intents = json.load(f)

        self.use_seq2seq = False
        enc_path = os.path.join(base_dir, "model_tf", "encoder.h5")
        dec_path = os.path.join(base_dir, "model_tf", "decoder.h5")

        if os.path.exists(enc_path) and os.path.exists(dec_path):
            custom_objects = { "NotEqual": tf.keras.layers.Lambda(lambda x: x) }
            with custom_object_scope(custom_objects):
                self.encoder_model = load_model(enc_path, compile=False, safe_mode=False)
                self.decoder_model = load_model(dec_path, compile=False, safe_mode=False)
            self.use_seq2seq = True
            print("Seq2Seq model loaded successfully.")
        else:
            print("No Seq2Seq model found. Using retrieval mode only.")


    def get_response(self, text, mode="retrieval"):
        if mode == "lstm" and self.use_seq2seq:
            return self._predict_seq2seq(text)
        else:
            return self._predict_retrieval(text)

    def _predict_retrieval(self, text):
        responses = []
        for intent in self.intents["intents"]:
            responses += intent["responses"]
        return random.choice(responses)

    def _predict_seq2seq(self, text):
        # Dummy placeholder (the actual generation logic is in train_tf.py)
        return "This is a generated response from the Seq2Seq LSTM model."
