from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import nltk
import re
from nltk.tokenize import word_tokenize

nltk.download("punkt")

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = "data/model/lstm_w_770.h5"
TOKENIZER_PATH = "data/model/tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # Set this to the padding length used during training


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return " ".join(tokens)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        user_input = request.form["sentence"]
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
        prob = model.predict(padded)[0][0]
        prediction = "Positive" if prob > 0.5 else "Negative"
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
