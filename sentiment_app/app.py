from flask import Flask, render_template, request, session
from flask_session import Session  # type: ignore
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import nltk
import re
from nltk.tokenize import word_tokenize

# Download necessary NLTK resource
nltk.download("punkt")

# Flask app configuration
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Constants
MODEL_PATH = "data/model/lstm_w_770.h5"
TOKENIZER_PATH = "data/model/tokenizer.pkl"
MAX_LEN = 100  # Update this if your model was trained with a different sequence length

# Load model and tokenizer
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# Preprocessing function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return " ".join(tokens)


# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

    prediction = ""

    if request.method == "POST":
        if request.form.get("action") == "clear":
            session["history"] = []
            session.modified = True
            return render_template(
                "index.html", prediction="", history=session["history"]
            )

        # If analysing sentiment
        user_input = request.form["sentence"]
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
        prob = model.predict(padded)[0][0]
        prediction = "Positive" if prob > 0.5 else "Negative"

        # Append result to history
        session["history"].append({"text": user_input, "prediction": prediction})
        session.modified = True

    return render_template(
        "index.html", prediction=prediction, history=session["history"]
    )


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
