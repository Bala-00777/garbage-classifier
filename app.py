import pickle
import cv2
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))
    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)

    return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)