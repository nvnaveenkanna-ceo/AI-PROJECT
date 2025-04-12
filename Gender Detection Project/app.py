from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = "my_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure correct label mapping
LABELS = {0: "Female", 1: "Male"}  # Fixed mapping

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save and process the image
    file_path = "uploaded.jpg"
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply MobileNet preprocessing
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    
    # Fix label interpretation
    predicted_class = int(prediction[0][0] > 0.5)  # 1 = Male, 0 = Female
    gender = LABELS[predicted_class]

    return jsonify({"gender": gender})

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()
    print(f"Received message: '{user_message}'")  # Debugging

    # Enhanced chatbot responses
    responses = {
        "hello": "Hi there! How can I assist you today?",
        "who are you": "I am a chatbot that predicts gender from images and answers basic questions!",
        "how does gender prediction work": "I use a deep learning model to analyze facial features and predict gender.",
        "which model": "I use MobileNet, a lightweight deep learning model, for gender classification.",
        "how accurate": "My accuracy depends on the image quality, lighting, and dataset used for training.",
        "machine learning": "Machine learning is a branch of AI where computers learn from data to make predictions.",
        "deep learning": "Deep learning is a subset of machine learning that uses neural networks to learn patterns.",
        "do you store": "No, your images are only used for prediction and are not stored.",
        "can i train": "Yes! You can collect data, preprocess it, and train a model using TensorFlow or PyTorch.",
        "what programming language": "This chatbot is built using Python with Flask for the backend.",
        "how can i improve": "You can improve it by using a larger dataset, data augmentation, and hyperparameter tuning.",
        "why is my prediction incorrect": "It may be due to poor image quality, unusual lighting, or bias in the training dataset.",
        "deploy a machine learning model": "You can deploy it using Flask, FastAPI, or cloud platforms like AWS, GCP, and Heroku.",
        "bye": "Goodbye! Have a great day!"
    }

    # Match based on partial keywords
    bot_response = "I'm still learning! Try asking something else."
    for key, response in responses.items():
        if key in user_message:  # Partial match
            bot_response = response
            break  # Stop searching once a match is found

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
