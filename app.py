from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('/Users/sagarsoni/Downloads/weapon detection/VGG16.h5')  # Make sure VGG16.h5 is in the same directory

# Class labels based on your Alzheimer's dataset
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file to a temporary path
    filepath = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]

    return render_template('index.html', prediction=predicted_class, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)