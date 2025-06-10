from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf

# Enable memory growth for TensorFlow (if GPU is present)
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load model once at startup
model = load_model('Alzheimer_Disease.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    try:
        # Load and preprocess the image
        img = image.load_img(file, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        predicted_class = classes[np.argmax(predictions[0])]

        return render_template('result.html', prediction=predicted_class)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)