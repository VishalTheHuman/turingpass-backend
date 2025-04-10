from flask import Flask, request, jsonify
import numpy as np
import ast
from scipy.spatial.distance import euclidean
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import joblib
from tensorflow.keras import layers
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Constants
CHARACTERS = ['3', 'S', 'E', 'B', 'G', 'P', 'V', 'j', 'T', 'C', '9', 'F', 'm', 
              'a', '2', 'A', 'U', 'r', 'h', 'v', 'Z', 'z', 'w', 's', '1', 'R', 
              '4', 'Y', 'l', '6', 'k', 'O', 'I', 'u', 'N', 't', 'K', 'Q', 'M', 
              'W', 'X', 'D', 'd', 'b', '8', 'p', 'g', '5', 'y', 'f', 'L', 'q', 
              'J', 'n', 'i', 'x', 'c', 'H', 'e', '7']

IMG_WIDTH, IMG_HEIGHT = 150, 40

# StringLookup layers
char_to_num = layers.StringLookup(vocabulary=list(CHARACTERS),
                                  num_oov_indices=0,
                                  mask_token=None)

num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), 
                                  mask_token=None, 
                                  num_oov_indices=0,
                                  invert=True)


# Load model
model = load_model("lstm_autoencoder_model.h5", compile=False)
model.compile(optimizer=Adam(), loss=MeanSquaredError())
scaler = joblib.load("scaler.save")
captcha_model = load_model('captcha_model.h5', custom_objects={'StringLookup': char_to_num})

def compute_mouse_jitter(movement_str: str) -> float:
    try:
        points = ast.literal_eval(movement_str)
        distances = [euclidean(p1.values(), p2.values()) for p1, p2 in zip(points[:-1], points[1:])]
        return np.std(distances) if distances else 0.0
    except Exception:
        return 0.0
    
def preprocess_image_from_bytes(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img

def decode_predictions(preds):
    output_text = []
    for i in range(preds.shape[1]):
        char_index = tf.argmax(preds[0][i])
        char = num_to_char(char_index)
        output_text.append(char)
    return tf.strings.reduce_join(output_text).numpy().decode("utf-8")

@app.route('/is-bot', methods=['POST'])
def predict():
    data = request.get_json()
    jitter = compute_mouse_jitter(data['mouse_movements'])

    features = [
        data['time_taken'],
        data['is_correct'],
        data['attempts'],
        data['key_presses'],
        data['backspace_presses'],
        jitter
    ]

    X_scaled = scaler.transform([features])
    X_reshaped = X_scaled.reshape((1, 1, len(features)))
    reconstruction = model.predict(X_reshaped)
    mse = np.mean(np.square(X_reshaped - reconstruction))
    threshold = np.percentile([mse], 95)
    prediction = 1 if mse > threshold else 0

    return jsonify({"prediction": "bot" if prediction == 1 else "human"})

@app.route('/predict', methods=['POST'])
def predict_captcha():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        img_tensor = preprocess_image_from_bytes(image_bytes)
        prediction = captcha_model.predict(img_tensor)
        decoded = decode_predictions(prediction)
        return jsonify({'prediction': decoded})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return 'CAPTCHA Solver Detection'
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
