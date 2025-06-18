from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.vgg19 import preprocess_input
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
model = load_model("mn.h5")

dog_breeds = [ 'Affenpinscher', 'Beagle', 'Appenzeller', 'Basset', 'Bluetick', 'Boxer', 'Cairn', 'Doberman', 'German Shepherd', 'Golden Retriever', 'Kelpie', 'Komondor', 'Leonberg', 'Mexican_hairless', 'Pug', 'Redbone', 'Shih-tzu', 'Toy Poodle', 'Vizsla', 'Whippet']

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index.html')
def index2():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')
@app.route('/contactus.html')
def contactus():
    return render_template('contactus.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join("static", filename)
    file.save(filepath)

    # image = load_img(filepath, target_size=(224, 224))
    image = cv2.imread(str(filepath))
    target_size = (224, 224)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    # print(predictions)
    predicted_class = dog_breeds[np.argmax(predictions)]

    os.remove(filepath)
    # print("Working.................................................")
    return jsonify({'breed': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
