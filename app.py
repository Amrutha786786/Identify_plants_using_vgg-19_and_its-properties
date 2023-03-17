from __future__ import division, print_function
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import numpy as np

# Flask utils
from flask import Flask, request, render_template

app = Flask(__name__)

model== load_model('model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file']
        img_bytes = img.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((224,224))
        x = image.img_to_array(img)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        l=['Citrus Limon (Lemon)', 'Tabernaemontana Divaricata (Crape Jasmine)', 'Muntingia Calabura (Jamaica Cherry-Gasagase)', 'Alpinia Galanga (Rasna)', 
           'Santalum Album (Sandalwood)', 'Jasminum (Jasmine)','Basella Alba (Basale)', 'Artocarpus Heterophyllus (Jackfruit)', 'Murraya Koenigii (Curry)',
           'Ficus Religiosa (Peepal Tree)', 'Punica Granatum (Pomegranate)', 'Mangifera Indica (Mango)', 'Nerium Oleander (Oleander)', 'Carissa Carandas (Karanda)',
           'Psidium Guajava (Guava)', 'Hibiscus Rosa-sinensis', 'Syzygium Cumini (Jamun)', 'Syzygium Jambos (Rose Apple)', 'Mentha (Mint)',
           'Ficus Auriculata (Roxburgh fig)', 'Amaranthus Viridis (Arive-Dantu)', 'Ocimum Tenuiflorum (Tulsi)', 'Pongamia Pinnata (Indian Beech)',
           'Nyctanthes Arbor-tristis (Parijata)', 'Brassica Juncea (Indian Mustard)', 'Trigonella Foenumgraecum (Fenugreek)', 'Piper Betle (Betel)', 
           'Moringa Oleifera (Drumstick)', 'Azadirachta Indica (Neem)', 'Plectranthus Amboinicus (Mexican Mint)']
        pred_index = np.argmax(preds)
        result = l[pred_index].upper()
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
