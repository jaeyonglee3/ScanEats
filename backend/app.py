from flask import Flask, request, jsonify, render_template, request
from cnn.classify import get_prediction
from reverse_proxy import proxy_request
import os
from flask_cors import CORS

MODE = os.getenv('FLASK_ENV')
DEV_SERVER_URL = "http://localhost:3000/"

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

if MODE == "development":
    app = Flask(__name__, static_folder=None)

@app.route('/')
def home_page():
    return "Hello"

@app.route('/classify/apple', methods=['POST'])
def classify_apple():
    print("Analyzing apple")
    if (request.files['image']):
        image = request.files['image']
        result = get_prediction(image)
        print('Model classification: ' + result)
        return jsonify({"result": result})

@app.route('/classify/banana', methods=['POST'])
def classify_banana():
    print("Analyzing banana")
    if (request.files['image']):
        image = request.files['image']
        result = get_prediction(image)
        print('Model classification: ' + result)
        return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
