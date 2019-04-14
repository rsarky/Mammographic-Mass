from flask import Flask, request
from flask_cors import CORS
from flask import jsonify
import pickle
import json

app = Flask(__name__)
CORS(app)
model = pickle.load(open("modelNN.pkl","rb"))

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=['POST'])
def predict():
    rawFeatures = request.get_json()['features']
    features = parse(rawFeatures[0])
    ans = model.predict_proba([features])
    ansList = ans.tolist()
    print(ansList)
    return json.dumps({ 'probabilities': ansList })

    
def validInput(features):
    # TODO: Validate the form inputs here
    return True
def parse(ip):
    features = []
    # Append BIRADS and shape as is
    features.append(ip[0])
    features.append(ip[1])
    arrShape = [ 0 for i in range(4) ]
    arrShape[ip[2]] = 1
    arrMargin = [ 0 for i in range(5) ]
    arrMargin[ip[3]] = 1
    features = features + arrShape + arrMargin
    features.append(ip[4])
    return features
    
