from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("modelNN.pkl","rb"))

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=['POST'])
def predict():
    features = request.get_json()['features']
    if validInput(features):
        # TODO: Call the machine learning model and return the appropriate data
        ans = model.predict(features)
        print(ans)
        return np.array2string(ans)
    return "Invalid form input."

    
def validInput(features):
    # TODO: Validate the form inputs here
    return True
