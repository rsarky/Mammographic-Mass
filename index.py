from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=['POST'])
def predict():
    if validInput(request.form):
        # TODO: Call the machine learning model and return the appropriate data
        return "The input is valid"
    return "Invalid form input."

    
def validInput(form):
    # TODO: Validate the form inputs here
    return True
