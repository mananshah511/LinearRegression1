from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


app = Flask(__name__)


@app.route('/', methods= ['GET'])
def homepage():
    return render_template("index.html")


@app.route('/predict', methods = ['POST'])
def predict():
    model1 = pickle.load(open('model1.pkl', 'rb'))
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    #print(data)
    output = model1.predict(final_features)[0]
    #print(output)
    return render_template('index.html', prediction_text="predicted price is  {}".format(output))

if __name__ == "__main__":
    app.run()