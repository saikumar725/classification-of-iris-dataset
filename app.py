# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('classification-model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    Prediction = model.predict(final_features)


    return render_template('index.html',prediction=Prediction)



if __name__ == "__main__":
    app.run(debug=True)