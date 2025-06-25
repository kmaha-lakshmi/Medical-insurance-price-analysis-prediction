from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("insurance_model.pkl", "rb"))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        features = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"Error occurred: {e}"

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT env variable
    app.run(host='0.0.0.0', port=port, debug=True)


