# app.py
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the pipeline (includes scaler + model)
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from form
    f1 = float(request.form['Day from 1 Jan'])
    f2 = float(request.form['Pressure'])
    f3 = float(request.form['Maximum Temperature'])
    f4 = float(request.form['Temperature'])
    f5 = float(request.form['Minimum Temperature'])
    f6 = float(request.form['Dew Point'])
    f7 = float(request.form['Humidity'])
    f8 = float(request.form['Cloud'])
    f9 = float(request.form['Sunshine'])
    f10 = float(request.form['Wind Direction'])
    f11 = float(request.form['Wind Speed'])

    features = [[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]]

    # Directly predict using pipeline (it handles scaling inside)
    prediction = pipeline.predict(features)[0]

    return jsonify({'prediction': "It will rain" if prediction == 1 else "It won't rain"})

if __name__ == '__main__':
    app.run(debug=True)
