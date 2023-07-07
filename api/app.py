from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('solarModel.pkl')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    value1 = data['value1']
    value2 = data['value2']

    prediction = model.predict([[value1,value2]])

    prediction = prediction.tolist()

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run()