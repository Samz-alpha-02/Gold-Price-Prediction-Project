import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the scaler and model
scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/model.pkl", "rb"))

# Route for Welcome Page
@app.route('/')
def welcome():
    return render_template('home.html')

# Route for Gold Price Prediction
@app.route('/predict_price', methods=['POST'])
def predict_price():
    result = ""

    if request.method == 'POST':
        SPX = float(request.form.get("SPX"))
        GLD = float(request.form.get('GLD'))
        USO = float(request.form.get('USO'))
        rolling_mean = float(request.form.get('rolling_mean')) if request.form.get('rolling_mean') else None

        # Use the loaded scaler to transform the input features
        new_data = scaler.transform([[SPX, GLD, USO, rolling_mean]])

        # Predict using the loaded model
        predict = model.predict(new_data)

        result = f"The predicted gold price is: {predict[0]:.2f} EUR/USD"

    return render_template('single_prediction.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
