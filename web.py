from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Load the trained pipeline from the pickle file
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = {
        'Crop': request.form['Crop'],
        'Crop_Year': float(request.form['Crop_Year']),
        'Season': request.form['Season'],
        'State': request.form['State'],
        'Area': float(request.form['Area']),
        'Production': float(request.form['Production']),
        'Annual_Rainfall': float(request.form['Annual_Rainfall']),
        'Fertilizer': float(request.form['Fertilizer']),
        'Pesticide': float(request.form['Pesticide'])
    }

    # Convert the data to a DataFrame
    df = pd.DataFrame([data])

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(df)

    # Return the result to the template
    result = f'Predicted Yield: {predictions[0]:.2f} Tons'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
