from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained KNN model
model = joblib.load('flight_nb_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get user input from the HTML form
            input_data = [
                float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4']),
                float(request.form['feature5']),
                float(request.form['feature6']),
                float(request.form['feature7']),
                float(request.form['feature8']),
                float(request.form['feature9']),
                float(request.form['feature10'])
            ]

            # Make a prediction using the model
            prediction = model.predict([input_data])[0]
        except Exception as e:
            # Handle errors gracefully, e.g., invalid input
            print(f"Error: {str(e)}")
            prediction = "Invalid input"

    return render_template('index.html', prediction=prediction)

@app.route('/dash', methods=['GET', 'POST'])
def dashboard():
    # Add code for your dashboard here
    return render_template('dash.html')

if __name__ == '__main__':
    app.run(debug=True)
