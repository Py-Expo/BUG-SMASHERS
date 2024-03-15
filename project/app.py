from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the Ridge regression model
model = joblib.load('ridge_regression_model.joblib')

# Mapping for converting string features to numeric values
gender_mapping = {'male': 0, 'female': 1}
education_mapping = {'some high school': 0, 'high school': 1, 'some college': 2,
                     "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5}

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the form
        gender = gender_mapping[request.form['gender']]
        parent_education = education_mapping[request.form['parent-education']]
        writing_score = float(request.form['writing-score'])
        reading_score = float(request.form['reading-score'])

        # Create an array with all features
        features = [gender, parent_education, writing_score, reading_score]

        # Add dummy values for the remaining 8 features (assuming they are 0)
        features.extend([0] * 8)

        # Perform prediction
        prediction = model.predict([features])[0]

        # Render the template with the prediction
        return render_template('index.html', prediction=prediction)
    else:
        # Render the template without prediction initially
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
