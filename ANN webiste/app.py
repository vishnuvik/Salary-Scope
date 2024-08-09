from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the pre-trained models and encoders
label_encoders = joblib.load('C:/Users/vishn/OneDrive/Desktop/Job market prediction/label_encoders.pkl')

minmax_scaler = joblib.load('C:/Users/vishn/OneDrive/Desktop/Job market prediction/MinMaxScaler.pkl')


salary_scaler = joblib.load('C:/Users/vishn/OneDrive/Desktop/Job market prediction/Salary_scale.pkl')

# Load the target encoder for the company
company_target_encoder = joblib.load('C:/Users/vishn/OneDrive/Desktop/Job market prediction/company_target_encoder.pkl')

model = load_model('C:/Users/vishn/OneDrive/Desktop/Job market prediction/best_model.h5')

# Define high and low skills sets (all lowercase)
high_skills_set = {'tableau', 'powerpoint', 'vba', 'office', 'phd', 'machine learning', 'mba', 'docker', 'sap', 'spark', 'master', 'dynamics 365', 'sql', 'agile', 'python', 'jira', 'snowflake', 'erp', 'excel', 'hadoop', 'javascript', 'azure', 'tensor flow', 'word', 'databricks', 'deep learning', 'access', 'artificial intelligence', 'power bi', 'oracle', 'teradata', 'aws', 'c++', 'cpa', 'r', 'bachelor', 'java', 'english', 'google cloud'}
low_skills_set = {'github', 'looker', 'css', 'mongodb', 'pandas', 'dax', 'hyperion', 'spanish', 'seaborn', 'scikit', 'google sheets', 'php', 'matlab', 'c#', 'power pivot', 'french', 'd3', 'polars', 'power automate', 'matplotlib', 'numpy', 'sage', 'plotly', 'russian', 'qlik', 'angular', 'rust', 'power query', 'neural network', 'fabric', 'essbase', 'dash', 'ssis', 'sap analytics cloud', '.net', 'japanese', 'navision', 'salesforce', 'german', 'quickbooks', 'abap', 'react', 'html', 'snaplogic', 'airflow', 'cma', 'cfa', 'adobe analytics', 'kaggle', 'streamlit', 'chat gpt', 'jupyter', 'ssrs', 'chinese', 'power apps', 'cognos', 'domo', 'ssas'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data
            job_title = request.form['job_title']
            position = request.form['position']
            company_name = request.form['company_name']
            state = request.form['state']
            skills = request.form.getlist('skills[]')
            tenure = int(request.form['tenure'])
            experience = int(request.form['experience'])

            # Process skills
            skills = [skill.lower() for skill in skills]
            high_skills = len([skill for skill in skills if skill in high_skills_set])
            low_skills = len([skill for skill in skills if skill in low_skills_set])

            # Encode categorical features using label encoders
            job_title_encoded = label_encoders['Job_Title'].transform([job_title])[0]
            position_encoded = label_encoders['position'].transform([position])[0]
            state_encoded = label_encoders['State'].transform([state])[0]
            
            # Mean encode company name using the target encoder
            company_encoded = company_target_encoder.transform(pd.DataFrame({'Company': [company_name]})).values[0][0]

            # Combine all features into a single array
            features = np.array([
                job_title_encoded, 
                position_encoded, 
                state_encoded,
                company_encoded,
                high_skills, 
                low_skills, 
                tenure, 
                experience
            ])

            # Ensure all features are numerical
            features = features.astype(float)

            # Scale features
            features_scaled = minmax_scaler.transform([features])

            # Predict salary
            salary_scaled = model.predict(features_scaled)
            predicted_salary = salary_scaler.inverse_transform(salary_scaled)

            return render_template('pred.html', prediction=int(predicted_salary[0][0]))

        except Exception as e:
            return render_template('pred.html', prediction=f'Error: {str(e)}')

    return render_template('pred.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
