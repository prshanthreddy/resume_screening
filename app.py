import streamlit as st
import pdfplumber
import os
import re
from joblib import load

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    resumeText = re.sub('https\+S\s*','',text) 
    resumeText = re.sub('RT|cc','',resumeText)
    resumeText = re.sub('#\S+','',resumeText)
    resumeText = re.sub('@\+S','',resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

def load_model():
    model = load('Resume_Screening.joblib') 
    return model

vectorizer = load('Word_Vec_Resume.joblib')
actual_labels = [
    "Advocate", "Arts", "Automation Testing", "Blockchain", "Business Analyst", 
    "Civil Engineer", "Data Science", "Database", "DevOps Engineer", "DotNet Developer", 
    "ETL Developer", "Electrical Engineering", "HR", "Hadoop", "Health and fitness", 
    "Java Developer", "Mechanical Engineer", "Network Security Engineer", "Operations Manager", 
    "PMO", "Python Developer", "SAP Developer", "Sales", "Testing", "Web Designing"
]

def predict(file_path):
    text = extract_text_from_pdf(file_path)
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    model = load_model()
    result = model.predict(text_vectorized)
    actual_result = [actual_labels[label] for label in result]
    result=actual_result[0]
    return result

st.title("Resume Screening App")
st.write("This app predicts the job category of a resume")

file_path = st.file_uploader("Upload a resume", type="pdf")
if file_path is not None:
    st.write("File uploaded successfully")
    result = predict(file_path)
    st.write(f"The job category of the resume is:")
    st.success(f"{result}")
else:
    st.write("Please upload a resume")