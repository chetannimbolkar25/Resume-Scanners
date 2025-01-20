import streamlit as st
import pickle
import re
import nltk
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# 1. Define a function to clean the resume text
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# 2. Train the model (This is a simple example with sample data)
texts = [
    "This is a sample resume text for a Java Developer position.",
    "Experience in Python, Django, and Flask.",
    "Web Designing and UX/UI experience.",
    "Data Science and Machine Learning projects."
]
labels = [15, 20, 24, 6]  # Corresponding labels for each text

# Initialize the TfidfVectorizer and SGDClassifier
tfidf = TfidfVectorizer()
clf = SGDClassifier()

# Create a pipeline that first transforms the data with TfidfVectorizer, then fits the classifier
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', clf),
])

# Fit the pipeline on the sample data
pipeline.fit(texts, labels)

# Save the pipeline to a pickle file
with open('clf.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# 3. Streamlit app to use the trained model
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        
        # Load the pipeline (tfidf + clf) from the pickle file
        pipeline = pickle.load(open('clf.pkl', 'rb'))
        
        # Predict the category
        prediction_id = pipeline.predict([cleaned_resume])[0]
        
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
