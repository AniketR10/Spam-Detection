# SPAM DETECTION WEBAPP
The Spam Detection App is a machine learning-based application built with Streamlit that classifies text messages as either "spam" or "ham" (non-spam). It uses Natural Language Processing (NLP) techniques and a Naive Bayes classifier to analyze message content and make predictions.

#Features
Interactive web interface for real-time spam detection

Text preprocessing with NLTK for improved accuracy

TF-IDF vectorization for feature extraction

Naive Bayes classification algorithm

Performance optimization using Streamlit's caching system

#Installation
Prerequisites
Python 3.8+

pip (Python package installer)

#Setup
Clone the repository or download the source code

Navigate to the project directory

Install the required dependencies:

#bash
pip install pandas numpy scikit-learn nltk streamlit
NLTK Resources
The application requires specific NLTK resources. They will be downloaded automatically when the app is run, but you can pre-install them with:

python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
Usage
Run the Streamlit app:

bash
streamlit run spam.py
Once the application starts, you can access it in your web browser (typically at http://localhost:8501)

Enter a message in the text area and click the "Predict" button to classify it as spam or ham

#How It Works
Data Loading: The app loads a Twitter dataset for training

Preprocessing: Text messages are cleaned, tokenized, and filtered for stopwords

Model Training: The app uses TF-IDF vectorization and Naive Bayes classification

Prediction: User-entered text is processed the same way and classified by the trained model

#Performance Considerations
The app uses Streamlit's caching system to improve performance

Data loading and model training happen only once during the first run

Subsequent interactions with the app are much faster due to the cached resources

#Technical Details
Text Preprocessing: Converts text to lowercase, removes special characters, tokenizes, and removes stopwords

Feature Extraction: Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization

Classification Algorithm: Implements Multinomial Naive Bayes classifier

UI Framework: Built with Streamlit for an interactive web interface
