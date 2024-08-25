# Step 1: Import all the libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset's word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
try:
    model = load_model('SimpleRnn/simple_rnn_imdb.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Step 2: Helper functions

# Function to decode integer back to words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user's input text to padded vector
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Designing the Streamlit app
st.title('Movie Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        
        # Make prediction in the app
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.6 else 'Negative'
        
        # Displaying the result in the Streamlit app
        st.write(f"It's a {sentiment} review")
        st.write(f'Prediction score: {prediction[0][0]:.2f}')
    else:
        st.warning('Please enter a movie review to classify.')
else:
    st.write('Tell me how was the movie!')
