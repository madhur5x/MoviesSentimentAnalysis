import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import os

# Streamlit app title and description
st.title('Movie Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

try:
    # Load the pre-trained model
    if not os.path.exists('SimpleRnn/simple_rnn_imdb.h5'):
        st.error("Error: Model file not found. Please check if 'SimpleRnn/simple_rnn_imdb.h5' exists.")
    else:
        model = load_model('SimpleRnn/simple_rnn_imdb.h5')
        st.info("Model loaded successfully.")

    # Load the IMDB dataset words index
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    # User input
    user_input = st.text_area('Movie Review')

    if st.button('Classify'):
        if not user_input:
            st.warning("Please enter a movie review.")
        else:
            # Preprocess the input
            words = user_input.lower().split()
            encoded_review = [word_index.get(word, 2) + 3 for word in words]
            padded_review = sequence.pad_sequences([encoded_review], maxlen=500)

            # Make prediction
            prediction = model.predict(padded_review)

            # Display result
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            st.write(f"It's a {sentiment} review")
            st.write(f'Prediction score: {prediction[0][0]:.4f}')

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check if all required libraries are installed and the model file exists.")
