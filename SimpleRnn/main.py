# step 1 is to Import all the libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the imdb dataset words index

word_index=imdb.get_word_index()
reverse_word_index= { value: key for key,value in word_index.items()}

#load the pre-trained model with reLU activation
model = load_model('SimpleRnn/simple_rnn_imdb.h5')

## STEP 2 helper function 

## function to decode integer back to the words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# function to preprocess users input text to padded vector 
def preprocess_text(text):
    words = text.lower().split()
    ## gets key for every text
    encoded_review = [word_index.get(word,2)+3 for word in words]
    ## converting it to vector form
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    ## sequence is use for providing utility for preprocessing for sequiential data
    return padded_review

import streamlit as st
### designing STREAMLIT app
st.title('Movie Sentiments Anaslysis')
st.write('Enter A Movie review to classify it as a positive and negitive.')

## user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    
## make prediction in app

    prediction=model.predict(preprocessed_input)
    sentiment ='Positive' if prediction[0][0] > 0.6  else 'Negative'
    
## displaying the result in streamlit app
    st.write(f'It's a {sentiment} review')
    st.write(f'prediction score:{prediction[0][0]}')
else:
    st.write('Tell me how was the movie!!! .')

             
    



    
    


