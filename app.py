from utils import viterbi_backward, my_preprocess
import streamlit as st
from pandas import read_csv


#app sidebar
st.sidebar.subheader('About the App')
st.sidebar.write("POS tagging app using a simple tagger.")
st.sidebar.write("These are just small models that predict the part of speech for each word in the sentence with respect to context.")
st.sidebar.write("The models are not perfect, so they may miss out some words...")
st.sidebar.write("Don't mind the crude display of the tags :)")
st.sidebar.write("Below is a table showing the tags and their meaning")
total_tags = read_csv('tags.csv')
st.sidebar.write(total_tags)


#start the user interface
st.title("POS (Part of Speech) Tagging App.")
st.write("Check the left sidebar for more information.")
st.write("Type in your sentence below and don't forget to press the enter button before clicking/pressing the button below.")

models = {"Simple model":"simple",
         "Hidden Markov model with Virtebi Algorithm":"hmm"}


my_model = st.selectbox("Select a model to use:", list(models.keys()), key="models")
model_to_use = models[my_model]
my_text = st.text_input("Enter your sentence...", "A sample sentence.", max_chars=100, key='to_classify')



if st.button('Get POS tags', 'run_model'):
    
    orig, prep = my_preprocess(my_text)
    
    
    pred = viterbi_backward(prep)
    

    to_display = {}

    for i in range(len(orig)):
        to_display[orig[i]] = pred[i]

    st.write("The POS tags for your sentence are:", to_display)
