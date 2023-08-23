import os
from apikey import apikey
import streamlit as st 
from langchain.llms import OpenAI


os.environ['OPENAI_API_KEY'] = apikey


# App framework
st.title('GPT connector')
prompt = st.text_input('Plug in your prompt here') 

#LLM
llm = OpenAI(temperature=0.9)

if prompt : 
    response =  llm(prompt)
    st.write(response)