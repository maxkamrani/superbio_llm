import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os 
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template


os.environ['OPENAI_API_KEY'] ='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'

def get_documents_from_df(doc_,name_col):
    df  =  pd.read_csv(doc_)
    loader = DataFrameLoader(df,page_content_column=name_col)
    documents = loader.load()
    return documents

def get_vector_store(documents_):
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents_, embedding)
    return  vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    st.set_page_config(page_title="Superbio AI-Helper",
                       page_icon=":dna:")
    
    st.header("Superbio GPT :dna:")


    #session states init
    #if uploaded_file: 
    user_question = st.text_input("I am chatbot to help blabla ") 
    
    if user_question:
        handle_userinput(user_question)

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    with st.sidebar:
        st.subheader("Your documents")
        uploaded_file = st.file_uploader(
            "Upload your CSV here and click on'Process'", type="csv")
        
        if uploaded_file:
            #quick fix 
            col_interrest = "name"
        if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    documents = get_documents_from_df(uploaded_file,col_interrest)

                    # create vector store
                    vectorstore = get_vector_store(documents)

                    st.session_state.conversation  = get_conversation_chain(vectorstore)



if __name__ == '__main__' :
    main() 



