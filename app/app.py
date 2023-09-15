
from header import * 


def get_documents_from_df(doc_,name_col):
    df  =  pd.read_csv(doc_)
    loader = DataFrameLoader(df,page_content_column=name_col)
    documents = loader.load()
    return documents

def get_vector_store(documents_):
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents_, embedding)
    print(embedding)
    print(vectorstore)
    return  vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
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
    
    st.header("SuperbioGPT :dna:")

    #if uploaded_file: 
    st.info('Hi there! I’m superbioGPT:dna: I can find apps for your use case and provide info on data requirements. For instance you can ask me: “What app should I use for protein folding?” or “What should my csv look like?”')
    user_question = st.text_input('Send a message below 👇') 
    
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

                    st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__ == '__main__' :
    main() 


