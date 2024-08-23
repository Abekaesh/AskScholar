from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores import TiDBVectorStore
from langchain.chains import RetrievalQA
import requests
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()
# TiDB setup
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
embedding_model = 'models/embedding-001'  # Replace with the actual model name

def text_to_embedding(text):
    embeddings = genai.embed_content(model=embedding_model, content=[text], task_type="retrieval_document")
    return embeddings['embedding'][0] if 'embedding' in embeddings else None

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name='models/embedding-001'):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [self.text_to_embedding(text) for text in texts]

    def embed_query(self, query):
        return self.text_to_embedding(query)

    def text_to_embedding(self, text):
        embeddings = genai.embed_content(
            model=self.model_name,
            content=[text],
            task_type="retrieval_document"
        )
        return embeddings['embedding'][0] if 'embedding' in embeddings else None

embedding_function = GeminiEmbeddings()

# Embedding setup
tidb_connection_string = os.getenv("TIDB_CONNECTION_STRING")
# Initialize vector store
vector_store = TiDBVectorStore(connection_string=tidb_connection_string, embedding_function=embedding_function, table_name="scholarship_embeddings")


# RAG pipeline without LLM (retrieval only)
def retrieve_relevant_documents(query):
    results = vector_store.similarity_search_with_score(query, k=5)
    
    # Print the first Document object to inspect its attributes
    
    # Extract the text from Document objects (replace 'text' with the correct attribute if different)
    rag_context = " ".join([doc[0].page_content for doc in results])
    sources = [doc[0].metadata for doc in results]
    return rag_context, sources


model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history=[
        {"role": "model", "parts": "You are a helpful scholarship advisor. Your job is to provide guidance on scholarships."},
    ]
)


st.set_page_config(
    page_title="AskScholar 🎓",
    page_icon=":student",
    layout="centered"
)

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
    
# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[{"role": "model", "parts": "My name is Scholar. I am a friendly scholarship advisor. Ask about scholarship. If I know i shall answer it."},])


# Display the chatbot's title on the page
st.title("Ask Scholar")

# Display the chat history
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Ask me about scholarships...")
if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)

    # Retrieve RAG context
    rag_context, sources = retrieve_relevant_documents(user_prompt)
    combined_prompt = f"Context: {rag_context}\nUser Query: {user_prompt}\nAnswer:"

    # Send user's message to Gemini-Pro and get the response
    #gemini_response = st.session_state.chat_session.send_message(combined_prompt)
    gemini_response = model.generate_content(combined_prompt)
    
    # Display Gemini-Pro's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response.text)