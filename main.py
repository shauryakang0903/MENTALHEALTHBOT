import streamlit as st
import pandas as pd
import os
import qrcode
from io import BytesIO
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.caches import InMemoryCache
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_TOKEN"] = hf_token
os.environ["GROQ_API_KEY"] = groq_api_key

ChatGroq.BaseCache = InMemoryCache
ChatGroq.model_rebuild()

# Load dataset
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "DATA", "Dataset.csv")
dataset = pd.read_csv(dataset_path)

# Streamlit UI Enhancements
st.set_page_config(page_title="Happiness Chatbot", page_icon="😊", layout="centered")
st.markdown("""
    <style>
        body {background-color: #f8f9fa; color: #333;}
        .success-box {background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;}
        .error-box {background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;}
        .chatbot-title {font-size: 36px; font-weight: bold; text-align: center; color: #333;}
        .chatbot-description {font-size: 18px; text-align: center; color: #666;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='chatbot-title'>😊 Happiness Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='chatbot-description'>Welcome to the Happiness Chatbot! 🌟 Ask any question, and our chatbot will provide compassionate and insightful responses to improve mental well-being.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = dataset['Response'].fillna("").astype(str).tolist()
    metadata = [{"label": label} for label in dataset['Response'].fillna("").astype(str).tolist()]
    vectorstore = FAISS.from_texts(documents, embedder, metadatas=metadata)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    retriever = vectorstore.as_retriever()
    return llm, retriever

with st.spinner('⏳ Building the FAISS vectorstore. Please wait...'):
    llm, retriever = load_model_and_vectorstore()
st.markdown("<div class='success-box'>✅ Vectorstore built successfully!</div>", unsafe_allow_html=True)

# Chat System Prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the user's query clearly using simple language."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_chain = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Provide supportive, warm, and age-appropriate responses."),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_chain, question_answer_chain)

def get_session_history(session: str) -> BaseChatMessageHistory:
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

session_id = st.text_input("Session ID", value="default_session", key="session_id")
user_input = st.text_input("Your question:")
if user_input:
    session_history = get_session_history(session_id)
    try:
        response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        st.write("### Assistant:", response['answer'])
    except Exception as e:
        st.markdown(f"<div class='error-box'>⚠️ Error: {str(e)}</div>", unsafe_allow_html=True)

# QR Code Generation for Deployment
st.sidebar.header("📱 Access the Chatbot")
deployed_url = "https://mentalhealthbot-4ctgdhtdeeffjsswwkefw8.streamlit.app/"
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
qr.add_data(deployed_url)
qr.make(fit=True)
img = qr.make_image(fill="black", back_color="white")
buffer = BytesIO()
img.save(buffer, format="PNG")
buffer.seek(0)
st.sidebar.image(buffer, caption="Scan this QR code to visit the Chatbot", use_column_width=True)
