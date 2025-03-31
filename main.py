from __future__ import annotations  # Must be the first line

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from typing import TYPE_CHECKING

# High-contrast Styling for Better Visibility
st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;  /* White background for clear contrast */
            font-family: Arial, sans-serif;
            color: #000000; /* Black text for readability */
        }
        .main {
            background: #f0f0f0; /* Light gray for subtle contrast */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        }
        .stTextInput>label, .stTextArea>label {
            font-size: 16px;
            font-weight: bold;
            color: #000000; /* Black labels */
        }
        .stButton>button {
            background-color: #0057b8; /* Dark blue for strong contrast */
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #003d82; /* Slightly darker blue on hover */
        }
        .stSuccess {
            background-color: #28a745 !important; /* Green success box */
            color: white !important;
            padding: 10px;
            border-radius: 5px;
        }
        .stWarning {
            background-color: #ffcc00 !important; /* Yellow warning */
            color: black !important;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Heading with Clear Visibility
st.title("💙 Happiness Chatbot 💙")
st.write("**A supportive chatbot for mental well-being.**")

if TYPE_CHECKING:
    from langchain.callbacks.base import BaseCallbackHandler as Callbacks
else:
    class Callbacks:
        pass

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
hf_token = st.secrets.get("HUGGINGFACE_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN"))
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
os.environ["HUGGINGFACE_API_TOKEN"] = hf_token
os.environ["GROQ_API_KEY"] = groq_api_key

ChatGroq.BaseCache = InMemoryCache
ChatGroq.model_rebuild()

# Load Dataset
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "DATA", "Dataset.csv")
dataset = pd.read_csv(dataset_path)

@st.cache_resource
def load_model_and_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = dataset['Response'].fillna("").astype(str).tolist()
    metadata = [{"label": label} for label in dataset['Response'].fillna("").astype(str).tolist()]
    vectorstore = FAISS.from_texts(documents, embedder, metadatas=metadata)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    retriever = vectorstore.as_retriever()
    return llm, retriever

with st.spinner('⏳ Building the FAISS vectorstore... Please wait...'):
    llm, retriever = load_model_and_vectorstore()
st.success("✅ Vectorstore built successfully!")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user input, which may reference previous context, "
    "rephrase the user's question into a clear, standalone query using gentle and simple language suitable for children. "
    "Provide some advice and try to give answers in a caring, parental tone."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_chain = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "As a compassionate and supportive mental health assistant, you provide guidance with the warmth and understanding of a caring mentor."
    "Use the following pieces of retrieved context to address the user's concern in a way that is both supportive and age-appropriate. "
    "If you are uncertain about the answer, kindly acknowledge it and suggest seeking further support from a trusted adult or professional. "
    "You also have language translation capabilities to assist users in different languages."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
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

# User Input Section with Clear Contrast
st.subheader("🗨️ Chat with the Happiness Chatbot")
session_id = st.text_input("🔑 **Session ID:**", value="default_session", key="session_id")
user_input = st.text_area("✏️ **Type your question here:**")

if st.button("Send"):
    if user_input.strip():
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        st.success(f"**Assistant:** {response['answer']}")
    else:
        st.warning("⚠️ Please enter a question before sending.")

# Generate QR Code for Easy Access
import qrcode
from io import BytesIO

deployed_url = "https://mentalhealthbot-4ctgdhtdeeffjsswwkefw8.streamlit.app/"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(deployed_url)
qr.make(fit=True)
img = qr.make_image(fill="black", back_color="white")
buffer = BytesIO()
img.save(buffer, format="PNG")
buffer.seek(0)

# Sidebar with QR Code
st.sidebar.header("🔗 **Access the Chatbot**")
st.sidebar.image(buffer, caption="📱 Scan the QR code to chat!", use_column_width=True)
