from __future__ import annotations
from typing import TYPE_CHECKING

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
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = st.secrets.get("HUGGINGFACE_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN"))
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
os.environ["HUGGINGFACE_API_TOKEN"] = hf_token
os.environ["GROQ_API_KEY"] = groq_api_key

ChatGroq.BaseCache = InMemoryCache
ChatGroq.model_rebuild()

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "Data", "Dataset.csv")
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

with st.spinner('Building the FAISS vectorstore. Please wait...'):
    llm, retriever = load_model_and_vectorstore()
st.success("Vectorstore built successfully!")

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
    "You are a caring and empathetic mental health assistant with the nurturing guidance of a parent. "
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

session_id = st.text_input("Session ID", value="default_session", key="session_id")
user_input = st.text_input("Your question:")
if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke({"input": user_input}, config = {"configurable": {"session_id":session_id}})
    st.write("Assistant:", response['answer'])


# qrcode
#from io import BytesIO

#deployed_url = "https://mentalhealthchatbot-meqxamf2gnxrybe3vqytkc.streamlit.app/"

#qr = qrcode.QRCode(
 #   version=1,
 #   error_correction=qrcode.constants.ERROR_CORRECT_L,
  #  box_size=10,
   # border=4,
#)
#qr.add_data(deployed_url)
#qr.make(fit=True)
#img = qr.make_image(fill="black", back_color="white")
#buffer = BytesIO()
#img.save(buffer, format="PNG")
#buffer.seek(0)
#st.sidebar.image(buffer, caption="Scan this QR code to visit the Chatbot", use_column_width=True)"""
