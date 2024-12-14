import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_KEY")

# Load PDF and process documents
loader = PyPDFLoader("data/Trading Bot using Reinforcement Learning.pdf")
document = loader.load()

# Create embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store with FAISS
vectorstore = FAISS.from_documents(document, embedding=embedding)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Define the LLM (Google Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# System prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know the answer. "
    "Be concise.\n\n{context}"
)

def answer_question(query):
    """Handles user queries and provides an answer using RAG."""
    if not query:
        return "Please enter a valid question."

    # Create the chat prompt
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create RAG chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Execute the RAG chain
    response = rag_chain.invoke({"input": query})

    return response["answer"]

# Gradio UI setup
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Simple RAG using Gemini")
        with gr.Row():
            query_input = gr.Textbox(label="Ask a question about Trading Bot using Reinforcement Learning")
            answer_output = gr.Textbox(label="Answer")
        ask_button = gr.Button("Get Answer")

        ask_button.click(answer_question, inputs=query_input, outputs=answer_output)

    return demo

# Launch Gradio app
demo = main()
demo.launch()

