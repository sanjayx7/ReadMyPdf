import gradio as gr
import os
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
import faiss
import numpy as np

dotenv.load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Global variable to store documents
docs = []

# Initializing the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
print("Groq model initialized.")

# Initialize Google Embeddings
print("Initializing GoogleGenAI Embeddings...")
embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Replace with the appropriate model name if needed
    api_key=os.getenv("GOOGLE_API_KEY")
)
print("Embeddings initialized.")

# Predefined responses for greetings and specific queries
predefined_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help?",
    "how can you help me" : "You can ask any questions related to the pdf you are uploading ",
    "how are you": "I'm just a virtual assistant, but I'm here and ready to help!",
    "who made you": "I was created by Jeny M Jerry.She made this in one day and shipped it for greater good.",
    "who created you": "Jeny M Jerry is my creator. She built this system for a better, more efficient way to navigate through PDF documents.",
    "thank you": "You're welcome! Let me know if there's anything else you need.",
    "thanks": "You're welcome! Have a great day!",
    "bye": "Goodbye! Take care.",
    "ok":"😃",
    "why was you created": (
        "I was created to address the problem of endless scrolling through PDF pages. "
        "Many people struggled to find information quickly, so this app was designed to make searching within PDFs fast and efficient."
    )
}

# Function to load and add documents to FAISS


def add_docs(path):
    global docs
    print("Loading documents from:", path)
    try:
        loader = PyPDFLoader(file_path=path)
        docs = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=300,
                length_function=len,
                is_separator_regex=False
            )
        )
        print(f"Loaded {len(docs)} document chunks.")

        # Embed the documents using the embedding function
        # Extract text from docs
        doc_texts = [doc.page_content for doc in docs]
        embeddings = embedding_function.embed_documents(doc_texts)  # Get embeddings for all documents

        # Convert embeddings to numpy array for FAISS indexing
        faiss_vectors = np.array(embeddings, dtype='float32')

        # Initialize FAISS index and add vectors
        # Using L2 distance for similarity search
        index = faiss.IndexFlatL2(faiss_vectors.shape[1])
        index.add(faiss_vectors)  # Add the vectors to the index

        os.makedirs("output", exist_ok=True)
        faiss.write_index(index, "output/faiss_index.index")

        print("Documents added to FAISS vector store.")
    except Exception as e:
        print("Error in add_docs:", e)

# Function to answer a query based on stored documents


def answer_query(message, chat_history):
    global docs
    print("Received query:", message)

    # Check for predefined responses
    lower_message = message.lower().strip()
    if lower_message in predefined_responses:
        response_content = predefined_responses[lower_message]
        formatted_response = {
            "role": "assistant",
            "content": response_content
        }

        # Update chat history
        chat_history.append({"role": "user", "content": message})
        chat_history.append(formatted_response)
        print("Predefined response generated:", response_content)
        return "", chat_history

    # If not a predefined response, proceed with RAG pipeline
    try:
        # Load FAISS index from disk
        index_path = "output/faiss_index.index"
        if not os.path.exists(index_path):
            print("FAISS index file not found. Please upload documents first.")
            return "No index found. Please upload a PDF document first.", chat_history

        index = faiss.read_index(index_path)

        # Embed the query using the embedding function
        query_vector = embedding_function.embed_query(
            message)  # Get the embedding for the query
        query_vector = np.array([query_vector], dtype='float32')

        # Search for the nearest vectors (using the query vector)
        D, I = index.search(query_vector, k=5)  # Get top 5 similar documents
        print(f"Found {len(I)} similar documents.")

        # Retrieve the documents (in practice, you'd map indices back to documents)
        context = "\n\n".join([docs[i].page_content for i in I[0]])

        # Define the prompt template
        template = """
        You are a knowledgeable assistant. Use the context provided to answer the question in detail and with clarity. 
        Structure your response into clear paragraphs and provide additional insights if they are relevant, but do not use any information or knowledge outside the context.
        If the given context does not have the information, respond: 'I don't know the answer to this question.'
        Context: ```{context}```
        ----------------------------
        Question: {query}
        ----------------------------
        Answer: """

        # Generate prompt
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            template=template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        prompt = chat_prompt.format_prompt(query=message, context=context)

        response = llm.invoke(input=prompt.to_messages())

        # Ensure the response is in the correct format
        formatted_response = {
            "role": "assistant",
            "content": response.content
        }

        # Update chat history
        chat_history.append({"role": "user", "content": message})
        chat_history.append(formatted_response)
        print("Response generated:", response.content)
        return "", chat_history
    except Exception as e:
        print("Error in answer_query:", e)
        return "", chat_history


# Build Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 align='center'> AI PDF Finder 📄🤖</h1>")

    with gr.Row():
        upload_files = gr.File(label='Upload a PDF', file_types=[
                               '.pdf'], file_count='single')

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Enter your question here")
    upload_files.upload(add_docs, upload_files)
    msg.submit(answer_query, [msg, chatbot], [msg, chatbot])

print("Launching Gradio app...")
demo.launch()
print("App is running...")
