import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page configuration
st.set_page_config(
    page_title="PDF Genius - Chat with your PDFs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved visibility
st.markdown("""
<style>
    /* Main title styles */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Sub-header styles */
    .sub-header {
        font-size: 1.5rem;
        color: #E0E0E0;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #0D47A1;
    }
    
    /* Response message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: rgba(30, 136, 229, 0.1);
        border-left: 5px solid #1E88E5;
    }
    
    /* Info boxes */
    .info-box {
        background-color: rgba(76, 175, 80, 0.1);
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
        color: #E0E0E0;
    }
    
    /* How it works section */
    .how-it-works {
        background-color: rgba(76, 175, 80, 0.1);
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1.5rem;
    }
    
    /* File uploader area */
    .upload-area {
        background-color: rgba(30, 136, 229, 0.1);
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Make labels more visible */
    label, .stTextInput label {
        color: #E0E0E0 !important;
        font-weight: 500 !important;
        font-size: 1.05rem !important;
    }
    
    /* Better text visibility throughout */
    p, li, ol, ul, h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0;
    }
    
    /* Make sure info box text is visible */
    .info-box p, .info-box li, .how-it-works p, .how-it-works li {
        color: #E0E0E0 !important;
    }
    
    /* Question text color */
    .stTextInput input {
        color: white !important;
    }
    
    /* Override Streamlit defaults */
    .css-145kmo2 {
        color: #E0E0E0 !important;
    }
    
    /* Help text styling */
    .help-text {
        color: #90CAF9 !important;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    with st.spinner("🤖 Generating response..."):
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)
        
        print(response)
        
        # Display the response with better styling
        st.markdown(f"""
        <div class="chat-message">
            <h4 style="color: #90CAF9;">Response:</h4>
            <p style="color: #E0E0E0;">{response["output_text"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Main function
def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="main-header" style="color: #1E88E5;">PDF Genius</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Upload & Process PDFs</div>', unsafe_allow_html=True)
        
        # How it works section - better styling
        st.markdown("""
        <div class="how-it-works">
            <p style="color: #E0E0E0; font-weight: bold; margin-bottom: 10px;">How it works:</p>
            <ol>
                <li style="color: #E0E0E0;">Upload one or more PDF files</li>
                <li style="color: #E0E0E0;">Click "Process PDFs" to extract and analyze content</li>
                <li style="color: #E0E0E0;">Ask questions about your documents in the main panel</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader with better styling
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown('<p style="color: #E0E0E0; font-weight: 500;">Upload your PDF files</p>', unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader("", 
                                    accept_multiple_files=True,
                                    type=["pdf"])
        
        # No empty box here anymore
        process_button = st.button("Process PDFs")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if process_button and pdf_docs:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Processing steps with visual feedback
            status_text.text("Extracting text from PDFs...")
            raw_text = get_pdf_text(pdf_docs)
            progress_bar.progress(33)
            
            status_text.text("Splitting text into chunks...")
            text_chunks = get_text_chunks(raw_text)
            progress_bar.progress(66)
            
            status_text.text("Creating vector embeddings...")
            get_vector_store(text_chunks)
            progress_bar.progress(100)
            
            st.success("✅ Processing complete! Your PDFs are ready for questions.")
            status_text.empty()
        
        st.markdown("---")
        st.markdown('<p style="color: #BBDEFB; font-size: 0.8rem;">Powered by Gemini 1.5 Flash</p>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 6])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/chatbot.png", width=70)
        
    with col2:
        st.markdown('<div class="main-header">Chat with your PDF Documents</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if documents have been processed
    if os.path.exists("faiss_index"):
        st.markdown("""
        <div class="info-box">
            <p style="color: #E0E0E0;">Ask any question about your uploaded PDF documents, and I'll provide answers based on their content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a container for the chat interface
        chat_container = st.container()
        
        with chat_container:
            # Query input with improved label visibility
            st.markdown('<p class="help-text">What would you like to know about your documents?</p>', unsafe_allow_html=True)
            user_question = st.text_input("", 
                                        placeholder="Ask a question...",
                                        key="user_question")
            
            if user_question:
                user_input(user_question)
    else:
        st.info("Please upload and process PDF documents using the sidebar first.")
        
        # Add a demo image or illustration
        col1, col2, col3 = st.columns([1, 2, 1])
       

# Run the app
if __name__ == "__main__":
    main()