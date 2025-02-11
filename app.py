import streamlit as st
import os
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def read_doc(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunks(docs, chunksize=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=chunksize, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)
    return docs

def initialize_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = PineconeClient(api_key=pinecone_api_key)
    if "langqa1" not in pc.list_indexes().names():
        pc.create_index(
            index_name="langqa1",
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc

def log_results(pdf_name, sentiment, summary, feedback):
    with open("logs.txt", "a") as log_file:
        log_file.write(f"PDF: {pdf_name}\nSentiment: {sentiment}\nSummary: {summary}\nFeedback: {feedback}\n{'-'*40}\n")

def extract_data_and_store(data, embeddings):
    pc = initialize_pinecone()
    index = Pinecone.from_documents(data, embeddings, index_name="langqa1")
    return index

def analyze_sentiment(llm, text):
    prompt_template = PromptTemplate(
        input_variables=["document"],
        template="Analyze the sentiment of the following document. Is it Positive, Neutral, or Negative?\n{document}"
    )
    prompt = prompt_template.format(document=text)
    try:
        sentiment = llm.predict(prompt).strip()
    except Exception:
        st.error("Sentiment analysis failed. Retrying with another model...")
        llm = ChatGroq(temperature=0.5, model="gemma2-9b-it")
        sentiment = llm.predict(prompt).strip()
    return sentiment

def rephrase_content(llm, original_content):
    rephrase_prompt = f"Reword the following content to provide an alternative version:\n{original_content}"
    return llm.predict(rephrase_prompt).strip()

def summarize_text(llm, text_chunks):
    summaries = []
    prompt_template = PromptTemplate(
        input_variables=["document"],
        template="Summarize the following document concisely:\n{document}"
    )
    
    for chunk in text_chunks[:10]:
        prompt = prompt_template.format(document=chunk)
        try:
            summary = llm.predict(prompt)
        except Exception:
            st.error("Summarization failed. Retrying with another model...")
            llm = ChatGroq(temperature=0.5, model="gemma2-9b-it")
            summary = llm.predict(prompt)
        summaries.append(summary)
    
    final_prompt = "Combine the following summaries into a final concise summary:\n" + "\n".join(summaries)
    try:
        final_summary = llm.predict(final_prompt)
    except Exception:
        st.error("Final summarization failed. Retrying...")
        final_summary = llm.predict(final_prompt)
    return final_summary

def main():
    st.title("PDF Task Execution & Verification App")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        pdf_name = uploaded_file.name
        st.write("Processing PDF...")
        doc = read_doc(file_path)
        data = chunks(doc)
        
        st.write("Extracting data and storing in Pinecone...")
        embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
        index = extract_data_and_store(data, embeddings)
        llm = ChatGroq(temperature=0.5, model="llama3-8b-8192")
        
        st.write("Analyzing sentiment...")
        sentiment = analyze_sentiment(llm, " ".join([chunk.page_content for chunk in data]))
        st.write("**Sentiment Analysis:**", sentiment)
        
        if "sentiment_confirmed" not in st.session_state:
            st.session_state.sentiment_confirmed = False
        
        if not st.session_state.sentiment_confirmed:
            if st.button("Confirm Sentiment Analysis & Proceed"):
                st.session_state.sentiment_confirmed = True
                st.rerun()
            elif st.button("Rephrase Sentiment Analysis"):
                sentiment = rephrase_content(llm, sentiment)
                st.write("**Updated Sentiment Analysis:**", sentiment)
            return
        
        st.write("Generating summary...")
        summary = summarize_text(llm, data)
        st.write("**Summary:**", summary)
        
        if "summary_confirmed" not in st.session_state:
            st.session_state.summary_confirmed = False
        
        if not st.session_state.summary_confirmed:
            if st.button("Confirm Summary & Proceed"):
                st.session_state.summary_confirmed = True
                st.rerun()
            elif st.button("Rephrase Summary"):
                summary = rephrase_content(llm, summary)
                st.write("**Updated Summary:**", summary)
            return
        
        feedback = st.text_area("Do you agree with the analysis? Provide feedback:")
        if st.button("Submit Feedback"):
            log_results(pdf_name, sentiment, summary, feedback)
            st.success("Results and feedback logged successfully!")

if __name__ == "__main__":
    main()
