# ============================================================
# RAG YouTube Summarizer - Complete Imports (OpenAI)
# ============================================================

# Standard Library
import os
import re
import json
from typing import List, Dict, Tuple, Optional

# UI Framework
import gradio as gr

# YouTube Processing
from youtube_transcript_api import YouTubeTranscriptApi

# Text Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI LLM & Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector Database
from langchain_community.vectorstores import FAISS

# Chains & Prompts
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# LangChain Core
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from yt_utils import get_transcript, process, generate_answer
from chunking import chunk_transcript
from model_setup import llm_model, embed_model
from faiss_db import create_faiss_index, perform_similarity_search
from prompt import create_summary_prompt, create_qa_prompt_template
from chain import create_summary_chain, create_qa_chain
from retriever import retrieve

# Environment Variables
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Sample YouTube URL
# url = "https://www.youtube.com/watch?v=T-D1OfcDW1M"


#transcript = get_transcript(url)
#text = process(transcript)
#chunks = chunk_transcript(text)
#print(f"Total Chunks: {len(chunks)}")
#print(f"First Chunk:\n{chunks[0]}")

#llm = llm_model()
#embeddings = embed_model()

#faiss_index = create_faiss_index(chunks, embeddings)
#query = "Provide a summary of the video content."
#results = perform_similarity_search(faiss_index, query, k=3)
#retrieved_docs = retrieve(query, faiss_index, k=3)
#print("Retrieved Documents:")

# Format retrieved docs into transcript string
#transcript_text = "\n".join([doc.page_content for doc in retrieved_docs])

#llm_chain = create_summary_chain(llm, create_summary_prompt(), verbose=True)
#summary = llm_chain.run(transcript=transcript_text)
#print("Summary:")
#print(summary)

# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

def summarize_video(video_url):
    """
    Title: Summarize Video
    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."
    if processed_transcript:
        # Step 1: Set up llm model
        llm = llm_model()
        # Step 2: embedding model setup
        embedding_model = embed_model()
        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)
        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."
    
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question
    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.
    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript
    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."
    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)
        # Step 1: Set up llm model
        llm = llm_model()
        # Step 2: embedding model setup
        embedding_model = embed_model()
        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        faiss_index = create_faiss_index(chunks, embedding_model)
        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)
        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."
    
with gr.Blocks() as interface:
    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)
    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")
    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)
    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)
# Launch the app with specified server name and port
interface.launch(server_name="localhost", server_port=7860)