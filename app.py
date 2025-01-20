%%writefile app.py
import streamlit as st
import torch
import pandas as pd
import os
import glob
from typing import List
from enhancedbot import EnhancedCustomerSupportBot, ChatbotConfig

def load_datasets():
    """Load and process all datasets"""
    datasets = []
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']

    dataset_paths = [
        '/content/banking_chatbot_dataset.csv',
        '/content/ecommerce_chatbot_dataset.csv',
        '/content/healthcare_chatbot_dataset.csv',
        '/content/telecom_chatbot_large_dataset.csv'
    ]

    def process_dataset(df):
        """Process different dataset formats and standardize columns"""
        query_columns = ['query', 'question', 'input', 'user_input', 'user_message', 'text']
        response_columns = ['response', 'answer', 'output', 'bot_response', 'reply']

        query_col = None
        for col in query_columns:
            if col in df.columns:
                query_col = col
                break

        response_col = None
        for col in response_columns:
            if col in df.columns:
                response_col = col
                break

        if query_col is None or response_col is None:
            st.warning(f"Warning: Could not identify query/response columns. Available columns: {df.columns}")
            return None

        df_processed = df[[query_col, response_col]].copy()
        df_processed.columns = ['query', 'response']
        df_processed = df_processed.dropna()
        return df_processed

    for path in dataset_paths:
        matching_files = glob.glob(path)
        for file_path in matching_files:
            if os.path.exists(file_path):
                st.info(f"Attempting to load: {file_path}")
                for encoding in encodings:
                    try:
                        chunks = pd.read_csv(file_path, encoding=encoding, chunksize=1000)
                        for chunk in chunks:
                            processed_chunk = process_dataset(chunk)
                            if processed_chunk is not None and not processed_chunk.empty:
                                datasets.append(processed_chunk)
                                st.info(f"Successfully loaded chunk of size {len(processed_chunk)}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        st.warning(f"Error loading file with {encoding}: {str(e)}")
                        continue
            else:
                st.warning(f"Warning: File not found - {file_path}")

    # If no datasets are loaded, create a small default dataset
    if not datasets:
        st.warning("No external datasets loaded. Using default dataset.")
        default_data = pd.DataFrame({
            'query': ['Hello', 'How are you?', 'What services do you offer?'],
            'response': ['Hi! How can I help you today?',
                        "I'm doing well, thank you! How can I assist you?",
                        "We offer a variety of customer support services. What specific assistance are you looking for?"]
        })
        datasets.append(default_data)

    return datasets

def initialize_chatbot():
    """Initialize the chatbot and knowledge base"""
    if 'chatbot' not in st.session_state:
        with st.spinner('Initializing chatbot... This may take a few minutes.'):
            config = ChatbotConfig()
            chatbot = EnhancedCustomerSupportBot(config)

            # Load and process datasets
            datasets = load_datasets()

            knowledge_base = chatbot.create_knowledge_base(datasets)

            st.session_state.chatbot = chatbot
            st.session_state.knowledge_base = knowledge_base
            st.session_state.messages = []

def main():
    st.title("Enhanced Customer Support Chatbot")
    st.markdown("""
    Welcome to the Enhanced Customer Support Chatbot!
    Type your message below and press Enter or click the 'Send' button.
    """)

    # Initialize the chatbot
    initialize_chatbot()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What can I help you with today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(
                    prompt,
                    st.session_state.knowledge_base
                )
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
