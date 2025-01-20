%%writefile enhancedbot.py
import os
import glob
import torch
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline,
    BitsAndBytesConfig
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

@dataclass
class ChatbotConfig:
    """Configuration class for chatbot parameters"""
    max_new_tokens: int = 256
    max_input_tokens: int = 1024
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    min_confidence_threshold: float = 0.3
    max_conversation_history: int = 3
    escalation_threshold: int = 3

class EnhancedCustomerSupportBot:
    def __init__(self, config: ChatbotConfig):
        """Initialize the enhanced chatbot with improved configuration"""
        self.config = config
        self.memory = ConversationBufferMemory(memory_key="chat_history", k=config.max_conversation_history)
        self.consecutive_low_confidence = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialize_components()
        self._setup_prompt_templates()

    def _setup_prompt_templates(self):
        """Setup structured prompt templates for different scenarios"""
        self.fallback_template = PromptTemplate(
            input_variables=["query"],
            template="""
            I want to make sure I understand your request correctly.
            Query: {query}

            Could you please:
            1. Provide more details about your issue
            2. Specify what kind of assistance you need
            3. Let me know if any of my previous responses were helpful
            """
        )

    def initialize_components(self):
        """Initialize all required components with GPU support"""
        try:
            import warnings
            warnings.filterwarnings("ignore")

            print("Initializing enhanced NLP components...")
            print(f"Using device: {self.device}")

            if self.device == "cuda":
                torch.cuda.empty_cache()

            self.llm_pipeline = self._load_llm()
            print("✓ LLM initialized")

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            self.intent_pipeline = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                token="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",
                device=0 if self.device == "cuda" else -1
            )
            print("✓ Intent classifier initialized")

            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                token="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",
                device=0 if self.device == "cuda" else -1
            )
            print("✓ Sentiment analyzer initialized")

            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                token="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",
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
            print("✓ NER initialized")

            print("All components initialized successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def create_knowledge_base(self, datasets: List[pd.DataFrame]) -> FAISS:
        """Create knowledge base with memory-efficient settings"""
        try:
            chunk_size = 1000
            processed_chunks = []

            for df in datasets:
                df = df[['query', 'response']]
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size].copy()
                    chunk['text'] = chunk['query'] + " [SEP] " + chunk['response']
                    loader = DataFrameLoader(chunk, page_content_column="text")
                    processed_chunks.extend(loader.load())
                    del chunk

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': self.device}
            )

            return FAISS.from_documents(
                processed_chunks,
                embeddings,
                distance_strategy="cosine"
            )

        except Exception as e:
            raise ValueError(f"Failed to create knowledge base: {str(e)}")

    def _load_llm(self) -> Pipeline:
        """Load Llama 2 model with memory-efficient configuration"""
        try:
            model_name = "meta-llama/Llama-2-7b-chat-hf"

            if self.device == "cuda":
                torch.cuda.empty_cache()

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token="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"
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                token="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",
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                device_map="auto"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM: {str(e)}")

    def generate_response(self, query: str, knowledge_base: FAISS) -> str:
        """Generate a response to the user query using the knowledge base and NLP components"""
        try:
            sentiment_result = self.sentiment_pipeline(query)[0]
            sentiment = sentiment_result['label']

            intent_result = self.intent_pipeline(
                query,
                candidate_labels=["account_issue", "technical_support", "billing", "general_inquiry"],
                multi_label=False
            )
            intent = intent_result['labels'][0]
            intent_score = intent_result['scores'][0]

            relevant_docs = knowledge_base.similarity_search(query, k=2)
            context = "\n".join([doc.page_content for doc in relevant_docs])[:200]

            if intent_score < self.config.min_confidence_threshold:
                self.consecutive_low_confidence += 1
                if self.consecutive_low_confidence >= self.config.escalation_threshold:
                    return "I apologize, but I'm having trouble understanding your requests. Let me connect you with a human agent who can better assist you."
                return self.fallback_template.format(query=query)

            self.consecutive_low_confidence = 0

            prompt = f"You are a helpful customer service assistant. Based on the context: '{context}' and the user query: '{query}', provide a natural, conversational response focused only on addressing the user's immediate concern. Response:"

            response = self.llm_pipeline(
                prompt,
                do_sample=True,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=1,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                truncation=True
            )[0]['generated_text']

            response = response.strip()
            if response.startswith("Response:"):
                response = response[9:].strip()

            lines = response.split('\n')
            response = ' '.join([line for line in lines
                              if not (line.strip().startswith(('1.', '2.', '3.', '4.', '5.')) or
                                    'Related topics:' in line or
                                    'Follow-up questions:' in line)])

            response = ' '.join(response.split())

            self.memory.save_context({"input": query}, {"output": response})

            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team."

def main():
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure you have selected GPU runtime.")

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        config = ChatbotConfig()
        chatbot = EnhancedCustomerSupportBot(config)

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
                print(f"Warning: Could not identify query/response columns. Available columns: {df.columns}")
                return None

            df_processed = df[[query_col, response_col]].copy()
            df_processed.columns = ['query', 'response']

            df_processed = df_processed.dropna()

            return df_processed

        for path in dataset_paths:
            matching_files = glob.glob(path)

            for file_path in matching_files:
                if os.path.exists(file_path):
                    print(f"\nAttempting to load: {file_path}")
                    for encoding in encodings:
                        try:
                            print(f"Trying {encoding} encoding...")
                            chunks = pd.read_csv(file_path, encoding=encoding, chunksize=1000)
                            for chunk in chunks:
                                processed_chunk = process_dataset(chunk)
                                if processed_chunk is not None:
                                    datasets.append(processed_chunk)
                                    print(f"Successfully loaded chunk of size {len(processed_chunk)}")
                            print(f"Successfully loaded {file_path} with {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            print(f"Error loading file with {encoding}: {str(e)}")
                            continue
                else:
                    print(f"Warning: File not found - {file_path}")

        if not datasets:
            raise ValueError("No datasets could be loaded")

        total_samples = sum(len(df) for df in datasets)
        print(f"\nTotal number of samples loaded: {total_samples}")
        print(f"Number of datasets loaded: {len(datasets)}")

        knowledge_base = chatbot.create_knowledge_base(datasets)
        print("\nKnowledge base created successfully!")

        print("\nEnhanced Customer Support Chatbot initialized!")
        print("Type 'exit' to end the conversation or 'help' for assistance.")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                print("Thank you for using our customer support chatbot. Goodbye!")
                break

            if user_input.lower() == 'help':
                print("""
                Available commands:
                - 'exit': End the conversation
                - 'help': Show this help message

                Tips for better responses:
                - Be specific about your issue
                - Provide relevant details
                - Ask one question at a time
                """)
                continue

            response = chatbot.generate_response(user_input, knowledge_base)
            print(f"\nBot: {response}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
