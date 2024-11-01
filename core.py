import os
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import logging
import shutil
from rich.logging import RichHandler

class CustomFileHandler(logging.FileHandler):
    """Custom file handler that ensures UTF-8 encoding"""
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

class CustomTextLoader:
    """Custom document loader with enhanced encoding support"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load and return documents from a single file"""
        return list(self.lazy_load())

    def lazy_load(self):
        """Generator method to lazily load documents"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(self.file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    metadata = {"source": self.file_path}
                    yield Document(page_content=text, metadata=metadata)
                    break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading file {self.file_path}: {str(e)}")
                return

def setup_logger():
    """Configure logging with Rich and timestamp"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/rag_system_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            CustomFileHandler(log_filename, encoding='utf-8'),
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=False,
                enable_link_path=False
            )
        ]
    )
    
    return logging.getLogger("rich")

logger = setup_logger()


class RAGSystem:
    """Main RAG system implementation"""
    
    def __init__(self):
        logger.info("Initializing RAG System")
        load_dotenv()
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.persist_directory = os.getenv("PERSIST_DIRECTORY")
        
        # Initialize components
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=self.model_name
        )
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        logger.info("RAG System initialized successfully")

    def load_documents(self, directory_path: str) -> List[Document]:
        """Load documents from a directory"""
        logger.info(f"Loading documents from {directory_path}")
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=CustomTextLoader
            )
            documents = loader.load()
            if not documents:
                logger.warning("No documents were loaded")
            else:
                logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error in load_documents: {str(e)}")
            return []

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info("Processing documents into chunks")
        if not documents:
            return []
        try:
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(texts)} text chunks")
            return texts
        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}")
            return []

    def create_vector_store(self, texts: List[Document]) -> Optional[Chroma]:
        """Create vector store"""
        logger.info("Creating vector store")
        if not texts:
            logger.warning("No texts to process. Vector store will be empty.")
            return None
            
        try:
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logger.info("Vector store created successfully")
            return vectordb
        except Exception as e:
            logger.error(f"Error in create_vector_store: {str(e)}")
            return None

    def load_vector_store(self) -> Optional[Chroma]:
        """Load existing vector store"""
        logger.info("Loading existing vector store")
        try:
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info("Vector store loaded successfully")
            return vectordb
        except Exception as e:
            logger.error(f"Error in load_vector_store: {str(e)}")
            return None

    def delete_vector_store(self) -> bool:
        """Delete the vector store"""
        logger.info("Deleting vector store")
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info("Vector store deleted successfully")
                return True
            else:
                logger.warning("Vector store directory does not exist")
                return False
        except Exception as e:
            logger.error(f"Error deleting vector store: {str(e)}")
            return False

    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store"""
        logger.info("Getting document count")
        try:
            vectordb = self.load_vector_store()
            if vectordb:
                count = len(vectordb.get()['ids'])
                logger.info(f"Found {count} documents in vector store")
                return count
            else:
                logger.warning("No vector store found")
                return 0
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0

    def create_qa_chain(self, vectordb: Chroma) -> Optional[RetrievalQA]:
        """Create QA chain with custom prompt"""
        logger.info("Creating QA chain")
        if not vectordb:
            raise ValueError("Vector store is empty or not initialized")
            
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )
        
        logger.info("QA chain created successfully")
        return qa_chain