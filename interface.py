import os
import sys
import textwrap
from typing import Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown
from core import RAGSystem, logger

# Initialize Rich console
console = Console()

def format_source_document(doc) -> str:
    """Format source document for display"""
    source = doc.metadata.get("source", "Unknown source")
    content = textwrap.fill(doc.page_content, width=100)
    return f"\nSource: {source}\nContent: {content}\n"

def chat_loop(qa_chain):
    """Interactive chat loop"""
    logger.info("Starting chat loop")
    
    # Clear any previous content
    console.clear()
    
    console.print(Panel.fit(
        "[bold green]Welcome to the Document Q&A System![/bold green]\n"
        "Type 'quit', 'exit', or press Ctrl+C to end the conversation.\n"
        "Type 'sources' to see the full source documents for the last answer.",
        title="Chat System"
    ))
    
    last_source_docs = None
    
    while True:
        try:
            query = Prompt.ask("\n[bold blue]Your question[/bold blue]")
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("Chat session ended by user")
                console.print("[bold green]Goodbye![/bold green]")
                break
                
            if query.lower() == 'sources' and last_source_docs:
                console.print("\n[bold]Detailed Source Documents[/bold]")
                for i, doc in enumerate(last_source_docs, 1):
                    console.print(Panel(
                        format_source_document(doc),
                        title=f"Source Document {i}",
                        border_style="blue"
                    ))
                continue
                
            if not query:
                console.print("[yellow]Please enter a question.[/yellow]")
                continue
            
            # Show thinking status
            with console.status("[bold green]Thinking...", spinner="dots"):
                logger.info(f"Processing query: {query}")
                response = qa_chain.invoke({"query": query})
                logger.info("Response generated successfully")
            
            last_source_docs = response.get('source_documents', [])
            
            # Display response in a panel
            console.print(Panel(
                response['result'],
                title="Answer",
                border_style="green"
            ))
            
            # Display sources in a table
            if last_source_docs:
                sources_table = Table(title="Sources")
                sources_table.add_column("№", style="cyan")
                sources_table.add_column("Source", style="green")
                
                for i, doc in enumerate(last_source_docs, 1):
                    source = doc.metadata.get("source", "Unknown source")
                    sources_table.add_row(str(i), source)
                
                console.print(sources_table)
                console.print("\nType 'sources' to see the full source documents.")
            
        except KeyboardInterrupt:
            logger.info("Chat session interrupted by user")
            console.print("\n[bold green]Goodbye![/bold green]")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
            console.print("[yellow]Please try asking your question in a different way.[/yellow]")

# 1. Index documents and create vector store 
def process_documents_with_status(rag: RAGSystem) -> Optional[Any]:
    """Process documents with status updates"""
    try:
        console.print("[bold blue]Loading documents...[/bold blue]")
        documents = rag.load_documents("./documents")
        
        if documents:
            console.print(f"[bold blue]Processing {len(documents)} documents...[/bold blue]")
            texts = rag.process_documents(documents)
            
            if texts:
                console.print("[bold blue]Creating vector store...[/bold blue]")
                vectordb = rag.create_vector_store(texts)
                return vectordb
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
    return None

def display_menu() -> str:
    """Display the main menu"""
    menu = Table(title="RAG System Menu", show_header=False, show_lines=True)
    menu.add_column("Option", style="cyan")
    menu.add_row("1. Index documents")
    menu.add_row("2. Check total number of documents")
    menu.add_row("3. Delete document store")
    menu.add_row("4. Start RAG chat")
    menu.add_row("5. Exit")
    
    console.print(menu)
    return Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4", "5"])

# 4. Start chat
def start_chat(rag: RAGSystem):
    """Initialize and start the chat system"""
    vectordb = None
    qa_chain = None
    
    try:
        # First try to load the vector store
        console.print("[bold blue]Loading vector store...[/bold blue]")
        vectordb = rag.load_vector_store()
        
        if not vectordb:
            console.print("[bold red]No document store found. Please index documents first.[/bold red]")
            return
            
        # Create QA chain
        console.print("[bold blue]Creating QA chain...[/bold blue]")
        qa_chain = rag.create_qa_chain(vectordb)
        
        if not qa_chain:
            console.print("[bold red]Failed to create QA chain.[/bold red]")
            return
            
        # Start chat loop
        console.print("[bold green]Chat system ready![/bold green]")
        chat_loop(qa_chain)
        
    except Exception as e:
        logger.error(f"Error starting chat: {str(e)}")
        console.print(f"\n[bold red]Failed to start chat:[/bold red] {str(e)}")
        
        if not vectordb:
            console.print("[yellow]Hint: Make sure you have indexed documents first (Option 1)[/yellow]")
        elif not qa_chain:
            console.print("[yellow]Hint: There might be an issue with the OpenAI API key or connection[/yellow]")

def main():
    """Main function"""
    try:
        logger.info("Starting RAG System")
        rag = RAGSystem()
        
        while True:
            choice = display_menu()
            
            if choice == "1":
                logger.info("User selected: Index documents")
                vectordb = process_documents_with_status(rag)
                if vectordb:
                    console.print("[bold green]Documents indexed successfully![/bold green]")
                else:
                    console.print("[bold red]Failed to index documents.[/bold red]")
                    
            elif choice == "2":
                logger.info("User selected: Check document count")
                console.print("[bold blue]Getting document count...[/bold blue]")
                count = rag.get_document_count()
                console.print(f"\nTotal documents: [bold green]{count}[/bold green]")
                
            elif choice == "3":
                logger.info("User selected: Delete document store")
                console.print("[bold red]Deleting document store...[/bold red]")
                if rag.delete_vector_store():
                    console.print("[bold green]Document store deleted successfully![/bold green]")
                else:
                    console.print("[bold red]Failed to delete document store.[/bold red]")
                
            elif choice == "4":
                logger.info("User selected: Start RAG chat")
                start_chat(rag)
                
            elif choice == "5":
                logger.info("User selected: Exit")
                console.print("[bold green]Goodbye![/bold green]")
                break
            
            # Add a small pause between operations
            console.print("\nPress Enter to continue...")
            input()

    except KeyboardInterrupt:
        console.print("\n[bold green]Goodbye![/bold green]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        console.print(f"\n[bold red]A fatal error occurred:[/bold red] {str(e)}")
        sys.exit(1)

def check_environment():
    """Check and validate environment setup"""
    # Check for required directories
    if not os.path.exists("documents"):
        os.makedirs("documents")
        console.print("[yellow]Created 'documents' directory[/yellow]")
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
        console.print("[yellow]Created 'logs' directory[/yellow]")
    
    # Check for .env file
    if not os.path.exists(".env"):
        console.print("[bold red]Error: .env file not found![/bold red]")
        console.print(Panel(
            "Please create a .env file with the following contents:\n\n" +
            "OPENAI_API_KEY=your-api-key-here\n" +
            "MODEL_NAME=gpt-3.5-turbo\n" +
            "COLLECTION_NAME=my_documents\n" +
            "PERSIST_DIRECTORY=db",
            title="Required Configuration",
            border_style="red"
        ))
        return False
    
    return True

if __name__ == "__main__":
    try:
        # Show welcome banner
        console.print(Panel.fit(
            "[bold blue]RAG System[/bold blue]\n" +
            "Document Q&A System using LangChain and ChromaDB",
            border_style="blue"
        ))
        
        # Check environment
        if not check_environment():
            sys.exit(1)
            
        # Run main application
        main()
        
    except KeyboardInterrupt:
        console.print("\n[bold green]Goodbye![/bold green]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        console.print(Panel(
            str(e),
            title="Error Details",
            border_style="red"
        ))
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass