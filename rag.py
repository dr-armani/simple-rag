# Developed by Daniel Armani (August 2025)

!pip install -U langchain-google-genai langchain-community langchain-huggingface sentence-transformers chromadb pypdf

# Set up the embedding model

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding is set up using HuggingFace.")

# Set up the ingest function

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

persist_directory = "./chroma_db"
db = None

def ingest(pdf_path: str):
    global db
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return db

    db_exists = os.path.exists(persist_directory) and bool(os.listdir(persist_directory))
    if db_exists:
        print(f"\n Database found at {persist_directory}.")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # Check if any document with the same source path already exists:
        existing_docs = db.get(include=['metadatas'])
        for existing_doc in existing_docs['metadatas']:
            if existing_doc.get('source') == pdf_path:
                print(f"\n PDF file '{pdf_path}' already exists in the database. Skipping.")
                return db # Return db even if skipping addition
        print(f"\n Adding new documents.")
    else:
        print(f"\n Database not found. Creating a new database.")

    print(f"Processing {pdf_path}...")
    documents = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

    if len(chunks) > 1:
        print("\n chunk 0 : " + str(chunks[0]))
        print("\n chunk 1 : " + str(chunks[1]))

    # Add the new chunks to the database (either new or existing)
    if db_exists:
        db.add_documents(chunks)
    else:
        db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)


    db.persist()
    print(f"\n Successfully processed {pdf_path} and stored {len(chunks)} embeddings in {persist_directory}")
    return db # Return the updated or newly created db object

print("The function ingest() is created.")

# Set up the LLM

try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    # genai.configure(api_key=GOOGLE_API_KEY) # No need to configure genai separately if using ChatGoogleGenerativeAI
except userdata.SecretNotFoundError:
    print("Error: GOOGLE_API_KEY not found in Colab secrets.")
    GOOGLE_API_KEY = None # Set to None to avoid errors later
    import sys
    sys.exit(1)

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

print("LLM is set to Gemini.")

# Set up the retrieve function

def retrieve(question: str):

    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    # Print each chunk with its page number and file name
    print("Retrieved Document Chunks: ")
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page', 'N/A')
        print(f"--- Chunk {i+1} from {source} (Page {page}) ---")
        print(doc.page_content)
        print("-" * 40) # Separator

    # Combine the retrieved chunks as the context
    context = "\n".join([doc.page_content for doc in docs])

    import textwrap
    prompt = textwrap.dedent(f"""
        Answer the question based on the following context. If the question cannot be answered using the information in the context, only reply 'I don't know'.

        Question: {question}

        Context: {context}

        """)

    print ("\n Generated Prompt: ")
    print (prompt)
    print("\n Answer: ")
    answer = llm.invoke(prompt)
    print(answer.content)

print("The function retrieve() is created.")

# Ingest a PDF

file_path = input("Your File Path: ")

db = ingest(file_path)

if db:
    collection = db.get()
    print(f"\nNumber of total embedding vectors in the database: {len(collection['ids'])}")
else:
    print("File does not exist!")

# Answer a question

query = input("Your Question: ")
retrieve(query)
