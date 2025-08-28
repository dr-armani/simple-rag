# Building a simple RAG using LangChain  

In 2011, HP acquired Autonomy for approximately $11 billion. Within a year, HP recorded an $8.8 billion impairment charge, alleging materially overstated revenues and margins. The dispute produced extensive litigation and even criminal prosecutions.  

HP’s due diligence relied on high-level representations and summaries rather than detailed contract-to-cash reconciliation. Critical gaps included insufficient cross-checking of invoices against recognized revenue, inadequate probing of reseller arrangements and side letters, and incomplete follow-ups prior to closing, which removed HP’s leverage to require contract fixes, escrows, or price adjustments.  

This M&A failure was driven by document-level risks that were not detectable through routine ledger analytics. The decisive evidence did not exist in summary financials but in transaction-level documentation: contracts, side letters, invoices, and accounting work papers. In many instances, transactions were described and contracted as software sales, whereas they were actually hardware deals with much smaller margins. This highlights the need to analyze vast volumes of unstructured documents while preserving auditable provenance.  

Retrieval-Augmented Generation (RAG) addresses this need by combining semantic search over indexed source documents with LLM-based synthesis that cites the originating passages. A RAG workflow could materially increase the probability of surfacing contract and booking anomalies before close by enabling targeted retrieval of contract language and by producing evidence-based summaries tied to source documents.  

A minimal RAG pipeline has two functions:  

**Ingest:** Loads source documents (PDFs, emails, contracts, etc.), splits them into chunks, computes their embedding vectors using an embedding model (cloud or local), and stores the vectors in a database such as Pinecone or FAISS. Each vector is indexed and linked back to its originating source, preserving a full audit trail.  

**Retrieve:** Converts the user query into an embedding vector, searches the vector database for the most similar chunks (usually via cosine similarity), and assembles them into a prompt:  

"""  
Answer this question using the following context:  
 Question: {query}  
 Context: {chunks}  
"""  
 
This augmented prompt is then sent to an LLM like OpenAI or Gemini, which returns an evidence-based response with citations to the retrieved passages.  

In my experience, while open-source LLMs (e.g., Hugging Face, Ollama) can support RAG, their out-of-the-box performance is not acceptable for enterprise-grade due diligence tasks, and you need to dedicate resources for fine-tuning as well as hosting.  

Finally, while RAG can reduce hallucinations, it cannot eliminate them entirely. You may use another tool to double-check responses. 

Meanwhile, there are many RAG diagrams available, but I was not convinced that any were clear enough. So I created this diagram. Please let me know if you’ve seen a better version.  

![Simple RAG Diagram](https://github.com/dr-armani/simple-rag/blob/main/RAG.jpg)
