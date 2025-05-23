from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import os, re, requests, sys

class RAGSystem:
    def __init__(self, base_url="http://localhost:11434"):
        # Initialization and model setup
        self.base_url = base_url
        self.dir = os.path.join(os.path.dirname(__file__), "vector_store")
        os.makedirs(self.dir, exist_ok=True)

        self.map = {"Larsen & Toubro": "India", "L&T": "India", "BYD Company Limited": "China", "BYD": "China"}
        self.embed_model = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
        self.llm = Ollama(model="llama3", base_url=base_url, temperature=0.1)
        self.vectorstore = None
        self.retrievers, self.qa_chains = {}, {}

    def load_and_chunk_pdf(self, path, company):
        # Load PDF and split into chunks with metadata
        if not os.path.exists(path): return []
        docs = PyPDFLoader(path).load()
        country = self.map.get(company, "Unknown")
        for d in docs: d.metadata.update({"source": company, "country": country})
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    def create_vector_store(self, docs):
        # Create and save vector store using FAISS
        self.vectorstore = FAISS.from_documents(docs, self.embed_model)
        self.vectorstore.save_local(self.dir)

    def load_vector_store(self):
        # Load existing FAISS vector store
        if not os.path.exists(self.dir): return False
        try:
            self.vectorstore = FAISS.load_local(self.dir, self.embed_model)
            return True
        except: return False

    def setup_retrievers(self):
        # Create retrievers filtered by company and country
        docs = self.vectorstore.as_retriever(search_kwargs={"k": 1000}).get_relevant_documents("test")
        for doc in docs:
            for key in ["source", "country"]:
                if key in doc.metadata:
                    self.retrievers.setdefault(doc.metadata[key], self.vectorstore.as_retriever(
                        search_kwargs={"k": 5, "filter": {key: doc.metadata[key]}}))

    def setup_qa_chains(self):
        # Set up QA chains for each retriever using prompt template
        prompt = PromptTemplate(
            template="Use the following context to answer the question:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"])
        for key, retriever in self.retrievers.items():
            self.qa_chains[key] = RetrievalQA.from_chain_type(
                self.llm, retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})

    def detect_entity(self, q):
        # Detect which company or country is referenced in the query
        for k in self.retrievers:
            if re.search(rf"\b{k}\b", q, re.IGNORECASE): return k

    def query(self, question):
        # Answer user query using detected entity or default to countries
        if not self.qa_chains: return {"error": "QA chains not initialized"}
        ent = self.detect_entity(question)
        if ent and ent in self.qa_chains:
            return {ent: self.qa_chains[ent].invoke(question)}
        return {c: self.qa_chains[c].invoke(question) for c in ["India", "China"] if c in self.qa_chains}

    def check_ollama(self):
        # Ensure Ollama is running with required models
        try:
            r = requests.get(f"{self.base_url}/api/tags")
            names = [m["name"] for m in r.json().get("models", [])]
            return all(m in names for m in ["llama3", "nomic-embed-text"])
        except: return False

    def initialize(self, pdf_paths, force=False):
        # Full initialization: load or build vector store, then prepare retrievers and QA
        if not self.check_ollama(): return False
        if force or not os.listdir(self.dir):
            docs = sum((self.load_and_chunk_pdf(p, c) for c, p in pdf_paths.items()), [])
            if not docs: return False
            self.create_vector_store(docs)
        else:
            if not self.load_vector_store(): return False
        self.setup_retrievers()
        self.setup_qa_chains()
        return True

if __name__ == "__main__":
    rag = RAGSystem()
    pdfs = {"L&T": "L&T_report.pdf", "BYD": "BYD_COMPANY_LIMITED_Report.pdf"}
    if len(sys.argv) > 2:
        pdfs["L&T"], pdfs["BYD"] = sys.argv[1], sys.argv[2]
    if not rag.initialize(pdfs, force=True): sys.exit("Initialization failed")

    print("Enter your queries (type 'exit' to quit):")
    while True:
        q = input("Query: ")
        if q.lower() in ["exit", "quit", "q"]: break
        res = rag.query(q)
        for ent, val in res.items():
            print(f"\n{ent}: {val['result'] if isinstance(val, dict) and 'result' in val else val}")
