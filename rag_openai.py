import os
import streamlit as st
from io import BytesIO
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile

st.set_page_config(page_title="RAG | BYD & L&T Analyzer", layout="wide")
st.title("📊 Financial Report RAG – BYD 🇨🇳 vs L&T 🇮🇳")

# 1. API Key Input
openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# 2. PDF Upload
uploaded_files = st.file_uploader("📂 Upload BYD and L&T Reports (PDFs)", type=["pdf"], accept_multiple_files=True)

query = st.text_input("💬 Ask your question about these reports")

def classify_query(q):
    q_lower = q.lower()
    if any(word in q_lower for word in ["china", "byd", "hkd", "shenzhen", "rmb", "hong kong"]):
        return "BYD"
    elif any(word in q_lower for word in ["india", "l&t", "mumbai", "inr", "toubro", "larsen"]):
        return "L&T"
    return "BOTH"

# Main RAG Logic
if openai_key and uploaded_files and query:
    os.environ["OPENAI_API_KEY"] = openai_key

    with st.spinner("🔍 Indexing documents and searching..."):
        # 3. Load PDFs
        docs = []
        metadatas = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for file in uploaded_files:
            filename = file.name.lower()
            label = "BYD" if "byd" in filename else "L&T" if "l&t" in filename else "UNKNOWN"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load()
                for page in pages:
                    chunks = splitter.split_text(page.page_content)
                    docs.extend(chunks)
                    metadatas.extend([{"source": label}] * len(chunks))

        # 4. Embed & create FAISS vectorstore
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(docs, embedding, metadatas)

        # 5. Query Routing
        target = classify_query(query)
        if target == "BOTH":
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": {"source": target}})

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa({"query": query})

        # 6. Display Results
        st.markdown("### 🧠 Answer")
        st.write(result["result"])

        st.markdown("### 📚 Source Snippets")
        for doc in result["source_documents"]:
            st.info(f"**From {doc.metadata['source']}**:\n\n{doc.page_content[:500]}...")

elif not openai_key:
    st.warning("🔐 Please enter your OpenAI API key.")
elif not uploaded_files:
    st.warning("📁 Please upload at least two PDF files: one for BYD and one for L&T.")
elif not query:
    st.info("💡 Enter a question to get started.")
