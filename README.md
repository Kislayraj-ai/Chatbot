# 🔍 RAG Chatbot – Retrieval-Augmented Generation with Google Gemini

A simple RAG (Retrieval-Augmented Generation) pipeline built with **LangChain** and **Google Gemini AI** that lets you chat with your own documents.

> 📝 **Note:** Local model support via **Ollama (DMR)** will be added in a future update.

---

## 🧠 What is RAG?

RAG combines document retrieval with AI generation. Instead of relying solely on the LLM's training data, it:

1. Loads your documents
2. Splits them into chunks
3. Stores them as vector embeddings
4. Retrieves relevant chunks based on your query
5. Passes them to the LLM to generate a grounded answer

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | Google Gemini (`gemini-flash-lite-latest`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Env Management | python-dotenv |

---

## 📦 Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd RAG_Chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install langchain langchain-google-genai langchain-community \
            langchain-text-splitters chromadb python-dotenv
```

---

## 🔑 Setup API Key

Create a `.env` file in the root directory:

```
API_KEY=your_google_gemini_api_key_here
```

Get your free API key at: https://aistudio.google.com/app/apikey

---

## 🚀 How to Run

```bash
python chatbot.py
```

Then type your questions in the terminal:

```
>> What is the solar system?
>> How many planets are there?
>> exit
```

---

## 📁 Project Structure

```
RAG_Chatbot/
│
├── chatbot.py          # Main chatbot loop
├── 1.0_rag_ownt.ipynb  # Jupyter notebook (development)
├── .env                # API key (do NOT commit this)
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Pipeline Steps

### Step 1 – Load Document
```python
from langchain_core.documents import Document
doc = [Document(page_content=my_text, metadata={"source": "ABC"})]
```

### Step 2 – Split into Chunks
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(doc)
```

### Step 3 – Create Embeddings
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get('API_KEY')
)
```

### Step 4 – Store in Vector DB
```python
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
```

### Step 5 – Query with Relevance Threshold
```python
results = vectorstore.similarity_search_with_score(query, k=1)
doc, score = results[0]
if score > 0.75:
    print("Not relevant!")
else:
    print(doc.page_content)
```

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `SyntaxError` on step labels | Missing `#` before comment lines | Add `#` to all plain text lines |
| `404 NOT_FOUND` on embedding | Wrong model name | Use `models/gemini-embedding-001` |
| `429 RESOURCE_EXHAUSTED` | Free tier rate limit hit | Add `time.sleep(1)` between calls |
| `KeyError: API_KEY` | `.env` not loaded or key missing | Use `os.environ.get('API_KEY')` |

---

## 📊 Similarity Score Guide (ChromaDB L2 Distance)

| Score Range | Meaning |
|---|---|
| `0.0 – 0.5` | Highly relevant ✅ |
| `0.5 – 0.75` | Somewhat relevant 🟡 |
| `> 0.75` | Not relevant ❌ (blocked) |

---

## 🔮 Upcoming Features

- [ ] Local model support via **Ollama (DMR)**
- [ ] PDF document loading
- [ ] Multi-document support
- [ ] Conversational memory

---

## 📄 License

MIT License
