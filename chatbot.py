import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
if os.environ.get('API_KEY'):
    print("API KEY IS Set.", os.environ.get('API_KEY'))
local_llm =  ChatGoogleGenerativeAI(api_key=os.environ.get('API_KEY'),
                model="gemini-flash-lite-latest"
)


doc_path = "./knowledge/solar_system.pdf"
local_loader = PyPDFLoader(doc_path)

docs = local_loader.load()

for info in docs:
    info.metadata = {
        "producer" : "kislay_raj",
        "source"   : "google",
        "created_on" :  datetime.now().strftime("%Y-%m-%d")
    }

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

get_chunks =  splitter.split_documents(docs)


embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get('API_KEY')
)


vectorDB =  Chroma.from_documents(
    documents=get_chunks,
    embedding=embedding_model,
    persist_directory="./vector/"
)

THRESHOLD = 0.75

while True:
    stringInput = input(">> ")
    if stringInput.lower() == 'exit':
        break

    search_results = vectorDB.similarity_search_with_score(stringInput, k=1)

    if len(search_results) == 0:
        print("No result found!")
        continue

    doc, score = search_results[0]

    # # only for debuggin purpose
    # print(f"[debug score: {score:.4f}]")

    if score > THRESHOLD:
        print("Not relevant! Try asking something related to the Solar System.")
    else:
        print(f"\n Result (score: {score:.4f}):")
        print(doc.page_content)