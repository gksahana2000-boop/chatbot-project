
from flask import Flask, request, jsonify
from flask import Flask, render_template, request, jsonify
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Flask app
app = Flask(__name__)

# === Load and prepare data (only once) ===
file_path = "datasheet.csv"  # Make sure this file exists
loader = CSVLoader(file_path=file_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Prompt template
system_prompt = (
    "You are a question-answering assistant. Use the retrieved context to answer the user's question. "
    "If unsure, say you don't know. Keep your response concise, using up to three sentences.\n\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# === Flask route ===


@app.route("/")
def index():
    return render_template("index.html")
@app.route("/ask", methods=["POST"])
def ask():
    
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"error": "Question is required"}), 400

    # Retrieve relevant documents
    relevant_docs = retriever.invoke(user_input)
    context = "\n".join(doc.page_content for doc in relevant_docs)

    # Format input for the prompt
    final_input = {
        "context": context,
        "input": user_input
    }

    prompt_messages = prompt_template.invoke(final_input)
    response = llm.invoke(prompt_messages)

    return jsonify({"answer": response.content})

# === Run server ===
if __name__ == "__main__":
    app.run(debug=True)
