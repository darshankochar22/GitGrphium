# GitHub Agent Backend (FastAPI)

This is a **FastAPI-based backend** for an AI-powered GitHub agent that:


 -**Clones repositories** from GitHub.  
 -**Parses folders & files** to extract `.py` and `.js` functions.  
 -**Generates embeddings** for extracted functions.  
 -**Computes similarity scores** for user queries.  
 -**Uses an LLM** to explain the most relevant function.  
 -**Integrates with a chatbot** to return responses.  

---

## 🚀 Features  

1. **Clone GitHub Repositories**  
   - Takes a **GitHub repo link** as input.  
   - Clones it to a temporary storage location.  

2. **Parse Directory Structure**  
   - Extracts **folders & files** while ignoring `.git`, `.config`, etc.  
   - Stores metadata (name, path, size, type) in JSON format.  

3. **Function Extraction**  
   - Parses **Python (`.py`)** and **JavaScript (`.js`)** files.  
   - Extracts **function names, parameters, and docstrings**.  
   - Stores extracted functions in a structured format.  

4. **Embedding & Similarity Search**  
   - Converts extracted function descriptions into **embeddings**.  
   - Computes **similarity scores** with respect to the user's query.  
   - Retrieves the **most relevant function** based on the highest score.  

5. **LLM-Based Explanation**  
   - Sends the retrieved function to an **LLM (e.g., GPT, Claude, Groq)**.  
   - Requests a **detailed explanation** of the function’s purpose.  
   - Returns the explanation to the chatbot for the user.  

---

## 📌 Tech Stack  

- **FastAPI** - Backend API framework  
- **LangChain** - LLM integration  
- **Neo4j** - Graph database for function storage  
- **MiniLM-L6-v2** - Embedding model for similarity search  
- **GitPython** - Cloning GitHub repositories  
- **Python’s AST** - Function parsing  
- **Pydantic** - Data validation  

---

## 🔧 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/github-agent-backend.git
cd content
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables  
Create a `.env` file with:  
```
GITHUB_ACCESS_TOKEN=your_personal_access_token
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_llm_api_key
```

### 4️⃣ Run FastAPI Server  
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🛠 API Endpoints  

### 🔹 **Clone a Repository**  
`POST /clone-repo/`  
```json
{
  "repo_url": "https://github.com/example/repo.git"
}
```
🔹 **Response:** ✅ Repo cloned successfully  

---

### 🔹 **Parse Repository**  
`GET /parse-repo/`  
🔹 **Response:** Returns directory structure and extracted functions.  

---

### 🔹 **Find Similar Functions**  
`POST /search-function/`  
```json
{
  "query": "How does the authentication system work?"
}
```
🔹 **Response:** Retrieves and ranks the most relevant function.  

---

### 🔹 **Get LLM Explanation**  
`GET /explain-function/{function_id}`  
🔹 **Response:** LLM-generated explanation of the function.  

---

## 🏗️ Future Enhancements  
- Support for more languages (`.ts`, `.java`, etc.).  
- Advanced **prompt engineering** for better LLM responses.  
- Integration with **vector databases** for fast retrieval.  

---

## 📜 License  
This project is licensed under the MIT License.  

---

## 📬 Contact  
For questions or issues, open a GitHub issue or reach out via email.  

