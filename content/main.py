import os
import json
import ast
import torch
import subprocess
from urllib.parse import urlparse
from git import Repo
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import time
import git
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from fastapi import FastAPI
import logging
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from fastapi import FastAPI, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from git import Repo, exc as git_exc
import logging

load_dotenv()
app = FastAPI()

groq_api_key = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RepoURL(BaseModel):
    repo_url: str

class ChatRequest(BaseModel):
    message: str

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def clear_neo4j_database(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
#clear_neo4j_database(driver) 

def clone_repo(repo_url, target_dir_prefix="cloned_repo"):
    timestamp = time.strftime("%Y%m%d-%H%M%S") 
    target_dir = f"{target_dir_prefix}_{timestamp}"
    print(f"Cloning repository from {repo_url} into {target_dir}...")
    Repo.clone_from(repo_url, target_dir)
    
    return target_dir

def process_directory(directory, driver, metadata_file='metadata.json', file_storage='storage.json'):
    def create_directory_node(tx, name, path):
        query = """
        CREATE (d:Directory {name: $name, path: $path})
        RETURN d
        """
        tx.run(query, name=name, path=path)

    def create_file_node(tx, name, path, size):
        query = """
        CREATE (f:File {name: $name, path: $path, size: $size})
        RETURN f
        """
        tx.run(query, name=name, path=path, size=size)

    def create_function_node(tx, name, args, docstring, file_path, body):
        if docstring is None:
            docstring = "No docstring provided."  
        query = """
        CREATE (func:Function {name: $name, args: $args, docstring: $docstring, file_path: $file_path, body: $body})
        RETURN func
        """
        tx.run(query, name=name, args=args, docstring=docstring, file_path=file_path, body=body)

    def create_relationship(tx, parent_path, child_path, relationship_type):
        query = """
        MATCH (p {path: $parent_path})
        MATCH (c {path: $child_path})
        MERGE (p)-[r:""" + relationship_type + """]->(c)
        """
        tx.run(query, parent_path=parent_path, child_path=child_path)

    def extract_functions(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=file_path)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name
                args = [arg.arg for arg in node.args.args]
                docstring = ast.get_docstring(node)
                body = ast.get_source_segment(content, node)  # Extract function body
                functions.append({"name": name, "args": args, "docstring": docstring, "body": body})
        return functions

    metadata = {}
    file_contents = {}

    with driver.session() as session:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '.config', '.gitattributes', '.gitignore']]
            files = [f for f in files if f not in ['.git', '.config', '.gitattributes', '.gitignore']]
            root_path = os.path.abspath(root)
            metadata[root_path] = {"directories": dirs, "files": files}
            session.execute_write(create_directory_node, os.path.basename(root), root_path)
            
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_contents[file_path] = f.read()
                
                session.execute_write(create_file_node, file, file_path, file_size)
                session.execute_write(create_relationship, root_path, file_path, "CONTAINS")
                
                if file.endswith(".py") or file.endswith(".js"):
                    functions = extract_functions(file_path)
                    for func in functions:
                        func_path = f"{file_path}::{func['name']}"
                        session.execute_write(
                            create_function_node,
                            func["name"],
                            ",".join(func["args"]),
                            func["docstring"] or "No docstring",
                            file_path,
                            func["body"] or "No body"
                        )
                        session.execute_write(create_relationship, file_path, func_path, "DEFINES")

    with open(metadata_file, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=4)
    
    with open(file_storage, "w", encoding="utf-8") as file_store:
        json.dump(file_contents, file_store, indent=4)
    
    print("Metadata and file storage saved.")

llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5,
)

class FunctionOutputModel(BaseModel):
    function_name: str
    description: str
    parameters: List[str]

parser = JsonOutputParser(pydantic_object=FunctionOutputModel)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the given Python function and generate a structured JSON description:
        {{
            "function_name": "Extracted function name",
            "description": "Brief but precise explanation of what the function does.",
            "parameters": ["param1: explanation", "param2: explanation"]
        }}"""),
    ("user", "{input}")
])

chain = prompt | llm | parser


def fetch_functions(driver):
    with driver.session() as session:
        query = "MATCH (f:Function) RETURN f.name, f.args, f.docstring, f.file_path, f.body"
        results = session.run(query)
        
        functions = []
        for record in results:
            func = {
                "name": record["f.name"],
                "args": record["f.args"],
                "docstring": record["f.docstring"],
                "file_path": record["f.file_path"],
                "body": record["f.body"]
            }
            functions.append(func)
    print("Successfully fetched functions")
    return functions  

def generate_descriptions(function_list):
    descriptions = {}

    for func in function_list:
        code = func['body']  # Function code is stored in 'body' field
        result = chain.invoke({"input": code})
        
        #print("Raw output:", result)  # Debugging, you can remove it later
        
        try:
            descriptions[func['name']] = {
                "function_name": result["function_name"],
                "description": result["description"],
                "parameters": result["parameters"]
            }
        except KeyError as e:
            print(f"Error extracting description for {func['name']}: {e}")
            descriptions[func['name']] = "Description not available."
        
        # Adding delay of 1 second between requests
        time.sleep(1)
    print("Successfully generated descriptions")
    return descriptions

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(descriptions):
    embeddings = {}

    for name, details in descriptions.items():
        if isinstance(details, dict) and "description" in details:
            desc = details["description"]
            tokens = tokenizer(desc, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embedding_vector = model(**tokens).last_hidden_state.mean(dim=1).squeeze().tolist()
            embeddings[name] = embedding_vector
    print("Successfully generated embeddings")
    return embeddings

def store_in_neo4j(function_list, descriptions, embeddings, driver):
    with driver.session() as session:
        for func in function_list:
            func_name = func['name']
            description = descriptions.get(func_name, {}).get("description", "Description not available")
            embedding_vector = embeddings.get(func_name, [])

            # Ensure the function node exists and update its description and embedding
            session.run(
                """MERGE (f:Function {name: $name})
                   SET f.description = $desc, 
                       f.parameters = $params, 
                       f.embedding = $embedding""",
                name=func_name,
                desc=description,
                params=", ".join(descriptions.get(func_name, {}).get("parameters", [])),
                embedding=embedding_vector
            )
    print(f"Stored {len(function_list)} functions with descriptions and embeddings in Neo4j.")

@app.post("/clone_repo")
async def clone_repo_endpoint(repo: RepoURL):
    """Endpoint to handle repository cloning and processing"""
    cloned_dir = clone_repo(repo.repo_url)
    process_directory(cloned_dir, driver)
    function_list = fetch_functions(driver)
    descriptions = generate_descriptions(function_list)
    embeddings = generate_embeddings(descriptions)
    store_in_neo4j(function_list,descriptions,embeddings,driver)
    return {"message": "Repository cloned and processed successfully", "directory": cloned_dir}


def fetch_functionsx(driver):
    """Fetch stored functions from Neo4j database."""
    with driver.session() as session:
        query = """
        MATCH (f:Function) 
        RETURN f.name, f.args, f.docstring, f.file_path, f.body, f.description, f.embedding
        """
        results = session.run(query)
        return [
            {
                "name": record["f.name"],
                "args": record["f.args"],
                "docstring": record["f.docstring"],
                "file_path": record["f.file_path"],
                "body": record["f.body"],
                "description": record["f.description"],
                "embedding": np.array(record["f.embedding"]) if record["f.embedding"] else None
            }
            for record in results
        ]

def find_best_match_with_embedding(functions, query):
    """Find the most relevant function based on cosine similarity."""
    if not functions:
        return None, 0.0
    
    tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        query_embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
    
    best_match, max_similarity = None, -1
    for func in functions:
        if func["embedding"] is not None:
            similarity = cosine_similarity([query_embedding], [func["embedding"]])[0][0]
            if similarity > max_similarity:
                max_similarity, best_match = similarity, func
    
    return best_match, max_similarity

def similarity(user_query, llm):
       """Retrieve the most relevant function based on user query and explain it."""
       functions = fetch_functionsx(driver)
       best_match, similarity = find_best_match_with_embedding(functions, user_query)

       if best_match is None:
           return {"error": "No matching function found."} 

       prompt = PromptTemplate(
           input_variables=["user_query", "function_body"],
           template="User query: {user_query}\n\nFunction Code:\n{function_body}\n\nExplain this function based on the user's query."
       )
       
       chain = prompt | llm
       output = chain.invoke({"user_query": user_query, "function_body": best_match["body"]})
       return output

def process_query(user_query, llm, memory):
    """Process user query and decide execution flow."""
    if any(keyword in user_query.lower() for keyword in ["explain", "how does", "what does", "tell me about", "describe"]):
        response = similarity(user_query, llm)
    else:
        chain = ConversationChain(llm=llm, memory=memory)
        response = chain.invoke(user_query)
    return response


@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest = Body(...)):
    memory = ConversationBufferMemory()
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = process_query(chat_request.message, llm, memory)
    #print("Assistant:", response)
    return {"Assistant" : response}
