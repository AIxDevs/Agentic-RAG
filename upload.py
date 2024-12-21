import os
import json
import random
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.tools import DuckDuckGoSearchRun
import dotenv
dotenv.load_dotenv()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )

url_to_name_map = {
    "https://lilianweng.github.io/posts/2023-06-23-agent/": "Agent_Post",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/": "Prompt_Engineering_Post",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/": "Adv_Attack_LLM_Post",
}

docs = {}
for url in url_to_name_map:
    try:
        loader = WebBaseLoader(url)
        docs[url] = loader.load()
    except Exception as e:
        print(f"Error loading {url}: {e}")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)

doc_splits = {}
for url, doc in docs.items():
    doc_splits[url] = text_splitter.split_documents(doc)

client = chromadb.PersistentClient(path="db")

def create_collection(collection_name):
    return client.get_or_create_collection(name=collection_name)


def upload_data_to_collection(collection_name, data):
    try:
        collection = create_collection(collection_name)
        id = random.randint(10000, 99999)
        embedding = openai_ef([data])[0]  
        
        collection.add(
            documents=[data],
            embeddings=[embedding],
            ids=[str(id)]
        )
        print(f"Data uploaded to collection {collection_name} with ID: {id}")
        return json.dumps({"Message": "Data uploaded successfully"})
    except Exception as e:
        print(f"Error uploading data to {collection_name}: {e}")
        return json.dumps({"Error": str(e)})
    
# for url, splits in doc_splits.items():
#     collection_name = url_to_name_map[url]  
#     for chunk in splits:
#         upload_data_to_collection(collection_name, chunk.page_content)

def search_db(collection_name: str, input_query: str, n=5):
    try:
        if not isinstance(n, int) or n <= 0:
            n = 5

        collection = create_collection(collection_name)
        print("\n\nSearching {}\n\n".format(collection))
        
        embedding = openai_ef([input_query])[0]  
        
        res = collection.query(
            query_embeddings=[embedding],
            n_results=n
        )
        
        
        result = [doc for doc in res['documents'][0]]
        
        return json.dumps({"Data": result})
    
    except Exception as e:
        print(f"Error searching in collection {collection_name}: {e}")
        return json.dumps({"Error": str(e)})
    
# search_results = search_db("Agent_Post", "what are ai agents?")
# print(search_results)


DuckDuckGo = DuckDuckGoSearchRun()
def Internet_search(input:str):
    print("\n\nSearching Internet \n\n")
    
    res = DuckDuckGo.run(input)
    return res

# print(Internet_search("who won jake paul vs mike tyson?"))