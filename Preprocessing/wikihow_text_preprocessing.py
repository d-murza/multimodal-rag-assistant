import os
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import time
import torch
import psutil


# TASK 1 - WikiHow Text Preprocessing
start_time = time.time() # the whole thing took ca 15 min

CHUNK_SIZE = 8000
embedding_dim = 384

DATA_DIR = "data/wikihow-cleaned"
OUTPUT_DIR = "processed-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# initialise faiss index
title_index = faiss.IndexFlatIP(embedding_dim)
combined_index = faiss.IndexFlatIP(embedding_dim)

# initialise model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {device}")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

print(f"Available RAM memory: {psutil.virtual_memory().available / 1e9:.1f} GB") # had issues with ram memory for this task


connect = sqlite3.connect(os.path.join(OUTPUT_DIR, "wikihow.db"))
cursor = connect.cursor()

# do all preprocessing in chunks to minimise ram memory needed
first_chunk = True
for df_chunk in pd.read_csv(os.path.join(DATA_DIR, "wikihow-cleaned.csv"), chunksize=CHUNK_SIZE):
    if first_chunk:
        df_chunk.to_sql("articles", connect, if_exists="replace", index=True, index_label="id")
        first_chunk = False
    else:
        df_chunk.to_sql("articles", connect, if_exists="append", index=True, index_label="id")
    
    # compute text embeddings over title
    title_embeddings = model.encode(df_chunk["title"].tolist(),batch_size=64,show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(title_embeddings)
    title_index.add(title_embeddings)  

    # compute text embeddings over title and summary concatenated
    combined = (df_chunk["title"] + ". " + df_chunk["summary"]).tolist() 
    combined_embeddings = model.encode(combined,batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(combined_embeddings)
    combined_index.add(combined_embeddings)
    
    print(f"Processed {len(df_chunk)} rows. Current index size: {title_index.ntotal}")
    print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

    connect.commit()



# verifying everything worked correctly
cursor.execute("SELECT COUNT(*) FROM articles")
print("Number of articles in database:", cursor.fetchone()[0])
print("Size of title FAISS index:", title_index.ntotal)
print("Size of combined FAISS index:", combined_index.ntotal)

# saving title index starting now
faiss.write_index(title_index, os.path.join(OUTPUT_DIR, "title_index.faiss"))
faiss.write_index(combined_index, os.path.join(OUTPUT_DIR, "combined_index.faiss"))



# do the query-based semantic retrieval test 
def retrieval_test(query, index, db_connection, k=5):
    print(f"\nResults for: {query}\n")

    # embedds query, normalise and search for k most similar
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    scores, ids = index.search(query_embedding, k)
    
    # get the results
    results = zip(ids[0], scores[0])
    for i, result in enumerate(results, start=1):
        article_id = int(result[0])
        score = result[1]

        # choose only the matching row from database
        query_db = "SELECT title, summary FROM articles WHERE id = ?"
        database = pd.read_sql_query(query_db, db_connection, params=(article_id,))
        
        title = database.iloc[0]['title']
        summary = database.iloc[0]['summary']
        
        # printing results
        print(f"{i}. Score: {score}")
        print(f"Title: {title}")
        print(f"Summary: {summary[:200]}")
        print()



# specify a query
query = "how to clean a saddle?"
print("Using the combined index:")
retrieval_test(query, combined_index, connect)

print("Using title index:")
retrieval_test(query, title_index, connect)

end_time = time.time()
execution_time = end_time - start_time
print(f"Time: {execution_time:.2f} seconds")



