import sqlite3
import os 
import numpy as np
import faiss

def _initialize_documents():
        """Create dummy documents database if it doesn't exist"""

        DOCUMENTS_DIR = "documents/"
        NUM_DOCUMENTS = 4500000
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

        db_path = os.path.join(DOCUMENTS_DIR, "documents.db")

        if not os.path.exists(db_path):
            print("Creating document database...")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table with index on doc_id for fast lookups
            cursor.execute('''
                CREATE TABLE documents (
                    doc_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL
                )
            ''')
            
            # Insert documents in batches for better performance
            batch_size = 10000
            documents = []
            
            for i in range(NUM_DOCUMENTS):
                documents.append((
                    i,
                    f'Document {i}',
                    f'This is the content of document {i}. It contains information about customer support issue {i % 100}.',
                    ['technical', 'billing', 'shipping', 'general'][i % 4]
                ))
                
                # Insert in batches
                if len(documents) >= batch_size:
                    cursor.executemany(
                        'INSERT INTO documents (doc_id, title, content, category) VALUES (?, ?, ?, ?)',
                        documents
                    )
                    conn.commit()
                    documents = []
                    print(f"Created {i + 1}/{NUM_DOCUMENTS} documents...")
            
            # Insert remaining documents
            if documents:
                cursor.executemany(
                    'INSERT INTO documents (doc_id, title, content, category) VALUES (?, ?, ?, ?)',
                    documents
                )
                conn.commit()
            
            conn.close()
            print(f"Document database created at {db_path}")
            print(f"Database size: {os.path.getsize(db_path) / 1e6:.2f} MB")

def _create_faiss_index():
        """Create a large FAISS index"""
        dim = 768
        num_docs = 4500000
        index_path = "faiss_index.bin"

        nlist=4096
        # Create index
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.train(np.random.randn(10000, dim).astype('float32'))
        # Add vectors in batches to manage memory
        batch_size = 10000
        for i in range(0, num_docs, batch_size):
            # Generate random embeddings (in real scenario, these would be document embeddings)
            batch_embeddings = np.random.randn(min(batch_size, num_docs - i), dim).astype('float32')
            index.add(batch_embeddings)
            
            if i % 100000 == 0:
                print(f"Added {i}/{num_docs} vectors to index...")
        
        # Save index
        index.nprobe = 64
        faiss.write_index(index, index_path)
        print(f"FAISS index created and saved")

if __name__ == "__main__":
    _initialize_documents()
    _create_faiss_index()