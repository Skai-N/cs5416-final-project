# Save this file and run on each host with NODE_NUMBER env var (0,1,2) and appropriate NODE_?_IP values.
import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import queue
from queue import Queue
import threading
import traceback
import requests
from flask_compress import Compress
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 3))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8001')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8002')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

# for pipeline testing:
BATCH_TIMEOUT = int(os.environ.get('BATCH_TIMEOUT', 25))
BATCH_MAX_SIZE = int(os.environ.get('BATCH_MAX_SIZE', 32))

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768,
    'max_tokens': 128,
    'retrieval_k': 10,
    'truncate_length': 512,
}

# Flask app
app = Flask(__name__)
Compress(app)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()

@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float

@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float

class MonolithicPipeline:
    def __init__(self):
        # Use CPU by default; you can change to cuda if available and models/support that.
        self.device = torch.device('cpu')
        print(f"[node {NODE_NUMBER}] Initializing pipeline on {self.device}")
        print(f"[node {NODE_NUMBER}] FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"[node {NODE_NUMBER}] Documents path: {CONFIG['documents_path']}")

        self.global_batch_size = BATCH_MAX_SIZE
        # Round-robin state for generation nodes (only used on node 0)
        self._gen_round_robin_counter = 0

        # Node responsibilities:
        if NODE_NUMBER == 0:
            # Node 0: orchestration + embeddings + FAISS + postprocessing (sentiment/safety)
            print("[Node 0] Loading embedding model, FAISS, DB, sentiment and safety models...")

            # Embedding model
            self.embedding_model_name = "BAAI/bge-base-en-v1.5"
            self.embedding_model = SentenceTransformer(self.embedding_model_name).to(self.device)
            print("[Node 0]   Embedding model loaded.")

            # FAISS index
            self.index = faiss.read_index(CONFIG["faiss_index_path"])
            print("[Node 0]   FAISS index loaded.")

            # Database
            self.db_path = f"{CONFIG['documents_path']}/documents.db"
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            print("[Node 0]   Documents DB connected.")

            # Sentiment & safety
            self.sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.safety_model_name = "unitary/toxic-bert"

            print("[Node 0]   Loading sentiment classifier...")
            self.sentiment_classifier = hf_pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                device=-1  # CPU pipeline uses -1
            )
            print("[Node 0]   Loading safety classifier...")
            self.safety_classifier = hf_pipeline(
                "text-classification",
                model=self.safety_model_name,
                device=-1
            )
            print("[Node 0] All Node-0 models loaded!")

        elif NODE_NUMBER in (1, 2):
            # Node 1 & Node 2: rerank + LLM generation
            print(f"[Node {NODE_NUMBER}] Loading reranker and LLM models...")
            self.reranker_model_name = "BAAI/bge-reranker-base"
            self.llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

            # Reranker
            print(f"[Node {NODE_NUMBER}]   Loading reranker...")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name
            ).to(self.device)
            self.reranker_model.eval()

            # LLM
            print(f"[Node {NODE_NUMBER}]   Loading LLM...")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name, torch_dtype=torch.float16
            ).to(self.device)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm_model.eval()

            # DB connection (fetching documents)
            self.db_path = f"{CONFIG['documents_path']}/documents.db"
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)

            print(f"[Node {NODE_NUMBER}] All models loaded!")

        else:
            raise RuntimeError("NODE_NUMBER must be 0, 1, or 2.")

    # Embedding + FAISS (Node 0)
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings

    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        query_embeddings = query_embeddings.astype('float32')
        _, indices = self.index.search(query_embeddings, CONFIG['retrieval_k'])
        return [row.tolist() for row in indices]

    # Fetch documents from DB (used by Node 1/2)
    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        cursor = self.db_conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            documents_batch.append(documents)
        return documents_batch

    # Rerank (Node 1/2)
    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores.tolist()))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        return reranked_batches

    # LLM generation (Node 1/2)
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        all_messages = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            all_messages.append(messages)

        texts = [
            self.llm_tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in all_messages
        ]

        model_inputs = self.llm_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                use_cache=True,
                do_sample=False,
            )

        # Extract only new tokens for each output
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

    # Sentiment & safety analysis (Node 0)
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.sentiment_classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        return sentiments

    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.safety_classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result['score'] > 0.5)
        return toxicity_flags

    # Helper: choose generation node URL (only used on Node 0)
    def _choose_generation_node(self) -> str:
        # Round-robin between NODE_1_IP and NODE_2_IP
        choice = self._gen_round_robin_counter % 2
        self._gen_round_robin_counter += 1
        if choice == 0:
            return f"http://{NODE_1_IP}/process_stage"
        else:
            return f"http://{NODE_2_IP}/process_stage"

# Global pipeline instance
pipeline = None

def send_to_generation_node(node_url: str, sub_queries: List[str], sub_doc_ids: List[List[int]], 
                             start_idx: int, end_idx: int) -> Tuple[int, int, List[str]]:
    """
    Helper function to send a sub-batch to a generation node.
    Returns (start_idx, end_idx, responses) for result placement.
    """
    try:
        r = requests.post(node_url, json={
            "stage": "fetch_rerank_llm",
            "payload": {
                "queries": sub_queries,
                "doc_ids": sub_doc_ids
            }
        }, timeout=300)
        
        r.raise_for_status()
        node_data = r.json()
        return (start_idx, end_idx, node_data["responses"])
    
    except Exception as ex:
        print(f"[node {NODE_NUMBER}] Error contacting {node_url}: {ex}")
        traceback.print_exc()
        # Return error responses for this sub-batch
        error_responses = [f"Error generating response: {ex}"] * len(sub_queries)
        return (start_idx, end_idx, error_responses)

def process_requests_worker():
    """Worker thread that processes requests from the queue on Node 0 (orchestration)."""
    global pipeline
    while True:
        try:
            request_data = request_queue.get()
            if request_data is None:
                break
            batch = [request_data]

            batch_wait_start = time.time()
            while len(batch) < pipeline.global_batch_size:
                remaining = BATCH_TIMEOUT - (time.time() - batch_wait_start)
                if remaining < 0:
                    break
                try:
                    req_data = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if req_data is None:
                    break
                batch.append(req_data)
            print(f"[node {NODE_NUMBER}] Processing batch of size ({len(batch)})")

            batch_start = time.time()
            reqs = [
                PipelineRequest(request_id=r["request_id"], query=r["query"], timestamp=batch_start)
                for r in batch
            ]

            start = time.time()

            # Node 0 orchestration flow:
            # 1) Embed & FAISS locally
            queries = [r.query for r in reqs]
            embeddings = pipeline._generate_embeddings_batch(queries)
            doc_ids = pipeline._faiss_search_batch(embeddings)

            # 2) Split batch between generation nodes for PARALLEL processing
            gen_nodes = [
                f"http://{NODE_1_IP}/process_stage",
                f"http://{NODE_2_IP}/process_stage"
            ]

            num_nodes = len(gen_nodes)
            batch_size = len(queries)

            # Compute partition indices
            # Example: for batch 7 → split like [3,4]
            splits = []
            base = batch_size // num_nodes
            remainder = batch_size % num_nodes
            start_split = 0
            for i in range(num_nodes):
                end_split = start_split + base + (1 if i < remainder else 0)
                splits.append((start_split, end_split))
                start_split = end_split

            responses_partial = [None] * batch_size

            # Dispatch sub-batches to generation nodes IN PARALLEL using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_nodes) as executor:
                futures = []
                
                for node_idx, (s, e) in enumerate(splits):
                    if s == e:
                        continue
                    
                    sub_queries = queries[s:e]
                    sub_doc_ids = doc_ids[s:e]
                    node_url = gen_nodes[node_idx]
                    
                    # Submit task to thread pool
                    future = executor.submit(
                        send_to_generation_node,
                        node_url,
                        sub_queries,
                        sub_doc_ids,
                        s,
                        e
                    )
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        start_idx, end_idx, responses = future.result()
                        # Place responses in correct position
                        for i, resp in enumerate(responses):
                            responses_partial[start_idx + i] = resp
                    except Exception as ex:
                        print(f"[node {NODE_NUMBER}] Error collecting future result: {ex}")
                        traceback.print_exc()

            generations = responses_partial

            # 3) Postprocess locally (sentiment + safety)
            sentiments = pipeline._analyze_sentiment_batch(generations)
            toxicity_flags = pipeline._filter_response_safety_batch(generations)

            processing_time = time.time() - start
            with results_lock:
                for req, gen, snt, tox in zip(reqs, generations, sentiments, toxicity_flags):
                    response_payload = {
                        "request_id": req.request_id,
                        "generated_response": gen,
                        "sentiment": snt,
                        "is_toxic": "true" if tox else "false",
                        "processing_time": processing_time,
                    }
                    results[req.request_id] = response_payload

            for _ in batch:
                request_queue.task_done()

        except Exception as e:
            print(f"[node {NODE_NUMBER}] Error in worker: {e}")
            traceback.print_exc()
            try:
                for _ in batch:
                    request_queue.task_done()
            except Exception:
                pass

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests (Node 0 only)"""
    try:
        if NODE_NUMBER != 0:
            return jsonify({'error': 'This node does not accept client queries! Send to Node 0.'}), 400

        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')

        if not request_id or not query:
            return jsonify({'error': 'Missing request_id or query'}), 400

        # Check if result already exists
        with results_lock:
            if request_id in results:
                return jsonify(results.pop(request_id)), 200

        print(f"[node 0] queueing request {request_id}")
        request_queue.put({
            'request_id': request_id,
            'query': query
        })

        # Wait for processing result
        timeout = 300  # 5 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    return jsonify(result), 200

            if time.time() - start_wait > timeout:
                return jsonify({'error': 'Request timeout'}), 504
            time.sleep(0.1)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_stage', methods=['POST'])
def process_stage():
    """
    Process pipeline stages for inter-node communication.

    Generation nodes (Node 1 & Node 2) respond to:
    - "fetch_rerank_llm" -> returns generated text
    """
    try:
        data = request.json
        stage = data.get("stage")
        payload = data.get("payload")

        # Node 1 & Node 2: fetch + rerank + LLM
        if stage == "fetch_rerank_llm":
            if NODE_NUMBER not in (1, 2):
                return jsonify({"error": "fetch_rerank_llm stage only on node 1 or 2"}), 400

            queries = payload.get("queries", [])
            doc_ids = payload.get("doc_ids", [])

            documents = pipeline._fetch_documents_batch(doc_ids)
            reranked = pipeline._rerank_documents_batch(queries, documents)
            responses = pipeline._generate_responses_batch(queries, reranked)

            return jsonify({
                "responses": responses
            })

        return jsonify({"error": "unknown stage"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200

def main():
    global pipeline
    print("="*60)
    print("DISTRIBUTED CUSTOMER SUPPORT PIPELINE (PARALLEL LLM)")
    print("="*60)
    print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")
    print("")
    if NODE_NUMBER == 0:
        print("  Node 0: Orchestration + embeddings + FAISS + postprocessing (sentiment, safety)")
        print("  >>> PARALLEL dispatch to Node 1 & Node 2 for LLM generation")
    else:
        print(f"  Node {NODE_NUMBER}: fetch + rerank + LLM generation")

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = MonolithicPipeline()
    print("Pipeline initialized!")

    # Apply method restrictions to ensure nodes only expose intended behavior
    if NODE_NUMBER == 0:
        def not_allowed(*args, **kwargs):
            raise RuntimeError("This node is not responsible for this stage (NODE 0).")
        # Node 0 should not run rerank/LLM generation
        pipeline._rerank_documents_batch = not_allowed
        pipeline._generate_responses_batch = not_allowed

    elif NODE_NUMBER in (1, 2):
        def not_allowed(*args, **kwargs):
            raise RuntimeError(f"This node is not responsible for this stage (NODE {NODE_NUMBER}).")
        # Node 1/2 should not run embedding/FAISS or sentiment/safety
        pipeline._generate_embeddings_batch = not_allowed
        pipeline._faiss_search_batch = not_allowed
        pipeline._analyze_sentiment_batch = not_allowed
        pipeline._filter_response_safety_batch = not_allowed

    print("Node restrictions applied successfully!")

    # Start worker thread on Node 0 to handle client queue
    if NODE_NUMBER == 0:
        worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
        worker_thread.start()
        print("Worker thread started on Node 0")

    # Start Flask server on this node
    node_ip = os.environ.get(f"NODE_{NODE_NUMBER}_IP", None)
    if node_ip:
        hostname = node_ip.split(':')[0]
        port = int(node_ip.split(':')[1]) if ':' in node_ip else 8000 + NODE_NUMBER
        print(f"Starting Flask on {hostname}:{port} (node {NODE_NUMBER})")
        app.run(host=hostname, port=port, threaded=True)
    else:
        raise RuntimeError("Node IP environment variable not set.")

if __name__ == "__main__":
    main()