import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue
import threading
import traceback
import requests
from memory_profiler import profile

warnings.filterwarnings("ignore")

# Read environment variables
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "localhost:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8002")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents/")

# Configuration
CONFIG = {
    "faiss_index_path": FAISS_INDEX_PATH,
    "documents_path": DOCUMENTS_DIR,
    "faiss_dim": 768,
    "max_tokens": 128,
    "retrieval_k": 10,
    "truncate_length": 512,
}

# Flask app
app = Flask(__name__)

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
        self.device = torch.device("cpu")
        print(f"[node {NODE_NUMBER}] Initializing pipeline on {self.device}")
        print(f"[node {NODE_NUMBER}] FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"[node {NODE_NUMBER}] Documents path: {CONFIG['documents_path']}")

        self.global_batch_size = 1  # batch size for each stage of pipeline

        # Model names
        if NODE_NUMBER == 0:
            self.embedding_model_name = "BAAI/bge-base-en-v1.5"
            self.embedding_model = SentenceTransformer(self.embedding_model_name).to(
                self.device
            )
        if NODE_NUMBER == 1:
            self.index = faiss.read_index(CONFIG["faiss_index_path"])
            self.db_path = f"{CONFIG['documents_path']}/documents.db"
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)

        if NODE_NUMBER == 2:
            self.reranker_model_name = "BAAI/bge-reranker-base"
            self.llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            self.sentiment_model_name = (
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )
            self.safety_model_name = "unitary/toxic-bert"

            # initialize reranker model/tokenizer
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_name
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name
            ).to(self.device)
            self.reranker_model.eval()

            # initialize llm model/tokenizer
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name, torch_dtype=torch.float16
            ).to(self.device)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm_model.eval()

            # initialize sentiement analysis model
            self.sentiment_classifier = hf_pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                device=self.device,
            )

            # initialize sensitivity model
            self.safety_classifier = hf_pipeline(
                "text-classification", model=self.safety_model_name, device=self.device
            )

    @profile
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 2: Generate embeddings for a batch of queries"""
        embeddings = self.embedding_model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        # del model
        # gc.collect()
        return embeddings

    @profile
    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 3: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG["faiss_index_path"]):
            raise FileNotFoundError(
                "FAISS index not found. Please create the index before running the pipeline."
            )

        print("Loading FAISS index")
        query_embeddings = query_embeddings.astype("float32")
        _, indices = self.index.search(query_embeddings, CONFIG["retrieval_k"])
        # del index
        # gc.collect()
        return [row.tolist() for row in indices]

    @profile
    def _fetch_documents_batch(
        self, doc_id_batches: List[List[int]]
    ) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        cursor = self.db_conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    "SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?",
                    (doc_id,),
                )
                result = cursor.fetchone()
                if result:
                    documents.append(
                        {
                            "doc_id": result[0],
                            "title": result[1],
                            "content": result[2],
                            "category": result[3],
                        }
                    )
            documents_batch.append(documents)
        # conn.close()
        return documents_batch

    @profile
    def _rerank_documents_batch(
        self, queries: List[str], documents_batch: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Step 5: Rerank retrieved documents for each query in the batch"""
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc["content"]] for doc in documents]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=CONFIG["truncate_length"],
                ).to(self.device)
                scores = (
                    self.reranker_model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        # del model, tokenizer
        # gc.collect()
        return reranked_batches

    @profile
    def _generate_responses_batch(
        self, queries: List[str], documents_batch: List[List[Dict]]
    ) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""

        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join(
                [f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]]
            )
            messages = [
                {
                    "role": "system",
                    "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ]

            text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(
                self.llm_model.device
            )
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=CONFIG["max_tokens"],
                temperature=0.01,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.llm_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            responses.append(response)
        # del model, tokenizer
        # gc.collect()
        return responses

    @profile
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""

        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = self.sentiment_classifier(truncated_texts)
        sentiment_map = {
            "1 star": "very negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "very positive",
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result["label"], "neutral"))
        # del classifier
        # gc.collect()
        return sentiments

    @profile
    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""

        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = self.safety_classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result["score"] > 0.5)
        # del classifier
        # gc.collect()
        return toxicity_flags


# Global pipeline instance
pipeline = None


def process_requests_worker():
    """Worker thread that processes requests from the queue.
    If we're on Node 0 it will distribute across the other 2 nodes
    """
    global pipeline
    while True:
        try:
            request_data = request_queue.get()
            if request_data is None:  # Shutdown signal
                break
            batch = [request_data]

            batch_wait_start = time.time()
            while len(batch) < pipeline.global_batch_size:
                req_data = request_queue.get()
                if req_data is not None:
                    batch.append(req_data)
                else:
                    break
                if (
                    time.time() - batch_wait_start > 5
                ):  # automatically send a batch after 5 seconds
                    break

            print(f"Processing batch of size ({len(batch)})")

            batch_start = time.time()
            # Create request object
            reqs = [
                PipelineRequest(
                    request_id=r["request_id"], query=r["query"], timestamp=batch_start
                )
                for r in batch
            ]

            start = time.time()

            # node 0 will embed, then put embed in node 1 for fetch, rerank/llm/post in node 2
            if NODE_NUMBER == 0:
                # 1- local embedding
                query_batch = [r.query for r in reqs]
                emb = pipeline._generate_embeddings_batch(query_batch)  # np array?
                emb_list = emb.astype("float32").tolist()

                # 2 - call Node 1 for FAISS+fetch
                node1_url = f"http://{NODE_1_IP}/process_stage"
                try:
                    r = requests.post(
                        node1_url,
                        json={"stage": "faiss_and_fetch", "payload": emb_list},
                        timeout=120,
                    )
                    r.raise_for_status()
                    node1_data = r.json()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error calling Node1 FAISS: {e}")

                documents = node1_data.get("documents", [[]])

                # 3 - call Node 2 for rerank/llm/sent/safe
                node2_url = f"http://{NODE_2_IP}/process_stage"
                try:
                    r2 = requests.post(
                        node2_url,
                        json={
                            "stage": "rerank_llm_sentiment_safety",
                            "payload": {"queries": query_batch, "documents": documents},
                        },
                        timeout=300,
                    )
                    r2.raise_for_status()
                    node2_data = r2.json()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error calling Node2 LLM: {e}")

                generated = node2_data["responses"]
                sentiment = node2_data["sentiments"]
                toxicity = node2_data["toxicity"]

                processing_time = time.time() - start

            # Store result
            with results_lock:
                for req, gen, snt, tox in zip(reqs, generated, sentiment, toxicity):
                    response_payload = {
                        "request_id": req.request_id,
                        "generated_response": gen,
                        "sentiment": snt,
                        "is_toxic": "true" if tox else "false",
                        "processing_time": processing_time,
                    }
                    results[req.request_id] = response_payload

            request_queue.task_done()
        except Exception as e:
            print(f"[node {NODE_NUMBER}] Error in worker: {e}")
            traceback.print_exc()
            request_queue.task_done()


@app.route("/query", methods=["POST"])
def handle_query():
    """Handle incoming query requests if we're on node 0"""
    try:
        if NODE_NUMBER != 0:
            return jsonify(
                {"error": "This node does not accept client queries!! Send to Node 0."}
            ), 400

        data = request.json
        request_id = data.get("request_id")
        query = data.get("query")

        if not request_id or not query:
            return jsonify({"error": "Missing request_id or query"}), 400

        # Check if result already exists (request already processed)
        with results_lock:
            if request_id in results:
                return jsonify(results.pop(request_id)), 200

        print(f"queueing request {request_id}")
        # Add to queue
        request_queue.put({"request_id": request_id, "query": query})

        # Wait for processing (with timeout). Very inefficient - would suggest using a more efficient waiting and timeout mechanism.
        timeout = 300  # 5 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    return jsonify(result), 200

            if time.time() - start_wait > timeout:
                return jsonify({"error": "Request timeout"}), 504

            time.sleep(0.1)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/process_stage", methods=["POST"])
def process_stage():
    """
    Main pipeline execution for a batch of requests.
    Expects JSON: {"stage": <stage_name>, "payload": <payload>}
    Stages:
    - Node1: "faiss_and_fetch" (input: embedding list) -> {"documents": [...]}
    - Node2: "rerank_llm_sentiment_safety" (input: {"queries": [...], "documents": [...]}) -> {"responses": [...], "sentiments": [...], "toxicity": [...]}
    Node0 also exposes a local embed stage if needed.
    """
    try:
        data = request.json
        stage = data.get("stage")
        payload = data.get("payload")

        # Node 0 embeds
        if stage == "embed":
            if NODE_NUMBER != 0:
                return jsonify({"error": "embed stage only on node 0"}), 400
            # payload = [text1, text2, ...]
            emb = pipeline._generate_embeddings_batch(payload)
            return jsonify({"emb": emb.astype("float32").tolist()})

        # node 1 FAISS and fetches
        if stage == "faiss_and_fetch":
            if NODE_NUMBER != 1:
                return jsonify({"error": "faiss_and_fetch stage only on node 1"}), 400
            # payload is embedding list or list of lists
            emb_arr = np.array(payload, dtype="float32")
            doc_ids = pipeline._faiss_search_batch(emb_arr)
            documents = pipeline._fetch_documents_batch(doc_ids)
            return jsonify({"documents": documents})

        # node 2 rerank+ LLM + sentiment and safety
        if stage == "rerank_llm_sentiment_safety":
            if NODE_NUMBER != 2:
                return jsonify(
                    {"error": "rerank_llm_sentiment_safety stage only on node 2"}
                ), 400
            queries = payload.get("queries", [])
            documents = payload.get("documents", [])
            reranked = pipeline._rerank_documents_batch(queries, documents)
            responses = pipeline._generate_responses_batch(queries, reranked)
            sentiments = pipeline._analyze_sentiment_batch(responses)
            toxicity = pipeline._filter_response_safety_batch(responses)
            return jsonify(
                {"responses": responses, "sentiments": sentiments, "toxicity": toxicity}
            )

        return jsonify({"error": "unknown stage"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}
    ), 200


def main():
    """
    Main execution function
    """
    global pipeline
    print("=" * 60)
    print("DISTRIBUTED CUSTOMER SUPPORT PIPELINE")
    print("=" * 60)
    print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")
    print("\nNOTE: This is the MVP distributed version. Node responsibilities:")
    print("  Node 0: client handle, embedding")
    print("  Node 1: FAISS +document fetch")
    print("  Node 2: reranker +LLM +sentiment +safety")
    print("")

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = MonolithicPipeline()
    print("Pipeline initialized!")

    if NODE_NUMBER == 0:
        # node 0 should not run anything
        def not_allowed(*args, **kwargs):
            raise RuntimeError("This node is not responsible for this stage (NODE 0).")

        pipeline._faiss_search_batch = not_allowed
        pipeline._fetch_documents_batch = not_allowed
        pipeline._rerank_documents_batch = not_allowed
        pipeline._generate_responses_batch = not_allowed
        pipeline._analyze_sentiment_batch = not_allowed
        pipeline._filter_response_safety_batch = not_allowed

    elif NODE_NUMBER == 1:
        # node 1 for FAISS + fetch
        def not_allowed(*args, **kwargs):
            raise RuntimeError("This node is not responsible for this stage (NODE 1).")

        pipeline._generate_embeddings_batch = not_allowed
        pipeline._rerank_documents_batch = not_allowed
        pipeline._generate_responses_batch = not_allowed
        pipeline._analyze_sentiment_batch = not_allowed
        pipeline._filter_response_safety_batch = not_allowed

    elif NODE_NUMBER == 2:
        # node 2 : rerank + LLM + sentiment + safety
        def not_allowed(*args, **kwargs):
            raise RuntimeError("This node is not responsible for this stage (NODE 2).")

        pipeline._generate_embeddings_batch = not_allowed
        pipeline._faiss_search_batch = not_allowed
        pipeline._fetch_documents_batch = not_allowed

    else:
        raise RuntimeError("NODE_NUMBER must be 0,1, or 2 for this MVP")

    print("Pipeline initialized (node-specific methods constrained).")

    # Start worker thread on Node 0 to handle client queue
    if NODE_NUMBER == 0:
        worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
        worker_thread.start()
        print("Worker thread started on Node 0")

    # Start Flask server on this node
    node_ip = os.environ.get(f"NODE_{NODE_NUMBER}_IP", None)
    if node_ip:
        hostname = node_ip.split(":")[0]
        port = int(node_ip.split(":")[1]) if ":" in node_ip else 8000 + NODE_NUMBER
        print(f"Starting Flask on {hostname}:{port} (node {NODE_NUMBER})")
        app.run(host=hostname, port=port, threaded=True)

    else:
        raise RuntimeError("Node IP not working to open program.")


if __name__ == "__main__":
    main()
