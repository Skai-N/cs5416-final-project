# CS 5416: Final Project
## Building and Optimizing a Distributed ML Inference Pipeline


## 0. Logistics
**Out:** November 7, 2025  
**Preliminary Check (required but not graded):** December 4, 2025 at 11:59pm  
**Final Project Due:** December 10, 2025 at 11:59pm. **There will be NO late days!**  
**Work in groups of 4 (minimum 3)**

Please read the project description carefully. If you have any questions (e.g. clarifications about the requirements, an idea for an unlisted optimization, appointment for running on GPU nodes, etc), please **post on Ed using the new CS 5416 final project tag**

## 1. Introduction

In real production systems, machine learning is almost never "just call the model." Inputs are validated, transformed, enriched, routed, and sometimes filtered before the model ever runs. Then the model output is post-processed, checked for safety, logged, and sent to other services.

One real-world example of this would be an LLM with tool calls or web search. That system might operate like:
- User sends query
- System classifies intent
- System calls a vector store / web search / internal tool
- Retrieved content is merged into a prompt
- LLM is called with context
- Output is safety-filtered and returned

Another example might be a fraud detection pipeline that operates as:
- Transaction arrives
- Features are constructed from recent customer history
- Model is run
- Output is thresholded and maybe sent to a second model or rule engine
- Alert is sent to customer service or the transaction is blocked

The point is: production ML = pipelines. And pipelines = systems problems (distribution, batching, network, memory, GPU/CPU placement, orchestration, correctness).

This project is designed to give you hands-on practice in taking a naive, monolithic ML pipeline and turning it into a distributed, optimized version. You will explore various optimization strategies and profile real tradeoffs.

This is the kind of work ML systems engineers actually do.

## 2. Objective

You will be given a barebones monolithic implementation of an ML pipeline (details provided in the starter code section). Your job is to:

1. **Transform it into a distributed service that runs across 3 nodes**
2. **Implement batching**
3. **Profile your design** and report on the effects of different optimizations
4. **Show that you did systems work** (even if an unoptimized version is faster, the primary objective of this project is to try different things and see how they affect performance)

You may employ advanced tactics for higher grades such as:
- Breaking apart the monolithic pipeline into microservices
- Implementing opportunistic batching
- Caching strategies
- GPU acceleration
- Data compression for network transfer
- And more... This is an open ended project, so any other optimizations that you can think of, you may implement

## 3. Grading Tiers

### **Grading Philosophy**
Grades for this project will be primarily based on how many systems concepts you successfully implement and analyze, NOT on raw performance. A slower but well-engineered microservice implementation with thorough profiling and a good report will still receive a good grade.

### Tier 1: Passing Grade (B range)
**Minimum Requirements:**
- Pipeline successfully runs across 3 nodes (can be 3 monolithic instances of the pipeline)
- Basic batching implementation (static batch sizes are fine, opportunistic batching not necessary)
- Node 0 receives client requests and returns responses in accordance with the spec
- Basic profiling showing throughput and latency of the pipeline vs batch size
- Correctness maintained
- Code runs within hardware constraints (16GB RAM per node)

### Tier 2: Good Grade (A- or above)
**All Tier 1 requirements PLUS:**
- True microservices architecture (pipeline separated into distinct parts that communicate with each other cross-node)
- Orchestration and request routing across nodes
- More robust profiling (details in report section)
- Clear documentation of design decisions in report
- Explanation of how your profiling led to your final configuration

### Tier 3: Excellent Grade (A or above)
**All Tier 2 requirements PLUS:**
- Opportunistic batching: Each step or microservice can take different batch sizes depending on load
And 1 or more of the following:
- A switch to turn on GPU acceleration
- Caching strategies
- Running stages simultaneously when possible
- Data compression for inter-node communication
- Exceptional profiling with detailed analysis of system bottlenecks
- Top tier performance on the leaderboard
- Fault tolerance or recovery mechanisms
- Other creative optimizations

## 4. Scenario

You've been "hired" by a large e-commerce company to fix their customer support AI pipeline. Their current version is a single Python script that runs everything in one process. It cannot handle production load.

Your task: implement the pipeline as a distributed system across 3 nodes.

The pipeline stages are:
1. **Embedder** on the query string
2. **FAISS RAG lookup** (ANN) — retrieve top-10 docs using the embedding
3. **Fetch documents** from disk based on step 2
4. **Reranker** on the docs from step 3
5. **LLM response generation** (max 128 tokens) using the reranked docs from step 4
6. **Sentiment analysis model** on the LLM output from step 5
7. **Sensitivity filter** on the LLM output from step 5
8. **Return to client:**
    - generated response
    - sentiment analysis result
    - sensitivity filter result

**Important:**
- The starter code does this monolithically
- Your submission must do this across 3 nodes
- Our client will only send requests to Node 0; you must route/orchestrate the rest

## 5. Provided Starter Code

Starter code is inside this repository

It Contains a monolithic pipeline that already performs the stages above. You must preserve accuracy/correctness while distributing and optimizing it.

It also contains a client script that follows the same spec that we will use to test your implementations. It is essential that your implemetation is compatible with this script.

Note that the given monolothic pipeline is not at all optimal. Since the combined size of the components is too large to fit in 16gb of memory, it has to remove from and reload components into memory during a single pipeline execution. Furthermore, it only uses a batch size of 1.

To run the starter code, 
- Create python venv if desired
- Install dependencies if not installed `pip install -r requirements.txt`
- Create dummy FAISS index and documents using `python3 create_test_docs.py`
- Find local IP `ifconfig`
- Run pipeline server `TOTAL_NODES=1 NODE_NUMBER=0 NODE_0_IP=<your ip>:<port of choice> ./run.sh`
- Run client `NODE_0_IP=<ip:port of pipeline server> python3 client.py`

## 6. System Requirements / Spec

### 6.1 Overall Requirement
Your pipeline must run on 3 nodes. You must provide:
- `install.sh`
- `run.sh`

We (the TAs) will run three separate processes (one per node), each calling your `run.sh`.

### 6.2 Config and Environment Variables (given by us)
When we run your `run.sh`, we will provide the following environment variables:
- `TOTAL_NODES` — will be 3
- `NODE_NUMBER` — will be 0, 1, or 2
- `NODE_0_IP`, `NODE_1_IP`, `NODE_2_IP` — IPs for each node
- `FAISS_INDEX_PATH` – Filepath for the pre-built FAISS index. Assume that the FAISS index will be up to 13gb in size
- `DOCUMENTS_DIR` – Directory for the collection of documents

You can use these to:
- Differentiate per-node roles
- Configure services
- Set up routing between nodes

**Client behavior:**
- Our client script will send requests one by one to the Node 0 IP address
- Node 0 must orchestrate the full pipeline and return the final response to the client

### 6.3 Request / Response Format
Request format:
- `request_id: string`
- `query: string`

Your final response to the client must include:
- `request_id: string`
- `generated_response: string`
- `sentiment: string | "very negative" or "negative" or "neutral" or "positive" or "very positive"`
- `is_toxic: string | "true" or "false"`

### 6.4 Implementation Examples

#### Basic Implementation (Passing Grade)
- **Simple distribution:** Can run 3 copies of the monolith, each handling the full pipeline
- **Basic batching:** Fixed batch sizes, process when batch is full
- **Simple routing:** Node 0 can round-robin or randomly distribute to other nodes
- **Basic profiling:** Memory usage, throughput measurements in report

#### Advanced Implementation (Higher Grades)
- **Microservices:** Separate pipeline stages into distinct services
- **Opportunistic batching:** Each step or microservice can take different batch sizes depending on load
- **Smart orchestration:** Optimize stage placement based on compute requirements
- **Performance-aware data passing:** Minimize data transfer, transfer IDs instead of full documents
- **Comprehensive profiling:** Detailed analysis of bottlenecks and tradeoffs

## 7. Allowed Environment

**Preinstalled:**
- torch
- transformers
- numpy
- datasets
- jinja2
- faiss
- requests
- flask
- sentence_transformers

You may request additional packages from Jamal if:
- They do not require sudo
- They do not abstract away what you are supposed to implement (e.g. you cannot use a library that automatically handles opportunistic batching for you)

You may use any programming language as long as you follow the spec, but Python is strongly recommended (starter code is Python).

## 8. Autograder & Leaderboard Hardware / Runtime Constraints

**3 nodes, each with:**
- CPU: Intel(R) Xeon(R) Gold 6242 @ 2.80GHz
- 16 GB RAM
- GPU: NVIDIA Tesla T4, 15 GB VRAM

**Key note:** 16 GB RAM is a hard constraint. If you go over, you'll hit swap → very slow.

**Initialization time:** We will wait for 5 minutes after calling `./run.sh` for any initialization you may have before sending your system requests

## 9. GPU Support (Optional)

GPU support is **NOT required** but can be an optimization for higher grades.

If you choose to implement GPU support:
- Post on Ed or meet with Jamal after recitation requesting an appointment to test your GPU-enabled code on the GPU nodes
- Make sure your code can still run fully on CPU if the environment variable `ONLY_CPU=true` is passed into your `run.sh`
- Document in your report:
  - What you offloaded to GPU
  - Where there were CPU<->GPU transfers
  - Whether you used both CPU and GPU simultaneously

## 10. Deliverables

### 10.1 Codebase
Submit your full codebase with:
- `install.sh` (no sudo)
- `run.sh` (entry point for each node)
- Full source code to run your pipeline
- Any code related to the experiments or microbenchmarks you ran

### 10.2 Report

**IMPORTANT:** The profiling that you do for the report should be done on UGClinux or your group's local computers. If you implement GPU acceleration, you do not need to worry about running your profiling on GPUs. As long your GPU implentation works, you will receive full credit for your GPU implementation even if your report profiling is done on CPU

**Report Weight:** Your report is an important part of your grade. A well documented system with thorough analysis will earn a higher grade than a poorly motivated one

**Report Goal:** The below requirements are there to give you ideas of what to include in your report. Since this is an open ended project, every group may implement different different optimizations or perform different experiments. The goal of the report is to:

1. Explain what you implemented and how you implemented it
2. Show us the thought processes behind your design decisions and the experiments that motivated what you ended up choosing 

Reports that better achieve these goals will receive higher grades

Your report must include:

#### Required Content (All Submissions):
1. **System design overview**
   - How you distributed work across 3 nodes
   - How requests flow through your system
   - Diagrams are encouraged for this

2. **System implementation details**
   - Detail what exactly you implemented or changed beyond the starter code. E.g.:
      - Your setup for routing requests
      - How you implemented your optimizations
      - Anything else you implemented
   - Explain any tools/packages you used in your implementation

3. **Batching implementation**
   - What stages use batching?
   - What was your choice of batch size, and what experiments informed that decision?
   - Plots are encouraged for this

4. **Basic profiling**
   - Show the peak memory used by your processes per node
   - Show the tradeoffs between throughput and memory usage across different choices of batch size for your pipeline
   - Measure the maximum throughput in requests/minute that your system can handle
   - Measure the latency of your system when under low load
   - Charts/Plots are encouraged here

#### Additional Content for Higher Grades:
5. **Optimization steps**
   - An explanation and a diagram showing how you broke up your system into microservices
   - An explanation and a diragram of how requests are routed through your system
   - If implemented, what your opportunistic batching strategy was and why you chose the configuration you did
   - If implemented, detail other optimizations you made (e.g. caching, data transfer, etc)
   - You should both explain the techincal implementation details of your optimization steps as well as justification for why you chose them

6. **Comprehensive experiments**
   - The exact experiments that you run are up to you. Your goal should be to run experiments that will help you understand your system and decide how to tune your optimizations. For example, you could:
      - Run microbenchmarks to profile the throughput and memory for each pipeline stage versus different batch sizes to help you decide how to break them up into microservices
      - Evaluate the performance of two different choices of microservice pipeline structure (something like having 1 faiss and 2 llm vs 2 faiss and 1 llm)
      - Explore the effects of tunable facets of your optimizations on latency and throughput
      - Isolate the impact of each individual optimization you made to see its effect by measuring throughput with it on/off
      - Identify were systems-level bottlenecks lie
      - Any other experiments you feel would help you optimize your pipeline
   - You should run experiments that will actually help you decide how to optimize your pipeline and document the process and results in your report
   - Charts/Plots encouraged

7. **Relating experiments to design decisions***
   - Justification of why you chose the experiments you chose to run 
   - Explanation of how your experiments informed the structure of your pipeline and the parameters of optimizations you made
   - Discussion of any tradeoffs you encountered 

8. **GPU section** (if implemented)
   - What you offloaded to GPU
   - Where there were CPU<->GPU transfers
   - Whether you used both CPU and GPU simultaneously

## 11. Example Architectures

These architectures are not optimal, they are just examples. Do your own analysis!

### Basic Architecture
- Node 0: Full pipeline, receives requests
- Node 1: Full pipeline, processes forwarded requests
- Node 2: Full pipeline, processes forwarded requests
- Simple load distribution from Node 0

### Advanced Architecture
- Node 0: Frontend + embedder + orchestration
- Node 1: FAISS + document retrieval + reranking
- Node 2: LLM + sentiment + sensitivity filters
- Request routing with opportunistic batching at each stage

## 12. Evaluation Criteria

### Core Requirements (Must Have to Pass):
- ✅ Runs on 3 nodes
- ✅ Node 0 handles client interface
- ✅ Some form of batching
- ✅ Maintains correctness
- ✅ Basic profiling data
- ✅ Stays within 16GB RAM limit

### Systems Concepts:
- Microservices architecture
- Opportunistic batching
- Request orchestration
- Performance profiling
- Caching strategies
- GPU utilization
- Network optimization
- Load balancing
- Any other optimizations you may think of

**Remember:** We value systems engineering and analysis over raw performance.

## 13. Testing / Logistics

You must test on 3 machines (3 laptops or 3 ugclinux instances). That's why groups of at least 3 are required.

## 14. Timeline

- **November 7** – Project released
- **December 4** – Preliminary submission (required)
  - Used to verify we can run your code
  - No direct effect on final grade
- **December 10** – Final project due. No late days allowed!

## 15. What to Turn In (Checklist)

### Required:
- [ ] `install.sh` (no sudo)
- [ ] `run.sh` (entry point; respects env vars)
- [ ] Full source code
- [ ] Report (PDF or md) with required sections
