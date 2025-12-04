monolithic: monolithic_pipeline.py
	TOTAL_NODES=1 NODE_NUMBER=0 NODE_0_IP=localhost:8000 python3 monolithic_pipeline.py

client: client.py
	NODE_0_IP=localhost:8000 python3 client.py