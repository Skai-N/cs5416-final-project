#!/bin/bash
# This script will be executed on each node
echo "Starting pipeline on Node $NODE_NUMBER..."
python3 pipeline_dist.py