#!/bin/bash

# Exit on error
set -e

# Print a message
echo "Starting the pipeline..."

# Run Python scripts
python3 src/data_ingestion.py
python3 src/data_preprocessing.py
python3 src/training.py

echo "Pipeline completed successfully!"
