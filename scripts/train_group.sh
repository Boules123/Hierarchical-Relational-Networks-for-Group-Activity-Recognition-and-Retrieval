#!/bin/bash
# Script to run group-level relational model training pipeline

# Set PYTHONPATH so that imports from 'src' work correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "Starting Group-Level Training..."
python src/training/training_group.py
