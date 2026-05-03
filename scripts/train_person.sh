#!/bin/bash
# Script to run person-level training pipeline

# Set PYTHONPATH so that imports from 'src' work correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "Starting Person-Level Training..."
python src/training/training_person.py
