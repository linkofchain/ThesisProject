#!/bin/bash

# Check if .venv already exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source .venv/bin/activate

#echo "Installing dependencies..."
#pip install -r requirements.txt

echo "Done!"
