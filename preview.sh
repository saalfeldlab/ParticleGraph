#!/bin/bash
# Clean and preview NeuralGraph documentation

set -e

echo "Cleaning previous build..."
rm -rf docs
rm -rf .quarto/preview

echo "Rendering site..."
quarto render

echo "Starting preview server..."
quarto preview
