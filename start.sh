#!/bin/bash
# ============================================
# Quick start script for the RAG Chatbot
# ============================================

set -e

echo "🚀 RAG Video Comparison Chatbot — Starting Up"
echo "=============================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys before running again."
    echo "   Required: GROQ_API_KEY (get free at https://console.groq.com/keys)"
    exit 1
fi

# Check Docker for Redis
echo "🐳 Checking Redis via Docker..."
if ! docker ps | grep -q rag-redis; then
    echo "   Starting Redis container..."
    docker compose up redis -d
    sleep 2
else
    echo "   ✅ Redis already running"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Start the server
echo ""
echo "=============================================="
echo "🟢 Starting FastAPI server..."
echo "   API Docs:  http://localhost:8000/docs"
echo "   Health:    http://localhost:8000/api/health"
echo "=============================================="

cd "$(dirname "$0")"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
