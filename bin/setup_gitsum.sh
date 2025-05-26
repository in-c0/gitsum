#!/bin/bash

# --- GITSUM EC2 SETUP SCRIPT ---

echo "🔧 Updating system..."
sudo apt update && sudo apt upgrade -y

echo "🐍 Installing Python and dependencies..."
sudo apt install -y python3-pip git unzip curl nodejs npm

echo "📦 Installing required Python packages..."
pip3 install --upgrade pip
pip3 install streamlit boto3 llama-index langchain openai PyGithub

echo "🌐 Cloning GITSUM repo..."
git clone https://github.com/in-c0/gitsum.git
cd gitsum

echo "🔐 Setting up environment..."
# Ensure AWS credentials are configured, or assume an EC2 role with SecretsManager access
# Optional: export manually if not using Secrets Manager
# export OPENAI_API_KEY=...
# export GITHUB_TOKEN=...

echo "🚀 Launching Streamlit app..."
streamlit run gitsum.py --server.port 8501 --server.address 0.0.0.0
