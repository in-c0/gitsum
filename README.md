GITSUM Frontend Demo:
https://youtube.com/shorts/8cLDQyLH_zU?feature=share

GITSUM Frontend + Backend Demo:
https://youtu.be/jFZptm7inQA

**GITSUM** is a web app that allows search / AI summary / Q&A on open-sourced Github repositories.

 First, it lets users search public GitHub repositories by keyword. It then uploads any PDFs found in the repository to Amazon S3, while summarizing the repository using Repomix, and queries the PDF contents with a Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex  +  OpenAI embeddings, for more accurate answers from Q&A chatbots. (LangChain is used for prompt control.)

## 🔧 Step-by-step User Flow

1. 🔍 Search GitHub repos with a keyword
2. 📦 Top 3 repos are summarized via Repomix CLI
3. 🧠 Indexing and querying with LangChain + LlamaIndex + OpenAI
4. 📄 PDF detection and upload to AWS S3
5. 🖥️ Streamlit UI for real-time interaction

The backend is deployed directly on an AWS EC2 instance. 
http://54.253.228.15:8501/


```bash
# activate virtualenv
source ~/gitsum-env/bin/activate

# run app
streamlit run gitsum.py --server.port 8501 --server.address 0.0.0.0
```