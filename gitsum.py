# GITSUM - GitHub Intelligence Summary
# Prototype MVP using Streamlit, LlamaIndex, LangChain, Repomix, AWS

from langchain_openai import ChatOpenAI
import streamlit as st
import os
import tempfile
import boto3
import json
from github import Github
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI as LlamaOpenAI
from llama_index.core.query_engine import RetrieverQueryEngine

# --- Load AWS Secrets ---
client = boto3.client('secretsmanager', region_name='ap-southeast-2')
secret = json.loads(client.get_secret_value(SecretId='GITSUM_KEYS')['SecretString'])
OPENAI_API_KEY = secret['OPENAI_API_KEY']
GITHUB_TOKEN = secret['GITHUB_TOKEN']
AWS_S3_BUCKET = 'gitsum-docs'

# --- Init Services ---
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
llm_wrapper = LlamaOpenAI(model="gpt-4", api_key=OPENAI_API_KEY, temperature=0.1)
service_context = ServiceContext.from_defaults(llm=llm_wrapper, embed_model=OpenAIEmbedding())

g = Github(GITHUB_TOKEN)
s3 = boto3.client('s3')

# --- Streamlit UI ---
st.title("🤖 GITSUM: GitHub Intelligence Summary")
st.markdown("Search, summarize, and query GitHub repos + PDFs with AI.")

keyword = st.text_input("Enter a keyword to search GitHub repos:")
search_button = st.button("Search & Summarize")

if search_button and keyword:
    with st.spinner("Searching GitHub and processing repos..."):
        repos = g.search_repositories(query=keyword, sort="stars", order="desc")
        repo_list = [repo for repo in repos[:3]]

        for repo in repo_list:
            st.subheader(repo.full_name)
            st.markdown(repo.description or "No description")

            with tempfile.TemporaryDirectory() as tmpdir:
                os.system(f"git clone {repo.clone_url} {tmpdir}")

                st.markdown("Generating summary with Repomix... 🔧")
                repomix_out = os.path.join(tmpdir, "repomix-summary.md")
                os.system(f"npx repomix {tmpdir} --output {repomix_out} --style markdown")

                # Upload PDFs to S3 (real logic)
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".pdf"):
                            local_path = os.path.join(root, file)
                            s3_key = f"{repo.name}/{file}"
                            s3.upload_file(local_path, AWS_S3_BUCKET, s3_key)

                # Load only the Repomix output file
                docs = SimpleDirectoryReader(input_files=[repomix_out]).load_data()
                index = VectorStoreIndex.from_documents(docs, service_context=service_context)
                query_engine = index.as_query_engine(similarity_top_k=5)

                st.session_state[f"engine_{repo.full_name}"] = query_engine
                st.success(f"Indexed {repo.full_name} ✅")

st.divider()
st.header("💬 Ask GITSUM")

question = st.text_input("Ask a question based on the indexed repositories:")
selected_repo = st.selectbox("Which repo to query?", options=[k for k in st.session_state if k.startswith("engine_")])
ask_button = st.button("Ask")

if ask_button and question and selected_repo:
    engine = st.session_state[selected_repo]
    with st.spinner("Querying the repo..."):
        response = engine.query(question)
        st.markdown("### 🤖 Answer")
        st.write(response.response)
        st.markdown("### 📄 Sources")
        for src in response.source_nodes:
            st.code(src.node.text[:1000])
