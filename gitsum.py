# GITSUM - GitHub Intelligence Summary
# Prototype MVP using Streamlit, LlamaIndex, LangChain, Repomix, AWS

from langchain_openai import ChatOpenAI
import streamlit as st
import os
import tempfile
import boto3
import json
import time
from github import Github
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import openai

# --- Load AWS Secrets ---
client = boto3.client('secretsmanager', region_name='ap-southeast-2')
secret = json.loads(client.get_secret_value(SecretId='GITSUM_KEYS')['SecretString'])
OPENAI_API_KEY = secret['OPENAI_API_KEY']
GITHUB_TOKEN = secret['GITHUB_TOKEN']
AWS_S3_BUCKET = 'gitsum-docs'

# --- Init Services ---
openai.api_key = OPENAI_API_KEY
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

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
                os.system(f"npx repomix {tmpdir} --output {repomix_out} --style markdown --ignore '**/data/** **/notebooks/**'")

                if os.path.getsize(repomix_out) > 100000:
                    st.warning("⚠️ Repo summary too large to embed. Try a smaller repo.")
                    continue

                # ✅ Truncate summary if it's too big
                if os.path.exists(repomix_out):
                    with open(repomix_out, "r") as f:
                        content = f.read()

                    if len(content) > 50000:
                        content = content[:50000]  # trim to 50k characters
                        with open(repomix_out, "w") as f:
                            f.write(content)

                if not os.path.exists(repomix_out):
                    st.error("❌ Repomix summary was not generated. Skipping this repo.")
                    continue  # skip to the next repo

                # Upload PDFs to S3
                pdfs_uploaded = 0
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".pdf"):
                            local_path = os.path.join(root, file)
                            s3_key = f"{repo.name}/{file}"
                            s3.upload_file(local_path, AWS_S3_BUCKET, s3_key)
                            pdfs_uploaded += 1
                if pdfs_uploaded == 0:
                    st.info("ℹ️ No PDFs found in this repository.")
                else:
                    st.success(f"✅ Uploaded {pdfs_uploaded} PDFs to S3.")

                # ✅ Show the user the Repomix summary (first 3000 chars)
                st.markdown("### 📦 Repomix Summary Preview")
                st.code(content[:3000], language="markdown")

                # ✅ Embed and index with feedback
                with st.spinner("Embedding & indexing the summary... ⏳"):
                    start = time.time()
                    docs = SimpleDirectoryReader(input_files=[repomix_out]).load_data()
                    index = VectorStoreIndex.from_documents(docs)
                    
                    retriever = index.as_retriever(similarity_top_k=5)

                    template = """Use the context to answer the question.
                    If the answer is not in the context, say you don't know.

                    Context:
                    {context}

                    Question: {question}
                    """

                    prompt = PromptTemplate(
                        input_variables=["context", "question"],
                        template=template,
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,  # this is your langchain_openai.ChatOpenAI model
                        retriever=retriever,
                        chain_type="stuff",
                        chain_type_kwargs={"prompt": prompt},
                    )

                    st.session_state[f"engine_{repo.full_name}"] = qa_chain


                    st.success(f"✅ Indexed in {time.time() - start:.2f} seconds")
                st.success(f"Indexed {repo.full_name} ✅")

st.divider()
st.header("💬 Ask GITSUM")

question = st.text_input("Ask a question based on the indexed repositories:")
selected_repo = st.selectbox("Which repo to query?", options=[k for k in st.session_state if k.startswith("engine_")])
ask_button = st.button("Ask")

if ask_button and question and selected_repo:
    engine = st.session_state[selected_repo]
    with st.spinner("Querying the repo..."):
        response = engine.run(question)
        st.markdown("### 🤖 Answer")
        st.write(response)