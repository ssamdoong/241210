import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import CharacterTextSplitter

s# 각 프로덕트 공식문서 메인 URL (설명, 용도 표시)
AI_GATEWAY_URL = "https://developers.cloudflare.com/ai-gateway/"
VECTORIZER_URL = "https://developers.cloudflare.com/vectorize/"
WORKERS_AI_URL = "https://developers.cloudflare.com/workers-ai/"
SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"

with st.sidebar:
    st.title("☁️ SiteGPT - Cloudflare Docs Chatbot")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    st.markdown("[🔗 GitHub Repo](https://github.com/yourusername/sideGPT-cloudflare)")

    # 공식문서 빠른 링크
    st.markdown("**Docs Quick Links**")
    st.markdown(f"- [AI Gateway]({AI_GATEWAY_URL})")
    st.markdown(f"- [Vectorize]({VECTORIZER_URL})")
    st.markdown(f"- [Workers AI]({WORKERS_AI_URL})")

st.title("💬 Cloudflare Docs Chatbot")
st.caption("AI Gateway / Vectorize / Workers AI 공식 문서 기반 Q&A")

if not openai_api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner="문서 임베딩 및 인덱싱 중...")
def get_retriever(api_key):
    # 하나의 사이트맵에서 전체 문서 수집
    loader = SitemapLoader(web_path=SITEMAP_URL)
    all_docs = loader.load()
    # 필요시 아래처럼 제품별로 필터링 가능 (참고용, 지금은 전체 문서 사용)
    # all_docs = [doc for doc in all_docs if any(
    #     x in doc.metadata['source'] for x in [AI_GATEWAY_URL, VECTORIZER_URL, WORKERS_AI_URL]
    # )]
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=700, chunk_overlap=100)
    split_docs = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = FAISS.from_documents(split_docs, embeddings)
    return vectordb.as_retriever()

retriever = get_retriever(openai_api_key)
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Cloudflare 공식문서에 대해 궁금한 점을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Cloudflare 공식문서를 검색 중..."):
            answer = qa_chain.run(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})