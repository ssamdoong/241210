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

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS





# ----- 프롬프트/체인 구성부 -----
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
    Examples:
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | ChatOpenAI(temperature=0.1, openai_api_key=st.session_state["openai_api_key"])
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke({"question": question, "context": doc.page_content}).content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", ""),
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | ChatOpenAI(temperature=0.1, openai_api_key=st.session_state["openai_api_key"])
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

@st.cache_data(show_spinner="Cloudflare 공식문서 임베딩 중...")
def load_cf_docs(api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        "https://developers.cloudflare.com/sitemap-0.xml",
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))
    return vector_store.as_retriever()

st.set_page_config(
    page_title="Cloudflare SiteGPT",
    page_icon="☁️",
)

st.markdown(
    """
    # ☁️ Cloudflare Docs SiteGPT
    Ask anything about Cloudflare's [AI Gateway](https://developers.cloudflare.com/ai-gateway/), 
    [Vectorize](https://developers.cloudflare.com/vectorize/), and [Workers AI](https://developers.cloudflare.com/workers-ai/) docs.
    \n
    _Start by entering your OpenAI API Key in the sidebar._  
    _Docs are loaded from [Cloudflare 공식 Sitemap](https://developers.cloudflare.com/sitemap-0.xml)_
"""
)

with st.sidebar:
    st.markdown("## 1. Enter your OpenAI API Key")
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
    st.markdown("## 2. See the source code:")
    st.markdown("[🔗 GitHub Repo](https://github.com/yourusername/sideGPT-cloudflare)")
    st.markdown("---")
    st.markdown("### Cloudflare Docs Quick Links")
    st.markdown("- [AI Gateway](https://developers.cloudflare.com/ai-gateway/)")
    st.markdown("- [Vectorize](https://developers.cloudflare.com/vectorize/)")
    st.markdown("- [Workers AI](https://developers.cloudflare.com/workers-ai/)")

if openai_api_key:
    retriever = load_cf_docs(openai_api_key)
    query = st.text_input("Ask a question about Cloudflare Docs.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))
else:
    st.info("Please enter your OpenAI API Key in the sidebar to start.")