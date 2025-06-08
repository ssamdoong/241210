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

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

# Sidebar API Key 입력
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    difficulty = st.selectbox("Select difficulty:", ["Easy", "Hard"])
    st.markdown("[View on GitHub](https://github.com/yourusername/your-repo-name)")

# API Key 없으면 멈춤
if not openai_api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

# LLM 초기화
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o",
    openai_api_key=openai_api_key,
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# Function schema 정의 (OpenAI Function Calling 사용)
functions = [
    {
        "name": "generate_quiz",
        "description": "Generate 10 quiz questions with 4 answer options each, marking the correct one.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {"type": "string"},
                                        "correct": {"type": "boolean"},
                                    },
                                    "required": ["answer", "correct"]
                                }
                            }
                        },
                        "required": ["question", "answers"]
                    }
                }
            },
            "required": ["questions"]
        }
    }
]

# 프롬프트 설정 (난이도 추가)
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant role playing as a teacher.
Create 10 quiz questions based ONLY on the context provided.

Each question should have 4 answers:
- 1 correct (marked with (o))
- 3 incorrect

Difficulty level: {difficulty}

Example:
Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(o)

Context: {context}
"""
        )
    ]
)

# Chain 구성
def generate_quiz(docs, difficulty):
    prompt = questions_prompt.format_messages(context=format_docs(docs), difficulty=difficulty)[0]
    response = llm.invoke(
        [SystemMessage(content=prompt.content)],
        functions=functions,
        function_call={"name": "generate_quiz"},
    )
    parsed = JsonOutputFunctionsParser().parse_result(response)
    return parsed

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

# Sidebar Input
with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        ("File", "Wikipedia Article"),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

# Main
if not docs:
    st.markdown(
        """
Welcome to QuizGPT.
                
I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
Get started by uploading a file or searching on Wikipedia in the sidebar.
"""
    )
else:
    if st.button("Generate Quiz"):
        response = generate_quiz(docs, difficulty)
        st.session_state["quiz"] = response
        st.session_state["score"] = 0
        st.session_state["submitted"] = False

    if "quiz" in st.session_state:
        quiz = st.session_state["quiz"]

        with st.form("questions_form"):
            user_answers = []
            for idx, question in enumerate(quiz["questions"]):
                st.write(f"Q{idx + 1}: {question['question']}")
                options = [ans["answer"] for ans in question["answers"]]
                user_choice = st.radio(
                    f"Select an option for Q{idx + 1}",
                    options,
                    key=f"q{idx}",
                )
                user_answers.append(user_choice)
            submitted = st.form_submit_button("Submit")

        if submitted:
            correct_count = 0
            for idx, question in enumerate(quiz["questions"]):
                selected = user_answers[idx]
                for ans in question["answers"]:
                    if ans["answer"] == selected and ans["correct"]:
                        correct_count += 1
            st.write(f"Your Score: {correct_count} / {len(quiz['questions'])}")

            if correct_count == len(quiz["questions"]):
                st.success("Perfect Score! 🎯")
                st.balloons()
            else:
                st.warning("Not a perfect score, try again!")

            st.session_state["score"] = correct_count
            st.session_state["submitted"] = True

        if st.session_state.get("submitted", False) and st.session_state["score"] < len(st.session_state["quiz"]["questions"]):
            if st.button("Retake Quiz"):
                del st.session_state["quiz"]
                del st.session_state["submitted"]
                del st.session_state["score"]
