import streamlit as st
import openai
import os
import time
import json

# ===== Streamlit UI =====
st.set_page_config(page_title="Research Assistant", page_icon="🔎")

with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    st.markdown("---")
    st.markdown("[GitHub Repo](https://github.com/yourusername/your-repo-url)")

st.title("🔎 Research Assistant")
st.write("Ask any research question! Tools: Wikipedia, DuckDuckGo, Web Scraper, Save-to-TXT.")

if not openai_api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
    st.stop()

openai_client = openai.OpenAI(api_key=openai_api_key)

# ===== Function Definitions for Assistant =====

def wikipedia_search(query):
    from langchain.utilities import WikipediaAPIWrapper
    return WikipediaAPIWrapper().run(query)

def duckduckgo_search(query):
    from langchain.tools import DuckDuckGoSearchResults
    return DuckDuckGoSearchResults().run(query)

def web_scraper(url):
    from langchain.document_loaders import WebBaseLoader
    loader = WebBaseLoader([url])
    docs = loader.load()
    return "\n\n".join([doc.page_content for doc in docs])[:2000]

def save_to_txt(text):
    with open("research_results.txt", "w", encoding="utf-8") as file:
        file.write(text)
    return "Results saved to research_results.txt"

# ===== Register Functions as OpenAI Assistant Tools =====
functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search and summarize information from Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Search DuckDuckGo for the most recent results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_scraper",
            "description": "Extract the main text content from a web page URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_txt",
            "description": "Save a string to a TXT file.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        }
    },
]

# ===== Session State: For Assistant, Thread, Messages =====
if "assistant_id" not in st.session_state:
    assistant = openai_client.beta.assistants.create(
        instructions=(
            "You are a research expert. Use Wikipedia, DuckDuckGo, and Web Scraping tools to gather detailed answers. "
            "When you find a website through DuckDuckGo, scrape the content using the web scraper. "
            "Summarize and cite all sources, and save the final research to a .txt file using the save_to_txt function. "
            "Include Wikipedia in your findings."
        ),
        tools=functions,
        model="gpt-4o"
    )
    st.session_state["assistant_id"] = assistant.id

if "thread_id" not in st.session_state:
    thread = openai_client.beta.threads.create()
    st.session_state["thread_id"] = thread.id

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ===== Chat UI =====
def display_chat(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

display_chat(st.session_state["messages"])

user_input = st.chat_input("Ask your research question!")
if user_input:
    # Add user message to UI and OpenAI Thread
    st.session_state["messages"].append({"role": "user", "content": user_input})

    openai_client.beta.threads.messages.create(
        thread_id=st.session_state["thread_id"],
        role="user",
        content=user_input
    )

    with st.spinner("Researching..."):
        run = openai_client.beta.threads.runs.create(
            thread_id=st.session_state["thread_id"],
            assistant_id=st.session_state["assistant_id"]
        )

        # Main loop: polling until run is finished, handling tool/function calls
        while True:
            run_status = openai_client.beta.threads.runs.retrieve(
                thread_id=st.session_state["thread_id"], run_id=run.id
            )

            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                st.error("Run failed.")
                break
            elif run_status.status == "cancelled":
                st.error("Run cancelled.")
                break
            elif run_status.status == "requires_action":
                # ===== Function Calling Handling =====
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    # 실제 파이썬 함수 실행
                    try:
                        if function_name == "wikipedia_search":
                            output = wikipedia_search(**args)
                        elif function_name == "duckduckgo_search":
                            output = duckduckgo_search(**args)
                        elif function_name == "web_scraper":
                            output = web_scraper(**args)
                        elif function_name == "save_to_txt":
                            output = save_to_txt(**args)
                        else:
                            output = f"Unknown function: {function_name}"
                    except Exception as e:
                        output = f"Tool {function_name} execution failed: {str(e)}"
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": output})
                # Assistant API에 tool_outputs로 전송
                openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=st.session_state["thread_id"],
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
            else:
                time.sleep(2)

        # Get all messages in thread
        messages = openai_client.beta.threads.messages.list(thread_id=st.session_state["thread_id"])
        # Messages come in reverse chronological order
        assistant_msgs = [
            {"role": m.role, "content": m.content[0].text.value}
            for m in reversed(messages.data) if m.role == "assistant"
        ]
        # Update UI message history
        if assistant_msgs:
            st.session_state["messages"].extend(assistant_msgs)
        display_chat(st.session_state["messages"])

# Streamlit UI/UX Tip: Reset thread/assistant with a button if needed
if st.sidebar.button("Reset Conversation"):
    for key in ["assistant_id", "thread_id", "messages"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()
