import streamlit as st
from agent.react_agent import ReactAgent
from agent.memory import ConversationMemory
from model.factory import get_chat_model

st.set_page_config(page_title="智能助手", page_icon="🎮")
st.title("🎮 英雄联盟专业智能问答助手")
st.caption("基于 LangChain ReAct Agent + RAG 检索增强")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationMemory(get_chat_model())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    summary, recent = st.session_state["memory"].get_context()

    response_messages: list[str] = []
    with st.spinner("智能助手思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt, summary=summary, history=recent)

        def stream_generator(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    yield char

        st.chat_message("assistant").write_stream(stream_generator(res_stream, response_messages))
        full_answer = "".join(response_messages)
        st.session_state["messages"].append({"role": "assistant", "content": full_answer})
        st.session_state["memory"].add_turn(prompt, full_answer)
        st.rerun()
