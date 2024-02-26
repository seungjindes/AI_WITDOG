from openai import OpenAI
import streamlit as st
import time




thread_id = "thread_nUlCEvOH1h5PB7x4xorily59"
assistant_id = "asst_r5LJYwmTq7XLjUoWPI0Hshda"

st.subheader(f"{thread_id}",divider="rainbow")
    
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

st.title("💬 Petbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요 윗독 서비스의 펫봇입니다 🐶"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    response = client.beta.threads.messages.create(
        thread_id,
        role="user",
        content= prompt
    )


    run = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assistant_id
    )

    run_id = run.id
    while True:

        run = client.beta.threads.runs.retrieve(
        thread_id = thread_id,
        run_id = run_id
        )

        if run.status =='completed':
            break
        else : time.sleep(2)

    thread_messages = client.beta.threads.messages.list(thread_id)
    #print(thread_messages.data)

    msg = thread_messages.data[0].content[0].text.value
    #print(msg)

    st.session_state.messages.append({"role" : "assistant" , "conetent" : msg})
    st.chat_message("assistant").write(msg)
