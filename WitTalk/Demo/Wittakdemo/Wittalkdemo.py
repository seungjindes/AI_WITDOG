from openai import OpenAI
import streamlit as st
import time
import json 
from TTS.tts import voice_generator
import librosa
import whisper
from tempfile import NamedTemporaryFile
from utils import DataPipeline , get_emotion_using_triton

 

JSON_PATH = "./asset/wittalkdemo.json"
USER = "별이"
DOG_EMO = None
ASSISTANT_ID = "asst_4AcrN8f4LuWzP5DyR8cRdaTG"

with open(JSON_PATH , 'r' , encoding='utf-8') as f:
    json_data = json.load(f)

pipleline = DataPipeline()


# 반려견의 음성 파일을 업로드하고, 해당 음성에 맞는 감정 상태를 추출합니다. 
# 감정추출의 결과는 좌측 하단에 체크박스에 표기됩니다.
# 감정추출은 Bark2text 모델이 triton 서버에서 추론하게됩니다. 따라서 triton 서버가 동작하고 있어야합니다.
upload_dogsound = st.file_uploader("Upload an dog audio(.wav) file", type=["wav"])
if upload_dogsound is not None:
    with NamedTemporaryFile(suffix=".wav") as temp:
        temp.write(upload_dogsound.getvalue())
        temp.seek(0)
        temp_dogsound  ,sr = librosa.load(temp  ,sr = 16000)
        #st.audio(temp_dogsound , sample_rate = 16000)
        input_data= pipleline.execute(temp_dogsound , sr)
        DOG_EMO = get_emotion_using_triton(input_data)


with st.sidebar:

    
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot password", type="password")
    client = OpenAI(api_key=openai_api_key)
    thread_id = json_data[USER]['thread_key']


    action = st.text_input("반려견의 현재 상황" , key = "action")

    sample_sound , sr= librosa.load('./TTS/resources/example_reference2.mp3' , sr = 44100)
    st.audio(sample_sound ,format = "audio/mp3" , sample_rate = 44100)

    st.caption("위 오디오는 사용자 목소리 샘플입니다.")


    # check box를 통해 반려견의 현재 감정을 보여줌 
    st.write('반려견 현재 감정')
    
    if DOG_EMO == 'bark':
        happy = st.checkbox('행복해' , value = True , disabled = True)
    else:
        happy = st.checkbox('행복해' , value = False , disabled = True)

    if DOG_EMO == 'whining':
        sad = st.checkbox('슬퍼' , value = True, disabled = True)
    else:
        sad = st.checkbox('슬퍼' , value = False, disabled = True)

    if DOG_EMO == 'growling':
        angry = st.checkbox('화난다' , value = True, disabled = True)
    else:
        angry = st.checkbox('화난다' , value = False, disabled = True)

    if DOG_EMO == 'howling':
        fear = st.checkbox('두렵다' , value = True, disabled = True)
    else:
        fear = st.checkbox('두렵다 ' , value = False, disabled = True)

    if happy == True:
        emotion = "행복함"
    elif angry == True:
        emotion = "화남"
    elif sad == True:
        emotion = "슬픔"
    elif fear == True:
        emotion = "두렵다"
        



st.title("🐶 WitTalk Demo")
st.subheader(f"현재 윗톡은 {USER}와 대화중이에요",divider=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": f"안녕하세요 저는 {USER}에요. 멍!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    #텍스트에서 음성으로 변환하는 부분
    # 하나는 Openai에서 제공하는 기본 tts , 그리고 나의 목소리를 입힌 tts두가지 버전을 제공한다.
    tts_speaker , my_spearker = voice_generator(prompt)
    #tts_sound  , sr = librosa.load(tts_speaker , sr = 44100)
    my_sound  , sr= librosa.load(my_spearker , sr = 44100)

    st.write("사용자의 채팅에 대해 사용자 음성입힌 TTS를 제공합니다.")
    #st.audio(tts_sound , format = "audio/wav" , sample_rate= 44100)
    st.audio(my_sound , format = "audio/wav" , sample_rate= 44100)

    # inference server 에서 감정값 emotion
    # action은 카메라에서 받아와야함 motion capture해서 강아지가 있는지 없는지

    text_form = f"""
                    상황 : {action}      
                    감정상태 : {emotion}
                    채팅 : {prompt}
                    
                """
    print(text_form)    
    
    response = client.beta.threads.messages.create(
        thread_id,
        role="user",
        content= text_form
    )

    run = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=ASSISTANT_ID
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

    msg = thread_messages.data[0].content[0].text.value
    st.session_state.messages.append({"role" : "assistant" , "content" : msg})
    st.chat_message("assistant").write(msg)

