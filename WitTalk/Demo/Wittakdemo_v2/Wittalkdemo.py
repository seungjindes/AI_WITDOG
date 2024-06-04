from openai import OpenAI
import streamlit as st
import time
import json 
from TTS.tts import voice_generator
import librosa

from tempfile import NamedTemporaryFile
from utils import DataPipeline , get_emotion_using_triton

# #개요
# WitTalk은 반려견과의 소통을 위한 앱인 WitDog의 일부 서비스 입니다. 
# 해당 페이지에서 BarkToText(BTT)는 반려견의 음성과 정보를 통해 반려견의 현재 상태를 분석하고 감정이 어떤지를 추출할 수 있는 기술로 정의합니다.
# 데모에서는 BTT의 실제 적용 예시를 구현하기 위해 챗봇 서버 제공 서비스인 Straemlt , ChatGPT API 를 위한 OpenAI ,
# 사용자가 보낸 문자를 단말기에서 voice 로 변환하기 위한 TTS를 제공하기 위해, OpenVoice의 오픈소스를 사용하였습니다.

# ## 플로우 
# 진행은 아래와 같이 이루어 집니다.
# 나중에 다시 쓸게여 ㅎ

## 진행상황
# 모델을 conven feature를 사용해서 적용하려했으나 extract_feature부터 data_utils까지 바꿔줘야해서 잠시보류
# 학습을 좀더 시키고 나서 데모를 완성시키는 게 좋을 듯 최선으로, 바꿀게 너무 많음

emot_map = {
        '관심': 0,
        '행복': 1,
        '흥분(화남)': 1,
        '요구': 2,
        '호기심': 2,
        '예민': 3,
        '경계': 4,
        '화남': 5,
        '슬픔': 6,
        '공포': 7,
        '반가움': 8,
        '하울링(행복)': 9,
        '하울링': 10,
        '외로움': 11,
        '신남': 12,
    }
 

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
        file_name = upload_dogsound.name
        (species, emotion , sound , target , age , num , index)  =  file_name.split('_')

        temp.write(upload_dogsound.getvalue())
        temp.seek(0)
        temp_dogsound  ,sr = librosa.load(temp  ,sr = 16000)
        #st.audio(temp_dogsound , sample_rate = 16000)
        input_data= pipleline.execute(temp_dogsound , sr ,species ,emotion, sound , target , age , num)
        DOG_EMO = get_emotion_using_triton(input_data , target)



with st.sidebar:

    # OpenAI Key :
    
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot password", type="password")
    client = OpenAI(api_key=openai_api_key)
    thread_id = json_data[USER]['thread_key']


    #action = st.text_input("반려견의 현재 상황" , key = "action")

    sample_sound , sr= librosa.load('./TTS/resources/example_reference2.mp3' , sr = 44100)
    st.audio(sample_sound ,format = "audio/mp3" , sample_rate = 44100)

    st.caption("위 오디오는 사용자 목소리 샘플입니다.")


    # check box를 통해 반려견의 현재 감정을 보여줌 
    st.write('반려견 현재 감정')

    #def display_emotions(DOG_EMO):

    interest = st.checkbox('관심' , value = (DOG_EMO == '관심') , disabled = True)
    happy = st.checkbox('행복해', value=(DOG_EMO == '행복'), disabled=True)
    very_angry = st.checkbox('매우화남', value=(DOG_EMO == '흥분(화남)'), disabled=True)
    sad = st.checkbox('슬퍼', value=(DOG_EMO == '슬픔'), disabled=True)
    angry = st.checkbox('화난다', value=(DOG_EMO == '화남'), disabled=True)
    fear = st.checkbox('두렵다', value=(DOG_EMO == '공포'), disabled=True)
    curious = st.checkbox('호기심', value=(DOG_EMO == '호기심'), disabled=True)
    demand = st.checkbox('요구', value=(DOG_EMO == '요구'), disabled=True)
    sensitive = st.checkbox('예민', value=(DOG_EMO == '예민'), disabled=True)
    alert = st.checkbox('경계', value=(DOG_EMO == '경계'), disabled=True)
    welcoming = st.checkbox('반가움', value=(DOG_EMO == '반가움'), disabled=True)
    howling_happy = st.checkbox('하울링(행복)', value=(DOG_EMO == '하울링(행복)'), disabled=True)
    howling = st.checkbox('하울링', value=(DOG_EMO == '하울링'), disabled=True)
    lonely = st.checkbox('외로움', value=(DOG_EMO == '외로움'), disabled=True)
    excited = st.checkbox('신남', value=(DOG_EMO == '신남'), disabled=True)
    unknown = st.checkbox('불명확' , value = False , disabled = True)


    if interest:
        emotion = "관심받고싶어"
    elif happy:
        emotion = "행복함"
    elif very_angry:
        emotion = "매우화남"
    elif sad:
        emotion = "슬픔"
    elif angry:
        emotion = "화남"
    elif fear:
        emotion = "두렵다"
    elif curious:
        emotion = "호기심"
    elif demand:
        emotion = "요구"
    elif sensitive:
        emotion = "예민"
    elif alert:
        emotion = "경계"
    elif welcoming:
        emotion = "반가움"
    elif howling_happy:
        emotion = "하울링(행복)"
    elif howling:
        emotion = "하울링"
    elif lonely:
        emotion = "외로움"
    elif excited:
        emotion = "신남"

    #    return emotion

    # Example usag
    #emotion = display_emotions(DOG_EMO)
    #st.write(f"Detected emotion: {emotion}")

        
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

