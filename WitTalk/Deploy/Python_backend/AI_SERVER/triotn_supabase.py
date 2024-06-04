import supabase
import datetime
import time
import json
import librosa
import os

# Supabase 연결 정보 설정
SUPABASE_URL = "https://fnjsdxnejydzzlievpie.supabase.co/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZuanNkeG5lanlkenpsaWV2cGllIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwMzIxMzAyOCwiZXhwIjoyMDE4Nzg5MDI4fQ.DBcvEFlnsh3jlLLDWNAE8BIgYaLAhO2sMBwTFvVx23c"
supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_KEY)


def execute_server():
    while True:
        # get fist data
        response = supabase_client.table('AI_QUEUE').select("id").execute()
        try:
            json_data = response.json()
            data = json.loads(json_data)
            response_datas = data["data"]
            if response_datas:
                print('start AIdog Inference Server')
                os.system(f'python AIdog_server_test.py')
            else: pass        
        except:
            print("Supabase query failed")
            raise Exception
        
        print('No data.')
        time.sleep(1)  


def test_upload(k):
    wav_path = '/home/sm27/AI_WITDOG/Inference_Server/AI_PC/data_pipeline/data/24-01-12-14-11-00.wav'
    x, sr = librosa.load(wav_path, sr=16000)  
    x = x[:len(x)//3]
    x_list = x.tolist()
    for i in range(k):
        print(i)
        data = supabase_client.table('AI_QUEUE').insert({'audio_data' : x_list}).execute()
        data = supabase_client.table('AI_QUEUE').update({'SR' : sr}).eq('id', i).execute()
        data = supabase_client.table('AI_QUEUE').update({'pet_id' : '056'}).eq('id', i).execute()


if __name__ == "__main__":
    execute_server()
    