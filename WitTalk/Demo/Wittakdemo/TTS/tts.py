import os 
import torch 
from TTS.openvoice import se_extractor
from TTS.openvoice.api import ToneColorConverter
import librosa

from openai import OpenAI
from dotenv import load_dotenv



ckpt_converter = './TTS/checkpoints_v2/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

load_dotenv()

client = OpenAI(api_key = "")

# text는 사용자의 입력 
def voice_generator(text):
    response = client.audio.speech.create(
        model = "tts-1-hd",
        voice = "nova",
        input = text
    )

    response.stream_to_file(f"./TTS/resources/tts_voice_sample.wav")
    
    # 변형이될 오디오 원본 
    base_spearker = f"./TTS/resources/tts_voice_sample.wav"

    target_speaker = "/home/sm27/AI_WITDOG/Wittakdemo/TTS/resources/demo_speaker0.mp3"
    target_se, audio_name = se_extractor.get_se(target_speaker, tone_color_converter, vad=True)


    source_se = torch.load('./TTS/resources/source_se.pth')
    #target_se = torch.load('./TTS/resources/target_se.pth')
    
    save_path = f"{output_dir}/my_voicesmple_output.wav"

    encode_message = "@Smartware"
    tone_color_converter.convert(
        audio_src_path = base_spearker,
        src_se = source_se,
        tgt_se=target_se, 
        output_path=save_path,
            message=encode_message)

    return base_spearker , save_path
    
if __name__ == "__main__":
    print("openai key is available") 



    


