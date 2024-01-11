from pydub import AudioSegment
from pytube import YouTube
import os ,argparse,sys


def mp4_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="mp4")
    audio.export(output_path, format="wav")

def download_youtube(args):
    count_down = 0
    count_url = 0
    with open(args.url_list , 'r') as file:
        sound_data = file.readlines()        
        species_count = dict()
        for sound_text in sound_data:
            if sound_text == '\n': continue
            count_url +=1
            try:
                species, url, emotion, action, method= sound_text.rstrip().split(" ")
                
                if species not in species_count.keys():
                    species_count[species] = 0
                else:
                    species_count[species] += 1 
                            
                if args.train:
                    save_path = os.path.join(species , 'train')
                elif args.valid:
                    save_path = os.path.join(species , 'valid')
                elif args.test:
                    save_path = os.path.join(species , 'test')
                    
                wav_filepath = emotion +'_'+ action + '_' + method + '_'+ "{:03d}".format(species_count[species]) +'.wav'
                
                if os.path.exists(os.path.join(save_path , wav_filepath)):
                    print('already downloaded' , url , wav_filepath)    
                    continue
            
                if not os.path.exists(save_path):
                    os.makedirs(save_path , exist_ok=True)
                yt = YouTube(url)         
                filepath = yt.streams.filter(only_audio = True).first().download(output_path = save_path)
                
                print(url , wav_filepath)
                mp4_to_wav(filepath , os.path.join(save_path , wav_filepath))
                os.remove(filepath)
                count_down+=1
            except Exception as e:
                # 그 외 모든 예외에 대한 처리
                print(f"An exception occurred: {e}")
                
                continue
    

    print('---------------------------------------')        
    print('Download Complete...')
    print('리스트에 존재하는 url의 개수: ' , count_url)
    print('다운로드한 wav 파일의 개수 : ' , count_down)
    print('경로에 존재하는 wav 파일의 개수 : ' , len(os.listdir(os.path.join(save_path))))
    print('다운로드된 파일 경로 : ' , save_path)
            
# 일단은 종에 따라 다를 수 있으니 구분해서 다운,
if __name__== "__main__":
    argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="extract .wav sample from youtube vidio data")
    
    parser.add_argument('--url_list' , type = str ,default= 'url_list.txt')
    parser.add_argument('--train' , type=bool ,default=False)
    parser.add_argument('--test' , type=bool ,default=False)
    parser.add_argument('--valid' , type=bool ,default=False)
    
    args = parser.parse_args(argv)    
    download_youtube(args)
    
    