from pydub import AudioSegment
from pytube import YouTube
import os ,argparse,sys


def mp4_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="mp4")
    audio.export(output_path, format="wav")

def download_youtube(args):
    with open(args.url_list , 'r') as file:
        sound_data = file.readlines()        
        species_count = dict()
        for sound_text in sound_data:
            if sound_text == '\n' : continue
            try:
                species, url, emotion, action, method= sound_text.rstrip().split(" ")
                if species not in species_count.keys():
                    species_count[species] = 0
                else:
                    species_count[species] += 1 
                
                if args.train:
                    save_path = os.path.join(species , 'train')
                elif args.val:
                    save_path = os.path.join(species , 'val')
                elif args.test:
                    save_path = os.path.join(species , 'test')

                            
                if not os.path.exists(save_path):
                    os.makedirs(save_path , exist_ok=True)
                yt = YouTube(url)
                filepath = yt.streams.filter(only_audio = True).first().download(output_path = save_path)
                wav_filepath = emotion +'_'+ action + '_' + method + '_'+ "{:03d}".format(species_count[species]) +'.wav'
                print(url , wav_filepath)
                mp4_to_wav(filepath , os.path.join(save_path , wav_filepath))
                os.remove(filepath)
            except:
                continue

            
    print('Download Complete...')
            
# 일단은 종에 따라 다를 수 있으니 구분해서 다운,
if __name__== "__main__":
    argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="extract .wav sample from youtube vidio data")
    
    parser.add_argument('--url_list' , type = str ,default= 'url_list.txt')
    parser.add_argument('--train' , type=bool ,default=False)
    parser.add_argument('--test' , type=bool ,default=False)
    parser.add_argument('--val' , type=bool ,default=False)
    
    args = parser.parse_args(argv)    
    download_youtube(args)
    
    