import os
from collections import defaultdict, OrderedDict


DOG_EMO_CODES = {
    
    'hap' : ['hap' , 'happy' , 'happiness'],
    'sad': ['sad', 'sadness'],
    'ang': ['ang', 'angry', 'anger'],
    'fea': ['fea', 'fear'],
    'oth': ['oth', 'other', 'others']
}

class DOG_EMO_database():
    """
    Bark dataset is extracted from Youtube vedio file. 

    For each session, 
        eg. maltese/                            -> 강아지의 품종 
                |-- angry_normal_growl_000     -> 강아지의 감정 상태 , 행동, 울음소리 , 순번으로 저장된다.
                |-- happy_running_bark_001     
                |-- ...

    This function extract utterance filenames and labels for improvised sessions,
    organized into dictionary of {'speakerID':[(conversation_wavs,lab),(wavs,lab),...,(wavs,lab)]}

        > speakerID eg. 1M: Session 1, Male speaker
    """

    def __init__(self, database_dir, emot_map,
                        include_scripted=False): 
        
        self.database_dir = database_dir
        self.emot_map = emot_map
        self.sessions = ['train' , 'valid' , 'test' , 'data01', 'data02', 'data03', 'data04', 'data05', 'data06'
                            , 'data07', 'data08', 'data09', 'data10', 'data11' ,'data12']

        self.all_emo_classes = DOG_EMO_CODES.keys()
            
    def get_classes(self):

        classes={}
        for key,value in self.emot_map.items():
            if value in classes.keys():
                classes[value] += '+'+key
            else:
                classes[value] = key
        
        return classes

    def get_files(self):
        """
        Get all the required .wav file paths for each speaker and organized into
            dictionary:
                keys   -> speaker ID
                values -> list of (.wav filepath, label) tuples for corresponding speaker
        """
        emotions = self.emot_map.keys()
        dataset_dir = self.database_dir
        all_speaker_files = dict()
        total_num_files = 0
        for session_name in os.listdir(dataset_dir):
           
            if session_name not in self.sessions:
                continue
            wav_dir = os.path.join(dataset_dir, session_name)
        
            # Get a list of paths to all *.wav files
            wav_files = []
            
            for wav_name in os.listdir(wav_dir):
                #omit hidden folders
                if wav_name.startswith('.'):
                    continue
                #omit non .wav files ,ext 는 확장자 분리 
                name, ext = os.path.splitext(wav_name)
                if ext != ".wav":
                    continue
                #emotion label
                if len(name.split('_')) == 5:
                    emotion, action , size , species , _ = name.split('_')
                else:
                    emotion, action , size ,  _ = name.split('_')


                if emotion not in emotions:
                    print('설정하지 않은 감정값 발생:' , name)
                    continue
                label = self.emot_map[emotion]
                if len(name.split('_')) == 5:
                    wav_files.append((os.path.join(wav_dir, wav_name), label , species , action , size))
                else:
                    wav_files.append((os.path.join(wav_dir, wav_name), label, species , action , size))
            #Put speaker utterance and label paths into dictionary
            all_speaker_files[session_name] = wav_files

            total_num_files += len(wav_files)
        print(f'\nNUMBER OF FILES: {total_num_files}\n')
        return all_speaker_files

DOG_EMO_DATABASES = {'DOG_EMO': DOG_EMO_database}

    

