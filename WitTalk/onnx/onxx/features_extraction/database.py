import os
from collections import defaultdict, OrderedDict
1
DOG_EMO_CODES = {
    
    'hap' : ['hap' , 'happy' , 'happiness'],
    'sad': ['sad', 'sadness'],
    'ang': ['ang', 'angry', 'anger'],
    'fea': ['fea', 'fear'],
    'oth': ['oth', 'other', 'others']
}

class DOG_EMO_database():
    def __init__(self, database_dir, emot_map = {'hap': 0, 'sad':1, 'ang':2, 'fea':3},
                        include_scripted=False): 
        
        self.database_dir = database_dir
        self.emot_map = emot_map
        #IEMOCAP available emotion classes 
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

        total_num_files = 0
        #시간으로 명명된 폴더 30초 단위로 잘라서 사용 
        wav_files = []
        
        for wav_name in os.listdir(dataset_dir):
            #omit hidden folders
            if wav_name.startswith('.'):
                continue
            #omit non .wav files ,ext 는 확장자 분리 
            name, ext = os.path.splitext(wav_name)
            if ext != ".wav":
                continue
            #emotion label
   
            
            wav_files.append((os.path.join(dataset_dir, wav_name), None))
            
        #Put speaker utterance and label paths into dictionary
       
        
        total_num_files = len(wav_files)
        
        #list
        return wav_files

DOG_EMO_DATABASES = {'DOG_EMO': DOG_EMO_database}

    

