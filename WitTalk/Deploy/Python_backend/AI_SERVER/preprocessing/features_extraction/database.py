import os
from collections import defaultdict, OrderedDict



class DOG_EMO_database():
    def __init__(self, database_dir, emot_map ,
                        include_scripted=False): 
        
        self.database_dir = database_dir
        self.emot_map = emot_map
        
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
        wav_files = []
        
        for wav_name in os.listdir(dataset_dir):
            #omit hidden folders
            if wav_name.startswith('.'):
                continue
            name, ext = os.path.splitext(wav_name)
            if ext != ".wav":
                continue

            (species, emotion , sound , target , age , num , index) = name.split('_')
            
            wav_files.append((os.path.join(dataset_dir, wav_name), sound, target , age))    
        total_num_files = len(wav_files)
        return wav_files

DOG_EMO_DATABASES = {'DOG_EMO': DOG_EMO_database}

    

