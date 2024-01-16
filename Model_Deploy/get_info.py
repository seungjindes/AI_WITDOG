
import torch
import numpy as np
from Model.ser_model import Ser_Model
from features_extraction.features_util import extract_features
from features_extraction.database import DOG_EMO_DATABASES
import argparse


import torch.nn.functional as f
from data_utils import SERDataset 

import os
from supabase import create_client, Client

# 3초마다 오디오 파일을 읽고나서 출력값으로 감정 레이블을 출력
class AIDOG:
    def __init__(self):
        
        self.params={'window'   : 'hamming',
                'win_length'    : 40,
                'hop_length'    : 10,
                'ndft'          : 800,
                'nfreq'         : 200,
                'nmel'          : 128,
                'segment_size'  : 300
                }
        self.coc_params={
                    'n_filter': 126,
                    'low_lim': 30,
                    'hi_lim': 5000,
                    'sample_factor':1,
                    'nonlinearity': 'power'
                    }

        self.dataset  = 'DOG_EMO'
        self.features = 'logmelspec'
        
        
        self.emot_map = {'hap': 0, 'sad':1, 'ang':2, 'fea':3}
        # url: str = os.environ.get("SUPABASE_URL")
        # key: str = os.environ.get("SUPABASE_KEY")

        #self.supabase = create_client(url, key)

        self.args = {
            'description' : "Deploy Bark2text Model. Load trained model and send to supabase server",
            'gpu' : False,
            'model_name' : "best_val_model",
            'data_path' : "./test_data/input_data_from_local/"
        }

        #select device
        if self.args['gpu'] == True:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(self.device)
        self.model = Ser_Model().to(torch.device('cpu')) 
        self.model_path = './asset/' + self.args['model_name']+ '.pth'
        self.model.load_state_dict(torch.load(self.model_path), map_location=torch.device('cpu'))    

    def main(self):
        
        #반복문 
        #data_seq = upload_wav_to_supabase(file_path)
        #-> 30초단위로 폴더를 생성해서 
        dataset_dir = './test_data/input_data_from_local'

        database = DOG_EMO_DATABASES[self.dataset](dataset_dir, emot_map=self.emot_map, 
                                include_scripted = False)
        speaker_files = database.get_files()
        
        data_features = extract_features(speaker_files, self.features, self.params , self.coc_params)
        ser_dataset = SERDataset(data_features)
        
        result = self.test(self.model , ser_dataset , self.device)
        
        return result
            
    def test(self, model, test_dataset, device):


        test_preds_segs = []
        batch_size = 4
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)        
        model.eval()

        # for i, test_batch in enumerate(test_loader):

        for test_batch in test_loader:  
            # Send data to correct device
            test_data_coc_batch = test_batch['seg_coc'].to(device)
            test_data_spec_batch = test_batch['seg_spec'].to(device)
            test_data_mfcc_batch = test_batch['seg_mfcc'].to(device)
            test_data_audio_batch = test_batch['seg_audio'].to(device)


            # Forward
            test_outputs = model(test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch , test_data_coc_batch)
            test_preds_segs.append(f.log_softmax(test_outputs['M'], dim=1).cpu())


        # Accumulate results for val data
        test_preds_segs = np.vstack(test_preds_segs)
        test_preds = test_dataset.get_preds(test_preds_segs)
        
        return test_preds
    
    
if __name__ == "__main__":
    aidog = AIDOG()
    
    