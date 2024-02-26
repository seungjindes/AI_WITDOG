import os 

import numpy as np 
import torch 
import  torch.nn.functional as f

from triton.preprocessing.1.features_extraction.features_util import AI_WITDOG.Model_Train.extract_features as extract_features
from features_extraction.database import DOG_EMO_DATABASES
from data_utils import SERDataset
from model.ser_model import Ser_Model
## model load


now_path = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(now_path, 'triton/AIdog/1')
MODEL_PATH = os.path.join(MODEL_DIR, 'AIdog.onnx')



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
        self.emot_map = {'hap': 0, 'sad':1, 'ang':2, 'fea':3}

        self.datapath =  "./test_data/samples"
        self.features = 'logmelspec'
        self.dataset = 'DOG_EMO'
        self.device = torch.device("cpu")
        self.model = Ser_Model()

    def make_onnx_file(self):
        
        database = DOG_EMO_DATABASES[self.dataset](self.datapath, emot_map=self.emot_map, 
                                include_scripted = False)
        speaker_files = database.get_files()
        data_features = extract_features(speaker_files, self.features, self.params , self.coc_params)
        ser_dataset = SERDataset(data_features)
 
        self.execute(self.model , ser_dataset , self.device)
                    
    def execute(self, model, ser_dataset, device):

        batch_size = 4
        test_dataset = ser_dataset.get_test_dataset()
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)        
        model.eval()

        test_batch = next(iter(test_loader))
 
        test_data_coc_batch = test_batch['seg_coc'].to(device)
        test_data_spec_batch = test_batch['seg_spec'].to(device)
        test_data_mfcc_batch = test_batch['seg_mfcc'].to(device)
        test_data_audio_batch = test_batch['seg_audio'].to(device)
        print("ONNX file create...")
        torch.onnx.export(model, (test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch, test_data_coc_batch) , MODEL_PATH, verbose=True)
        print("Complete..!")   
   


aidog = AIDOG()
aidog.make_onnx_file()