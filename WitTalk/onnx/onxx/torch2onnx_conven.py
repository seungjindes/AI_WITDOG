import os 

import numpy as np 
import torch 
import  torch.nn.functional as f

from data_utils import SERDataset
from model.ser_model_conven import convential_model
## model load
import onnx 
from onnx import shape_inference

import pickle

now_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(now_path, 'AIdog_conven.onnx')

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
        self.emot_map = {'Hap': 0, 'Sad':1, 'Ang':2, 'Howling':3}

        self.datapath =  "./samples"
        self.weightpath= "./last_model.pth"
        self.features = 'logmelspec'
        self.dataset = 'DOG_EMO'
        self.device = torch.device("cpu")
        self.model = convential_model()

    def make_onnx_file(self):
        # database = DOG_EMO_DATABASES[self.dataset](self.datapath, emot_map=self.emot_map, 
        #                         include_scripted = False)
        # speaker_files = database.get_files()
        # data_features = extract_features(speaker_files, self.features, self.params , self.coc_params)
        with open("./samples/valid_samples.pkl" ,"rb") as fin:
            data_features = pickle.load(fin)

        ser_dataset = SERDataset(data_features)
        weights = torch.load(self.weightpath)
        self.model.load_state_dict(weights)
        self.model.eval()

        self.execute(self.model , ser_dataset , self.device)
                    
    def execute(self, model, ser_dataset, device):

        batch_size = 1
        test_dataset = ser_dataset.get_test_dataset()


        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)        
        model.eval()
        test_batch = next(iter(test_loader))
 
        test_data_coc_batch = test_batch['seg_coc'].to(device)
        test_data_spec_batch = test_batch['seg_spec'].to(device)
        test_data_mfcc_batch = test_batch['seg_mfcc'].to(device)
        test_data_audio_batch = test_batch['seg_audio'].to(device)
        test_data_conven_batch = test_batch['seg_conven'].to(device)

        print('### Data Shape###')
        print(
            f"""
                seg_coc : {test_data_coc_batch.shape}\n
                seg_spec : {test_data_spec_batch.shape}\n
                seg_mfcc : {test_data_mfcc_batch.shape}\n
                seg_audio : {test_data_audio_batch.shape}\n
                seg_conven_fea : {test_data_conven_batch.shape}\n   
            """
            )
        
        print("ONNX file create...")

        torch.onnx.export(model, (test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch, test_data_coc_batch , test_data_conven_batch),
                          MODEL_PATH , verbose=True , input_names=['INPUT_SPEC' , 'INPUT_MFCC' , 'INPUT_AUDIO' , 'INPUT_COC' , 'INPUT_CONVEN'],
                          output_names=['result'] ,export_params= True)

        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(MODEL_PATH)) , MODEL_PATH)

        print("check onnx file..")
        onnx_model = onnx.load(MODEL_PATH)
        onnx.checker.check_model(onnx_model)
        print("Complete..!")   
   


aidog = AIDOG()
aidog.make_onnx_file()