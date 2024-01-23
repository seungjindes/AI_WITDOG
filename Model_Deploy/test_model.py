import os 

import numpy as np 
import torch 
import  torch.nn.functional as f

from features_extraction.features_util import extract_features
from features_extraction.database import DOG_EMO_DATABASES
from data_utils import SERDataset
from model.ser_model import Ser_Model
## model load
now_path = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(now_path, 'triton/AIdog/1')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_val_model.pth')

## customize here for using your image 
# default: zero array input 


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

        self.args = {
            'description' : "Deploy Bark2text Model. Load trained model and send to supabase server",
            #'gpu' : False,
            'model_name' : "best_val_model",
            'data_path' : "./test_data/input_data_from_local/",
        }

        self.features = 'logmelspec'
        self.dataset = 'DOG_EMO'

        # #select device
        # if self.args['gpu'] == True:
        #     self.device = torch.device("cuda")
        # else:
        #     self.device = torch.device("cpu")

        # self.model = torch.jit.load(MODEL_PATH)
        self.device = torch.device("cpu")
        self.model = Ser_Model()

    def infer(self):
        #supabase로 교체예정 , 받아와야하는데 시간이 얼마나 걸릴지 모르겠음
        dataset_dir = './test_data/input_data_from_local'

        database = DOG_EMO_DATABASES[self.dataset](dataset_dir, emot_map=self.emot_map, 
                                include_scripted = False)
        speaker_files = database.get_files()
        
        data_features = extract_features(speaker_files, self.features, self.params , self.coc_params)
        ser_dataset = SERDataset(data_features)
        result = self.test(self.model , ser_dataset , self.device)
        
        return result
            
    def test(self, model, ser_dataset, device):


        test_preds_segs = []
        batch_size = 4
        


        test_dataset = ser_dataset.get_test_dataset()

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

            # 이거 1영상당 2배치밖에 안나온다 수정 필요!!
            print(test_data_spec_batch.shape)
            print(test_data_mfcc_batch.shape)
            print(test_data_audio_batch.shape)
            print(test_data_coc_batch.shape)  


            print(type(test_data_spec_batch))
            print(type(test_data_mfcc_batch))
            print(test_data_audio_batch.dtype)
            print(test_data_coc_batch.dtype)  
   

            test_outputs_atttr = model(test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch, test_data_coc_batch)
            print(test_outputs_atttr.shape)
            print(test_outputs_atttr.dtype)
            raise("")

            #test_preds_segs.append(f.log_softmax(test_outputs_atttr, dim=1).cpu())
    
            #onnx_path = "./simple_model.onnx"
            #torch.onnx.export(model, (test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch, test_data_coc_batch) , onnx_path, verbose=True)

 
        return 
        # Accumulate results for val data
        test_preds_segs = np.vstack(test_preds_segs)
        test_preds = test_dataset.get_preds(test_preds_segs)
        
        return test_preds
    




## inference 
    

aidog = AIDOG()
result = aidog.infer()
# try:
#     aidog = AIDOG()
#     result = aidog.infer()
    
# except Exception as e: 
#     print('error: ', e)
# finally: 
#     print('test finish')