import torch
import numpy as np 
from features_extraction.database import DOG_EMO_DATABASES
from features_extraction.features_util import extract_features
from data_utils import SERDataset
import triton_python_backend_utils as pb_utils 


EMO_MAP = {'hap': 0, 'sad':1, 'ang':2, 'fea':3}

class TritonPythonModel:
    def initialize(self ,args):

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
        
        self.batch_size = 1
        print("Initialized...")


    def execute(self, requests):

        responses = []
        for request in requests:
            ## request is datapath
            raw_audio_file = pb_utils.get_input_tensor_by_name(request, "INPUT_AUDIO").as_numpy()
            database = DOG_EMO_DATABASES[self.dataset](request, emot_map=EMO_MAP, 
                                            include_scripted = False)
            
            speaker_files = database.get_files()
            data_features = extract_features(speaker_files, self.features, self.params , self.coc_params)
            ser_dataset = SERDataset(data_features)
            #batch size 유동으로 가능한것 같은데 
            test_dataset = ser_dataset.get_test_dataset()
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False)       

            test_batch = next(iter(test_loader))

            test_data_coc_batch = test_batch['seg_coc']
            test_data_spec_batch = test_batch['seg_spec']
            test_data_mfcc_batch = test_batch['seg_mfcc']
            test_data_audio_batch = test_batch['seg_audio']

            input_spec_batch = pb_utils.Tensor("INPUT_SPEC" , test_data_spec_batch)
            input_mfcc_batch = pb_utils.Tensor("INPUT_MFCC" , test_data_mfcc_batch)
            input_audio_batch = pb_utils.Tensor("INPUT_AUDIO" , test_data_audio_batch)
            input_coc_batch = pb_utils.Tensor("INPUT_COC" , test_data_coc_batch)


            response = pb_utils.inferenceResponse(
                output_tensor = [input_spec_batch , input_mfcc_batch,
                                 input_audio_batch, input_coc_batch]
            )
            responses.append(response)

        return responses

