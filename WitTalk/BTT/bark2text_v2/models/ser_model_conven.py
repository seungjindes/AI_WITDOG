import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import  Wav2Vec2Model
from models.ser_spec import SER_AlexNet , SER_EfficientNet


class moel_coc(nn.Module):
    def __init__(self):
        super(moel_coc, self).__init__()
        self.cnn1 =nn.Conv1d(128, 64 , kernel_size= 3, stride = 3)
        self.cnn2 = nn.Conv1d(64 , 32 , kernel_size=2 , stride = 2)
        self.cnn3 = nn.Conv1d(32 , 16 , kernel_size=2 , stride = 2)
        self.lstm = nn.LSTM(input_size = 16, hidden_size = 8 , num_layers=2 , batch_first=True , dropout= 0.5 , bidirectional=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64000 , 128)
        
    def forward(self , x):
        x =  F.relu(self.cnn1(x))
        x =  F.relu(self.cnn2(x))
        x =  F.relu(self.cnn3(x))
        x = x.transpose(2,1)
        x , _ = self.lstm(x)
        x =self.flatten(x)
        x = self.linear(x)

        return x

class model_convential_feature(nn.Module):
    def __init__(self):
        super(model_convential_feature ,self).__init__()
        self.flatten = nn.Flatten()
        #68 * 7
        self.linear1 = nn.Linear(408, 128)
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        return x # shape is 128
    
    
class model_convential_feature_conv(nn.Module):
    def __init__(self):
        super(model_convential_feature_conv , self).__init__()
        self.conv1 = nn.Conv1d(7 , 64 , 3 , stride= 2)
        #self.conv2 = nn.Conv1d(32 , 64 , 3 , stride= 3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2112 , 1024)
        self.linear2 = nn.Linear(1024 , 128)
    def forward(self , x):
        x = x.transpose(2,1)
        x = F.relu(self.conv1(x) ,(2,1))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
class model_convential_feature_conv_inverse(nn.Module):
    def __init__(self):
        super(model_convential_feature_conv , self).__init__()
        self.conv1 = nn.Conv1d(68 , 128 , 3 , stride= 2)
        #self.conv2 = nn.Conv1d(32 , 64 , 3 , stride= 3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256 , 128)
    def forward(self , x):
        x = F.relu(self.conv1(x) ,(2,1))
        x = self.flatten(x)
        x = self.linear1(x)
        return x

class model_dog_info(nn.Module):
    def __init__(self):
        super(model_dog_info , self).__init__()
        self.linear = nn.Linear(13 , 64)
        self.linear2 = nn.Linear(64 , 128)
    def forward(self , x):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        return x
    

# __all__ = ['Ser_Model']
class convential_model(nn.Module):
    def __init__(self):
        super(convential_model, self).__init__()
        
        # CNN for Spectrogram
        self.alexnet_model = SER_AlexNet(num_classes=13, in_ch=3, pretrained=True)
        
        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(9216, 128) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        
        # LSTM for MFCC        
        self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True

        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        self.post_mfcc_layer = nn.Linear(153600, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        
        # LSTM for Cochelagram
        self.coc_model = moel_coc()
        self.post_coc_dropout = nn.Dropout(p=0.1)

        self.convential_model = model_convential_feature()
        self.post_convential_dropout = nn.Dropout(0.1)

        self.dog_info_model = model_dog_info()

        
        # Spectrogram + MFCC  
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(256, 149) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        self.post_multi_att_layer = nn.Linear(512 , 149) # 128 * 5 
                        
        # WAV2VEC 2.0
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model.config.ctc_zero_infinity = True

        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(768, 128) # 512 for 1 and 768 for 2
        
        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(384, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 13)
        
                                                                     
    def forward(self, audio_spec, audio_mfcc, audio_wav , audio_coc , convential_feature):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]

        #convential_feature have 7 dimension 
        # [0 ~ 1] husrt 
        # [2] num of outlier
        # [3] skew 
        # [4] modes
        # [5] kurt
        # [6] species - 68 *
        # [7] sound vector -
        # [8] integral vector - 
        ### integral vector 
        #   [0] -> bark to owner (0 or 1) 0 is false , 1 is true
        #   [1] -> bark to other (0 or 1)
        #   [2] -> old dog (0 or 1)
        #   [3] -> puppy  (0 or 1)
        #   [4] -> number of dog in data , single (0 or 1)
        #   [5] -> number of dog in data , group (0 or 1)


            
        spicies_feature = convential_feature[: , : , 6]# 1 x 32 dimnsion
        sound_feature = convential_feature[: , : , 7]
        integral_feature = convential_feature[: , : ,8]   


        convential_feature = convential_feature[: , : , :6]    


        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        
        # spectrogram - SER_CNN
        audio_spec, output_spec_t = self.alexnet_model(audio_spec) # [batch, 256, 6, 6], []
        audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1) # [batch, 256, 36]  
        
        # audio -- MFCC with BiLSTM
        audio_mfcc, _ = self.lstm_mfcc(audio_mfcc) # [batch, 300, 512]  
        
        audio_spec_ = torch.flatten(audio_spec, 1) # [batch, 9216]  
        audio_spec_d = self.post_spec_dropout(audio_spec_) # [batch, 9216]  
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False) # [batch, 128]  

        #+ audio_mfcc = self.att(audio_mfcc)
        audio_mfcc_ = torch.flatten(audio_mfcc, 1) # [batch, 153600]  
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_) # [batch, 153600]  
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False) # [batch, 128]  
  
        
        # coc_data [batch, 128]
        audio_coc_p = self.coc_model(audio_coc)
        #audio_coc_d = self.post_coc_dropout(audio_coc_p)
    
        audio_convential_p = self.convential_model(convential_feature)
        audio_convential_d = self.post_convential_dropout(audio_convential_p)

        
        #multi_f = torch.cat([audio_spec_p , audio_mfcc_p , audio_coc_p , audio_convential_d] , dim=-1)  #[batch, 512]
        multi_f = torch.cat([audio_spec_p , audio_mfcc_p , audio_coc_p , audio_convential_d] , dim= -1)


        multi_d = self.post_spec_dropout(multi_f)
        multi_p = F.relu(self.post_multi_att_layer(multi_d) , inplace=False)# [batch, 512]
        multi_p = multi_p.reshape(multi_p.shape[0] , 1, -1) #[batch , 1 , 149]        
        # wav2vec 2.0 
    
        
        audio_wav = self.wav2vec2_model(audio_wav).last_hidden_state # [batch, 149, 768] 
        audio_wav = torch.matmul(multi_p, audio_wav) # [batch, 1, 768] 
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 768] 
        #audio_wav = torch.mean(audio_wav, dim=1)
        
        audio_wav_d = self.post_wav_dropout(audio_wav) # [batch, 768] 
        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=False) # [batch, 768] 
    

        
        ## combine()
        audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384] 
            
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] 
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 


        # predict, owner  , [8,13] , [8,1]
        return output_att, integral_feature[0]
    


