import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict
from tqdm import tqdm
from pysndfx import  AudioEffectsChain
import random
from transformers import BertTokenizer, BertModel, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, AutoTokenizer

import pycochleagram
import pycochleagram.cochleagram as pycoc

from  features_extraction.conven_feature import get_conven_features


def extract_features(speaker_files, features, params , coc_params):

    # data_mfcc = list()
    #print('speaker_id ' , speaker_id)
    for wav_path, emotion in speaker_files:
        data_tot, segs, data_mfcc, data_audio,data_coc ,data_conven = list(), list(), list(), list(), list() ,list()
        x, sr = librosa.load(wav_path, sr=16000)  
        duration = 3
        segments = extract_segment(x , duration , sr = sr)
        # Extract required features into (C,F,T)
        
        for x in segments: 
            if len(x) < sr * duration:
                x = padding(x , sr*duration)
                
            features_data = GET_FEATURES[features](x, sr, params)
            
            hop_length = 160 # hop_length smaller, seq_len larger
            # f0 = librosa.feature.zero_crossing_rate(x, hop_length=hop_length).T # (seq_len, 1)
            # cqt = librosa.feature.chroma_cqt(y=x, sr=sr, n_chroma=24, bins_per_octave=72, hop_length=hop_length).T # (seq_len, 12)
            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T # (seq_len, 20)
            
            #coc shape : (1 , 128, 48000) , batch, channel, time 
            coc = extract_erb_cochlegram(x , sr , coc_params)
            coc = np.expand_dims(coc, axis = 0)
            
            conven_features = get_conven_features(x)

            print(conven_features.shape, 'shape')
            raise Exception('extract featiure')


            conven_features = np.expand_dims(conven_features , axis =0)
            # Segment features into (N,C,F,T)
            features_segmented = segment_nd_features(x, mfcc, features_data, emotion)
            #(num_segs, data_tot, mfcc_tot, audio_tot)
            #Collect all the segments
            data_tot.append(features_segmented[1])
            segs.append(features_segmented[0])
            data_mfcc.append(features_segmented[2])
            data_audio.append(features_segmented[3])
            data_coc.append(coc)
            data_conven.append(conven_features)
                
        
        # Post process
        data_tot = np.vstack(data_tot).astype(np.float32)
        data_mfcc = np.vstack(data_mfcc).astype(np.float32)
        data_audio = np.vstack(data_audio).astype(np.float32)
        data_coc = np.vstack(data_coc).astype(np.float32)
        data_conven = np.vstack(data_conven).astype(np.float32)
        segs = np.asarray(segs, dtype=np.int8)
        
        

        #Put into speaker features dff
        audio_features = defaultdict()
        audio_features["seg_spec"] = data_tot
        audio_features["seg_num"] = segs
        audio_features["seg_mfcc"] = data_mfcc
        audio_features["seg_audio"] = data_audio
        audio_features["seg_coc"] = data_coc
        audio_features["seg_conven"] = data_conven
        
    return audio_features

def padding(feature, MAX_LEN):
    """
    mode: 
        zero: padding with 0
        normal: padding with normal distribution
    location: front / back
    """
    padding_mode  = 'normal'
    padding_location = 'back'

    length = feature.shape[0]
    if length >= MAX_LEN:
        return feature[:MAX_LEN, :]
        
    if padding_mode == "zeros":
        pad = np.zeros([MAX_LEN - length])
    elif padding_mode == "normal":
        mean, std = feature.mean(), feature.std()*0.05
        pad = np.random.normal(mean, std, (MAX_LEN-length))
    feature = np.concatenate([pad, feature], axis=0) if(padding_location == "front") else \
              np.concatenate((feature, pad), axis=0)
    return feature
                  
def paddingSequence(sequences):
    if len(sequences) == 0:
        return sequences
    feature_dim = sequences[0].shape[-1]
    lens = [s.shape[0] for s in sequences]
    # confirm length using (mean + std)
    final_length = int(np.mean(lens) + 3 * np.std(lens))
    # padding sequences to final_length
    final_sequence = np.zeros([len(sequences), final_length, feature_dim])
    for i, s in enumerate(sequences):
        final_sequence[i] = padding(s, final_length)

    return final_sequence
        
def extract_erb_cochlegram(x ,sr , params):
    """
    pycochleagram.cochleagram.cochleagram 
        input ->    signal, sr, n, low_lim, hi_lim, sample_factor
                    padding_size=None
                    downsample=None
                    nonlinearity=None : {None, 'db', 'power', callable}
                    fft_mode='auto'
                    ret_mode='envs' 
                    strict=True, **kwargs
                    
        output ->   out: The output, depending on the value of ret_mode. If the ret_mode , return type : array
    """
    
    n_filter, low_lim, hi_lim, sample_factor , nonlinearity =  params['n_filter'],params['low_lim'],params['hi_lim'],\
                                        params['sample_factor'] , params['nonlinearity']
                                    
    cochleagram = pycoc.cochleagram(x, sr, n_filter, low_lim, hi_lim, sample_factor ,nonlinearity=nonlinearity)
    return cochleagram


def extract_logspec(x, sr, params):
    
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    nfreq         = params['nfreq']

    #calculate stft
    spec = np.abs(librosa.stft(x, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)
    
    #extract the required frequency bins
    spec = spec[:nfreq]
    
    #Shape into (C, F, T), C = 1
    spec = np.expand_dims(spec,0)

    return spec




def extract_logmelspec(x, sr, params):
 
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    n_mels        = params['nmel']
    

    #calculate stft

    melspec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels,
                                        n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window)

    logmelspec =  librosa.power_to_db(melspec, ref=np.max)

    # Expand to (C, F, T), C = 3
    logmelspec =  np.expand_dims(logmelspec, 0)
    
    return logmelspec




def extract_logdeltaspec(x, sr, params):
      
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    n_freq        = params['nfreq']
    
    #calculate stft
    logspec = extract_logspec(x, sr, params) # (C, F, T)

    logdeltaspec = librosa.feature.delta(logspec.squeeze(0))
    logdelta2spec = librosa.feature.delta(logspec.squeeze(0), order=2)
    
    #Arrange into (C, F, T), C = 3
    logdeltaspec = np.expand_dims(logdeltaspec, axis=0)
    logdelta2spec = np.expand_dims(logdelta2spec, axis=0)
    logspec = np.concatenate((logspec, logdeltaspec, logdelta2spec), axis=0)
    
    return logspec


def segment_nd_features(input_values, mfcc, data, emotion):

    segment_size = 300
    segment_size_wav = segment_size * 160
    # Transpose data to C, T, F
    
    data = data.transpose(0,2,1)
    time = data.shape[1]
    time_wav = input_values.shape[0]
    nch = data.shape[0]
    start, end = 0, segment_size
    start_wav, end_wav = 0, segment_size_wav
    num_segs = math.ceil(time / segment_size) # number of segments of each utterance
    if num_segs > 1:
       num_segs = num_segs - 1
    mfcc_tot = []
    audio_tot = []
    data_tot = []
    sf = 0
    # 모델에 사용하기전 전처리 과정 

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
  
    
    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - segment_size)
        if end_wav > time_wav:
            end_wav = time_wav
            start_wav = max(0, end_wav - segment_size_wav)
        """
        if end-start < 100:
            num_segs -= 1
            print('truncated')
            break
        """
        # Do padding
        mfcc_pad = np.pad(
                mfcc[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
        
        audio_pad = np.pad(input_values[start_wav:end_wav], ((segment_size_wav - (end_wav - start_wav)), (0)), mode="constant")
  
        data_pad = []
        for c in range(nch):
            data_ch = data[c]
            data_ch = np.pad(
                data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
                #data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant",
                #constant_values=((-80,-80),(-80,-80)))
            data_pad.append(data_ch)

        
        #audio_wav = processor(audio_wav.cpu(), sampling_rate=16000, return_tensors="pt").input_values# [1, batch, 48000] 
        #audio_wav = audio_wav.permute(1, 2, 0) # [batch, 48000, 1] 
        #audio_wav = audio_wav.reshape(audio_wav.shape[0],-1) # [batch, 48000] 
        
        
        data_pad = np.array(data_pad)
        
        # Stack
        mfcc_tot.append(mfcc_pad)
        data_tot.append(data_pad)

        audio_pad_np = np.array(audio_pad)
        audio_pad_pt = processor(audio_pad_np, sampling_rate=16000, return_tensors="pt").input_values
        audio_pad_pt = audio_pad_pt.view(-1)
        audio_pad_pt_np = audio_pad_pt.cpu().detach().numpy()
        audio_tot.append(audio_pad_pt_np)
        
        # Update variables
        start = end
        end = min(time, end + segment_size)
        start_wav = end_wav
        end_wav = min(time_wav, end_wav + segment_size_wav)      
    
    mfcc_tot = np.stack(mfcc_tot)
    data_tot = np.stack(data_tot)
    audio_tot = np.stack(audio_tot)

    
    #Transpose output to N,C,F,T
    data_tot = data_tot.transpose(0,1,3,2)

    return (num_segs, data_tot, mfcc_tot, audio_tot)

def extract_segment(x , duration =3  ,sr = 16000):
    avg_amplitude = np.mean(np.abs(x))*10
    above_avg = np.where(np.abs(x) > avg_amplitude)[0]
    
    segments = []
    end_sample = 0
    for start_sample in above_avg:
        if end_sample < start_sample:
            end_sample = start_sample + duration * sr
            if end_sample < len(x):
                segments.append(x[start_sample:end_sample])
            else:
                #padding
                segments.append(x[start_sample:len(x)])
                return segments
    return segments


#Feature extraction function map
GET_FEATURES = {'logspec': extract_logspec,
                'logmelspec': extract_logmelspec,
                'logdeltaspec': extract_logdeltaspec
                }

