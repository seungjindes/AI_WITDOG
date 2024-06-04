import os
import sys
import argparse
import numpy as np
import pickle
from features_util import extract_features
from collections import Counter
import pandas as pd
from database import DOG_EMO_DATABASES
import random

import warnings
warnings.filterwarnings('ignore')


def main(args):
    
    #Get spectrogram parameters
    params={'window'        : args.window,
            'win_length'    : args.win_length,
            'hop_length'    : args.hop_length,
            'ndft'          : args.ndft,
            'nfreq'         : args.nfreq,
            'nmel'          : args.nmel,
            'segment_size'  : args.segment_size,
            'mixnoise'      : args.mixnoise
            }
    coc_params={
                'n_filter': args.n_filter,
                'low_lim': args.low_lim,
                'hi_lim': args.hi_lim,
                'sample_factor': args.sample_factor,
                'nonlinearity': args.nonlinearity
                }
    
    dataset  = args.dataset
    features = args.features
    dataset_dir = args.dataset_dir
    mixnoise = args.mixnoise
    
 
    if args.save_dir is not None:
        out_filename = args.save_dir + dataset+'_'+args.save_label +'.pkl'
    else:
        out_filename = 'None'

    print('\n')
    print('-'*50)
    print('\nFEATURES EXTRACTION')
    print(f'\t{"Dataset":>20}: {dataset}')
    print(f'\t{"Features":>20}: {features}')
    print(f'\t{"Dataset dir.":>20}: {dataset_dir}')
    print(f'\t{"Features file":>20}: {out_filename}')
    print(f'\t{"Add noise version":>20}: {mixnoise}')

    
    print(f"\nPARAMETERS:")
    for key in params:
        print(f'\t{key:>20}: {params[key]}')
    print('\n')

    # Random seed
    seed_everything(111)
    
    if dataset == 'DOG_EMO':
        #emot_map = {'Hap': 0, 'Sad':1, 'Ang':2, 'Howling':3}
        emot_map = {
            '관심' : 0, #가끔 멍멍 0.25 ~0.3
            '행복' : 1, # 0.5 #빠른 멍
            '흥분(화남)' : 1, #0.5 빠른멍 
            '요구' : 2, #중간멍
            '호기심' :2, #중간멍
            '예민' : 3, # 거의 맞음 
            '경계' : 4, #0.125 - 거의 못맞춤
            '화남' : 5, # 0.15
            '슬픔' : 6,# 거의 맞춤 
            '공포' :7, # -> 케이스 없음
            '반가움' : 8, #0.1666
            '하울링(행복)' : 9, # 자기각색,
            '하울링' : 10,# 거의 맞춤 1.0
            '외로움' : 11, #0.0 -> 아애 못맞춤 
            '신남' : 12,    #케이스 없음 
        }
        database = DOG_EMO_DATABASES[dataset](dataset_dir, emot_map=emot_map, 
                                        include_scripted = False)
        
        
    #Get file paths and label in database , train에 존재하는 data01 ~ 08을 id로 가진다.
    speaker_files = database.get_files()

    #Extract features
    if args.sub_dir:
        speaker_files = {args.sub_dir : speaker_files[args.sub_dir]}
    
    features_data = extract_features(speaker_files, features, params , coc_params)
    #Save features
    if args.save_dir is not None:
        with open(out_filename, "wb") as fout:
                pickle.dump(features_data, fout)

        
    print(f'\nSEGMENT CLASS DISTRIBUTION PER SPEAKER:\n')
    classes = database.get_classes()
    n_speaker=len(features_data)
    n_class=len(classes)
    class_dist= np.zeros((n_speaker,n_class),dtype=np.int64)
    speakers=[]
    data_shape=[]
    for i,speaker in enumerate(features_data.keys()):
    
        cnt = sorted(Counter(features_data[speaker]["seg_label"]).items())
        
        for item in cnt:
            class_dist[i][item[0]]=item[1]
        speakers.append(speaker)
        if mixnoise == True:
            data_shape.append(str(features_data[speaker]["seg_spec"][0].shape))
        else:
            data_shape.append(str(features_data[speaker]["seg_spec"].shape))
    class_dist = np.vstack(class_dist)
    df = {"speakerID": speakers}
    
    for c in range(class_dist.shape[1]):
        df[classes[c]] = class_dist[:,c]
    
    class_dist_f = pd.DataFrame(df)
    class_dist_f = class_dist_f.to_string(index=False) 
    print(class_dist_f)
    
    with open(out_filename.split('.')[0] + '_info.txt' , 'w') as file:
        file.write(class_dist_f)
     
    print('\n')
    print('-'*50)
    print('\n')



def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #DATASET
    parser.add_argument('--dataset', type=str, default='DOG_EMO',
        help='Dataset to extract features. Options:'
             '  - IEMOCAP '
             '  - DOG_EMO(default)')
    parser.add_argument('--dataset_dir', type=str, default='../data/',
        help='Path to the dataset directory.')
    parser.add_argument('--sub_dir' , type=str , default='')
    
    #FEATURES
    parser.add_argument('--features', type=str, default='logmelspec',
        help='Feature to be extracted. Options:'
             '  - logspec (default) : (1 ch.)log spectrogram'
             '  - logmelspec        : (1 ch.)log mel spectrogram'
             )
    
    parser.add_argument('--window', type=str, default='hamming',
        help='Window type. Default: hamming')

    parser.add_argument('--win_length', type=float, default=40,
        help='Window size (msec). Default: 40')

    parser.add_argument('--hop_length', type=float, default=10,
        help='Window hop size (msec). Default: 10')
    
    parser.add_argument('--ndft', type=int, default=800,
        help='DFT size. Default: 800')

    parser.add_argument('--nfreq', type=int, default=200,
        help='Number of lowest DFT points to be used as features. Default: 200'
             '  Only effective for <logspec, lognrevspec> features')
    parser.add_argument('--nmel', type=int, default=128,
        help='Number of mel frequency bands used as features. Default: 128'
             '  Only effectice for <logmel, logmeldeltaspec> features')
    
    parser.add_argument('--segment_size', type=int, default=300,
        help='Size of each features segment')

    parser.add_argument('--mixnoise', action='store_true',
        help='Set this flag to mix with noise.')

    #cochelgram
    
    parser.add_argument('--n_filter' , type = int , default=126)
    parser.add_argument('--low_lim' , type = int ,  default=30)
    parser.add_argument('--hi_lim' , type = int ,  default=5000)
    parser.add_argument('--sample_factor' , type = int ,  default=1)
    parser.add_argument('--nonlinearity' , type = str ,  default='power')


    #FEATURES FILE
    parser.add_argument('--save_dir', type=str, default='./features/',
        help='Path to directory to save the extracted features.')
    
    parser.add_argument('--save_label', type=str, default='multi',
        help='Label to save the feature')

    return parser.parse_args(argv)


# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(parse_arguments(sys.argv[1:]))