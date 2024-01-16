import os
import sys
import argparse
import numpy as np
from features_extraction.features_util import extract_features
from features_extraction.database import DOG_EMO_DATABASES
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
    
    dataset  = 'DOG_EMO'
    features = args.features
    dataset_dir = args.dataset_dir
    mixnoise = args.mixnoise

    # Random seed
    seed_everything(111)
    

    emot_map = {'hap': 0, 'sad':1, 'ang':2, 'fea':3}
    #30초단위로분리된 폴더의이름을준다.
    database = DOG_EMO_DATABASES[dataset](dataset_dir, emot_map=emot_map, 
                                    include_scripted = False)
        
        
    #Get file paths and label in database
    speaker_files = database.get_files()

    #Extract features
    features_data = extract_features(speaker_files, features, params , coc_params)

    return features_data



def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #DATASET


    parser.add_argument('--dataset_dir', type=str, default='../test_data/input_data_from_local/',
        help='Path to the dataset directory.')
    
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

    return parser.parse_args(argv)

# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(parse_arguments(sys.argv[1:]))