import train_ser
from train_ser import parse_arguments
import train_ser_conven
from train_ser_conven import parse_arguments
import sys
import pickle
import os
import time

from tqdm import tqdm

repeat_kfold = 800 # to  perform 10-fold for n-times with different seed
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

features_file = './features_extraction/features/DOG_EMO_data'
valid_features_file = './features_extraction/features/DOG_EMO_add_conven_features_valid.pkl'
 #leave-one-session-out
num_epochs  = '1'
batches = ['8']
lrs = ['0.00001']
random_seed = 120

margin = '0.5'
features_tags = ['01add_conven_features.pkl' , '02add_conven_features.pkl', '03add_conven_features.pkl','04add_conven_features.pkl',
                 '05add_conven_features.pkl','06add_conven_features.pkl','07add_conven_features.pkl',
                 '08add_conven_features.pkl','09add_conven_features.pkl','10add_conven_features.pkl','11add_conven_features.pkl',
                 '12add_conven_features.pkl']


#Start Cross Validation
all_stat = []
for lr in lrs:
    for batch in batches:
        save_label = 'lr-' + str(lr) + '_batch-'+ batch +'_spec-efficient'
        os.makedirs(os.path.join('./result' ,save_label) , exist_ok=True)
        for repeat in tqdm(range(800 , 1600)): 
            for features_tag in features_tags:
                seed = str(random_seed)
                train_ser_conven.sys.argv  = [
                                        'train_ser.py', 
                                        '--train_features_file',features_file + features_tag,
                                        '--valid_features_file', valid_features_file,
                                        '--repeat_idx', str(repeat),
                                        '--num_epochs', num_epochs,
                                        '--batch_size', batch,
                                        '--lr', lr,
                                        '--margin' , margin,
                                        '--seed', seed,
                                        '--save_label', save_label,
                                        '--pretrained'
                                    ]

                stat = train_ser_conven.main(parse_arguments(sys.argv[1:]))   
                all_stat.append(stat)       

            #os.remove('./result/'+save_label+'.pth')