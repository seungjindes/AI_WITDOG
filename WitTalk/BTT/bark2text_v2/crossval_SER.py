import train_ser_conven
from train_ser_conven import parse_arguments

#import train_ser
#from train_ser import parse_arguments
import sys
import pickle
import os
import time

from tqdm import tqdm

repeat_kfold = 400 # to  perform 10-fold for n-times with different seed
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

features_file = '/home/bark2text_newlabel/features_extraction/features/DOG_EMO_data'
valid_features_file = '/home/bark2text_newlabel/features_extraction/features/DOG_EMO_valid_label_14.pkl'
#leave-one-session-out
#soft-labeling 도 

num_epochs  = '1'
batches = ['12']
lrs = ['0.00001']
random_seed = 120


## 데이터 증강 실시하기 , 파인튜닝 후 학습시키기 ,ㄷ ㅔ이터 피드백하기 

margin = '0.5'
features_tags = [
                 '01_label_14.pkl' ,'02_label_14.pkl' ,'03_label_14.pkl' ,'04_label_14.pkl',
                 '05_label_14.pkl' ,'06_label_14.pkl' ,'07_label_14.pkl' ,'08_label_14.pkl',
                 '09_label_14.pkl' ,'10_label_14.pkl' ,'11_label_14.pkl' ,'12_label_14.pkl',
                 '13_label_14.pkl' ,'14_label_14.pkl' ,'15_label_14.pkl' ,'16_label_14.pkl',
                 '17_label_14.pkl' ,'18_label_14.pkl' ,'19_label_14.pkl' ,'20_label_14.pkl',
                 '21_label_14.pkl' ,'22_label_14.pkl'
                 ]



#Start Cross Validation
all_stat = []
for lr in lrs:
    for batch in batches:
        save_label = 'lr-' + str(lr) + '_batch-'+ batch +'label14_0529'
        os.makedirs(os.path.join('./result' ,save_label) , exist_ok=True)
        for repeat in tqdm(range(0 ,  repeat_kfold+1)): 

            with open(os.path.join('./result' , save_label , 'log.txt') ,'a') as file:
                file.write(f"### EPOCH :{repeat} ###\n")
                
            for features_tag in features_tags:
                seed = str(random_seed)
                train_ser_conven.sys.argv  = [
                                
                                        'train_ser_conven.py', 
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