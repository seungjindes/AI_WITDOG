import train_ser
from train_ser import parse_arguments
import sys
import pickle
import os
import time


repeat_kfold = 2 # to  perform 10-fold for n-times with different seed
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

features_file = './features_extraction/features/DOG_EMO_multi_maltese.pkl'
 #leave-one-session-out
val_id = ['valid']
test_id = ['test']

num_epochs  = '1'
batch_size  = '4'
#lr          = 0.00001
lr = '0.001'
random_seed = 12500
#Loss: 1.0824 - 2.1091 - WA: 5.56% <5.56%> - UA: 16.67% <16.67%>
save_label = str_time#'0930_01'#'alexnet_pm_0704'


#Start Cross Validation
all_stat = []
# 2.0869 - 1.2119
for repeat in range(repeat_kfold):

    random_seed +=  (repeat*100)
    seed = str(random_seed)

    for v_id, t_id in list(zip(val_id, test_id)):

        train_ser.sys.argv      = [
                        
                                  'train_ser.py', 
                                  features_file,
                                  '--repeat_idx', str(repeat),
                                  '--val_id',v_id, 
                                  '--test_id', t_id,
                                  '--num_epochs', num_epochs,
                                  '--batch_size', batch_size,
                                  '--lr', lr,
                                  '--seed', seed,
                                  '--save_label', save_label,#,
                                  '--pretrained'
                                  ]


        stat = train_ser.main(parse_arguments(sys.argv[1:]))   
        all_stat.append(stat)       
        #os.remove('./result/'+save_label+'.pth')
        
        
    
    
    # with open('allstat_iemocap_'+save_label+'_'+str(repeat)+'.pkl', "wb") as fout:
    #     pickle.dump(all_stat, fout)

# n_total = repeat_kfold*len(val_id)
# total_best_epoch, total_epoch, total_loss, total_wa, total_ua = 0, 0, 0, 0, 0

# for i in range(n_total):
#     print(i, ": ", all_stat[i][0], all_stat[i][1], all_stat[i][8], all_stat[i][9], all_stat[i][10]) 
#     total_best_epoch += all_stat[i][0]
#     total_epoch += all_stat[i][1]
#     total_loss += float(all_stat[i][8])
#     total_wa += float(all_stat[i][9])
#     total_ua += float(all_stat[i][10])

# print("AVERAGE:", total_best_epoch/n_total, total_epoch/n_total, total_loss/n_total, total_wa/n_total, total_ua/n_total )

# print(all_stat)
