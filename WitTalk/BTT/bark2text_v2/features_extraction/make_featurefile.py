import os


train_data_path="/home/dog_emotion_dataset_label14/train"

# train_sub_dirs=['data01','data02','data03','data04','data05','data06',
#                 'data07','data08','data09','data10','data11','data12',
#                 'data13','data14','data15','data16','data17','data18',
#                 'data19','data20','data21','data22'] 

train_sub_dirs=['data09'] 

for sub_dir in train_sub_dirs:
    features_name= sub_dir + '_label_14'
    os.system(f"python extract_features.py --dataset_dir={train_data_path} --save_label={features_name} --sub_dir={sub_dir}")


# valid_data_path="/home/dog_emotion_dataset_label14/valid"

# valid_sub_dirs=["data01"] 

# for sub_dir in valid_sub_dirs:
#     features_name= 'valid_label_14'
#     os.system(f"python extract_features.py --dataset_dir={valid_data_path} --save_label={features_name} --sub_dir={sub_dir}")
    
    
