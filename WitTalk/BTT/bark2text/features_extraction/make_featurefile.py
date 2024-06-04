import os



# data 폴더는 모든 강아지의 품종을 학습에 이용한다. 특정 종만을 학습시키고 싶을경우 해당 종의 폴더를 경로로 설정해준다.
train_data_path="../data/train"

train_sub_dirs=["data01","data02","data03","data04","data05","data06","data07","data08","data09","data10","data11","data12"] 

for sub_dir in train_sub_dirs:
    features_name= sub_dir + 'add_conven_features'
    os.system(f"python extract_features.py --dataset_dir={train_data_path} --save_label={features_name} --sub_dir={sub_dir}")

    #echo "complete"


valid_data_path="../data/train"

valid_sub_dirs=["valid"] 

for sub_dir in valid_sub_dirs:
    features_name= 'add_conven_features_valid'
    os.system(f"python extract_features.py --dataset_dir={valid_data_path} --save_label={features_name} --sub_dir={sub_dir}")
