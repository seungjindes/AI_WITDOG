#!/bin/sh



features_name="multi_maltese"

# data 폴더는 모든 강아지의 품종을 학습에 이용한다. 특정 종만을 학습시키고 싶을경우 해당 종의 폴더를 경로로 설정해준다.
data_path="../data/maltese"


python extract_features.py --dataset_dir=${data_path} --save_label=${features_name}

