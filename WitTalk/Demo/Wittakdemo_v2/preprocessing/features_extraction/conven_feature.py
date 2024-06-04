from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

import numpy as np

from hurst import compute_Hc
from scipy.stats import skew


FS = 16000

def compute_hurst(audio_file, window = 0.1, step = 0.02 , threshold = 0.5):

    # Load audio file
    x = audio_file
    
    # Check if x is 1D
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)  # Make it 2D by adding a dimension
    
    # Extract short-term features
    features = []
    [x_features, short_term_features] = ShortTermFeatures.feature_extraction(x[:, 0], FS, window * FS, step * FS)
    
    # Compute Hurst exponent for each feature
    for feature , feature_name , in zip(x_features , short_term_features):
        H = compute_Hc(feature)
        H_val = H[0]
        num_outliers = np.sum(feature > threshold)
        features.append((H_val ,num_outliers))

    return np.array(features)


def compute_outlier(audio_file, window=0.1, step=0.02, threshold=0.5):
    # Load audio file
    x = audio_file
    
    # Check if x is 1D
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)  # Make it 2D by adding a dimension
    
    # Extract short-term features
    [x_features, short_term_features] = ShortTermFeatures.feature_extraction(x[:, 0], FS, window * FS, step * FS)
    
    # Compute number of outliers for each feature
    outlier_features = []
    for feature in x_features:
        num_outliers = np.sum(feature > threshold)
        outlier_features.append(num_outliers)
    
    return np.expand_dims(np.array(outlier_features) , 1)

def compute_skewness(x, window = 0.3, step=0.04): 
    # 스테레오 오디오를 모노로 변환
    if len(x.shape) == 2:
        x = np.mean(x, axis=1)
    
    # 비대칭도 계산을 위한 프레임별 추출
    frame_size = int(window * FS)
    step_size = int(step * FS)
    
    skews = []
    for i in range(0, len(x) - frame_size, step_size):
        frame = x[i:i+frame_size]
        frame_skew = skew(frame)
        skews.append(frame_skew)
    
    return np.expand_dims(np.array(skews) , 1)


from scipy import stats

def compute_mode_features(x, window=0.3, step=0.04):

    # 스테레오 오디오를 모노로 변환
    if len(x.shape) == 2:
        x = np.mean(x, axis=1)
    
    # 모드 계산을 위한 프레임별 추출
    frame_size = int(window * FS)
    step_size = int(step * FS)
    
    modes = []
    for i in range(0, len(x) - frame_size, step_size):
        frame = x[i:i+frame_size]
        mode = stats.mode(frame)[0]
        modes.append(mode)
    
    return np.expand_dims(np.array(modes) , 1)



def compute_kurtosis_features(x , window=0.3, step=0.04):
    frame_size = int(window * FS)
    step_size = int(step * FS)
    
    kurtosis_values = []
    for i in range(0, len(x) - frame_size + 1, step_size):
        frame = x[i:i+frame_size]
        frame_kurtosis = stats.kurtosis(frame, fisher=True) 
        kurtosis_values.append(frame_kurtosis)
    
    return np.expand_dims(np.array(kurtosis_values) , axis = 1)

DOG_MAP = {
    "바셋" : 0,
    "비글" : 1,
    "비숑" : 2,
    "불독" : 3,
    "치와와" : 4, 
    "닥스훈트" :5,
    "그레이하운드" : 6,
    "허스키" : 7,
    "말티즈" : 8,
    "포메라니안" : 9,
    "레트리버" : 10,
    "세퍼드" : 11,
    "시바견" : 12,
    "시츄" : 13,
    '믹스견' : 14,
    "보더콜리" : 15,
    "요크셔테리어" : 16,
    "진돗개" : 17,
    "푸들" : 18 ,
    "허스키" : 19
}
def dog_onehotencoding(species):
    label = DOG_MAP[species]
    species_vector = np.zeros(68)
    species_vector[label] = 1
    return np.expand_dims(species_vector , axis =1)

def sound_onehotencoding(sound):
    sound_vector = np.zeros(68)
    if sound  == 'barking':
        sound_vector[0] = 1
    elif sound  == 'whining':
        sound_vector[1] = 1
    elif sound  == 'growling':
        sound_vector[2] = 1
    elif sound  == 'howling':
        sound_vector[3] = 1
    
    return np.expand_dims(sound_vector , axis =1)

def integral_onehotencoding(place ,age , num):

    integral_vector = np.zeros(68)

    if place  == 'own':
        integral_vector[0] = 1
    else:
        integral_vector[1] = 1
    if age  == 'o':
        integral_vector[2] = 1
    else:
        integral_vector[3] = 1
    if num =='s':
        integral_vector[4] = 1
    else:
        integral_vector[5] = 1


    return np.expand_dims(integral_vector , axis =1)


def get_conven_features(x, species , sound , target , age , num):

    husrt = compute_hurst(x)
    nol = compute_outlier(x)
    skew = compute_skewness(x)
    modes = compute_mode_features(x)
    kurt = compute_kurtosis_features(x)

    species_vector = dog_onehotencoding(species)
    sound_vector =  sound_onehotencoding(sound)
    integral_vector =  integral_onehotencoding(target , age , num)


    conven_features = np.concatenate([husrt , nol , skew , modes , kurt  ,species_vector ,sound_vector ,integral_vector] , axis =1)
    return conven_features 








8