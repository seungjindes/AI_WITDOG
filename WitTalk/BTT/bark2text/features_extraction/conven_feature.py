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
        frame_kurtosis = stats.kurtosis(frame, fisher=True)  # Fisher 첨도 사용 (정규 분포의 첨도=0)
        kurtosis_values.append(frame_kurtosis)
    
    return np.expand_dims(np.array(kurtosis_values) , axis = 1)



DOG_MAP = {
    "Basset" : 0,
    "Beagle" : 1,
    "Bichon" : 2,
    "Bulldog" : 3,
    "Chihuahua" : 4, 
    "Dachshund" :5,
    "Greyhound" : 6,
    "Hursky" : 7,
    "Maltese" : 8,
    "maltese" : 8,
    "Pomeranian" : 9,
    "Retriever" : 10,
    "Reariever" : 10,
    "shepherd" : 11,
    "ShibaInu" : 12,
    "Shihtau" : 13
}

def dog_onehotencoding(species):
    label = DOG_MAP[species]
    species_vector = np.zeros(68)
    species_vector[label] = 1
    return np.expand_dims(species_vector , axis =1)

def get_conven_features(x , species):

    husrt = compute_hurst(x)
    nol = compute_outlier(x)
    skew = compute_skewness(x)
    modes = compute_mode_features(x)
    kurt = compute_kurtosis_features(x)
 
    species_vector = dog_onehotencoding(species)

    print('husrt:' , husrt.shape) # 2
    print('nol:' , nol.shape)
    print('skew:' ,skew.shape)
    print('modes:' ,modes.shape)
    print('kurt:' ,kurt.shape)
    print('species:' ,species_vector.shape)

   # shape  (68, 7)
    conven_features = np.concatenate([husrt , nol , skew , modes , kurt , species_vector] , axis =1)
    return conven_features








8