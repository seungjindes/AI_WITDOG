import pickle
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn import preprocessing
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import defaultdict

SCALER_TYPE = {'standard':'preprocessing.StandardScaler()',
               'minmax'  :'preprocessing.MinMaxScaler(feature_range=(0,1))'
              }





class TestDataset(torch.utils.data.Dataset):
    """
    Holds data for a validation/test set.

    Parameters
    ----------
    data : ndarray
        Input data of shape `N x C x H x W`, where `N` is the number of examples
        (segments), C is number of input channels (3 in the case of image), `H` is image height, 
        `W` is image width
    actual_target : ndarray
        Actual target labels (labels for utterances) of shape `(U,)`, where
        `U` is the number of utterances.
    seg_target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape `(N,)`.
    num_segs : ndarray
        Array of shape `(U,)` indicating how many segments each utterance
        contains.
    num_classes :
        Number of classes.
    """
        
    def __init__(self, data, num_classes=4):
        super(TestDataset).__init__()
        # self.data = data
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        self.data_coc = data['seg_coc']
        # self.utter_label = data['utter_label']
        # self.seg_label = data['seg_label']
        self.num_segs = data['seg_num'] 
        self.conven_fea = data['seg_conven']
        # self.target = data['seg_label']
        # self.n_samples = len(self.target)
        # self.actual_target = data['utter_label']
        # self.n_actual_samples = len(self.actual_target)
        #self.num_segs = data['seg_num']
        self.num_classes = num_classes
        self.n_samples = len(self.num_segs)

        

        # shape is batch:33 , other shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index], 
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_coc' : self.data_coc[index],
            'seg_num' : self.num_segs[index],
            'seg_conven' : self.conven_fea[index]
            } 
        return sample


    def get_preds(self, seg_preds):
        """
        Get predictions for all utterances from their segments' prediction.
        This function will accumulate the predictions for each utterance by
        taking the maximum probability along the dimension 0 of all segments
        belonging to that particular utterance.
        """
        preds = np.empty(
            shape=(self.n_actual_samples, self.num_classes), dtype="float")

        end = 0
        
        for v in range(self.n_actual_samples):
            start = end
            end = start + self.num_segs[v]
            
            '''
            # remove the last one for long utterances
            if self.num_segs[v] > 1:
                end = end - 1
                
            preds[v] = np.average(seg_preds[start:end], axis=0)
            
            if self.num_segs[v] > 1:
                end = end + 1
            
            
            # choose the most certain one
            tmp_seg = -1
            for seg in range(self.num_segs[v]):
                end_seg = start + seg
                if np.max(seg_preds[end_seg]) - np.min(seg_preds[end_seg]) > tmp_seg:
                    tmp_seg = np.max(seg_preds[end_seg]) - np.min(seg_preds[end_seg])
                    preds[v] = seg_preds[end_seg]
            '''  
            preds[v] = np.average(seg_preds[start:end], axis=0)
                                 
        preds = np.argmax(preds, axis=1)
        return preds


    def weighted_accuracy(self, utt_preds):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Accuracy score.

        """

        acc = (self.actual_target == utt_preds).sum() / self.n_actual_samples
        return acc


    def unweighted_accuracy(self, utt_preds):
        """
        Calculate unweighted accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Unweighted Accuracy (UA) score.

        """
        class_acc = 0
        n_classes = 0
        
        for c in range(self.num_classes):
            class_pred = np.multiply((self.actual_target == utt_preds),
                                     (self.actual_target == c)).sum()

        
            if (self.actual_target == c).sum() > 0:    
                class_pred /= (self.actual_target == c).sum()
                n_classes += 1
                class_acc += class_pred
        
        return class_acc / n_classes

    
    def confusion_matrix_iemocap(self, utt_preds):
        """Compute confusion matrix given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        """
        conf = confusion_matrix(self.actual_target, utt_preds)
        
        # Make confusion matrix into data frame for readability
        print()
        print(self.actual_target)
        print(utt_preds)
        conf_fmt = pd.DataFrame({"hap": conf[:, 0], "sad": conf[:, 1],
                             "ang": conf[:, 2], "fea": conf[:, 3]})
        conf_fmt = conf_fmt.to_string(index=False)
        print(conf_fmt)
        return (conf, conf_fmt)

class SERDataset:

    def __init__(self, features_data, num_classes = 4):

        self.test_spec_data  = features_data['valid']['seg_spec'].astype(np.float32)
        self.test_mfcc_data  = features_data['valid']['seg_mfcc'].astype(np.float32)
        self.test_audio_data  = features_data['valid']['seg_audio'].astype(np.float32)
        self.test_coc_data = features_data['valid']['seg_coc'].astype(np.float32)
        self.test_num_segs   = features_data['valid']['seg_num']
        self.conven_data = features_data['valid']['conven_feature'].astype(np.float32)


        #Normalize dataset to the range of [0, 1] suitable as image pixel
        self._normalize('minmax')
        #convert normalized spectrogram to 3 channel image, apply AlexNet image pre-processing
        self.test_spec_data = self._spec_to_gray(self.test_spec_data)
        self.num_in_ch = 1  
        #self.test_data = self.test_spec_data, self.test_mfcc_data
        self.test_data = defaultdict()
        self.test_data["seg_spec"] = self.test_spec_data
        self.test_data["seg_mfcc"] = self.test_mfcc_data
        self.test_data["seg_audio"] = self.test_audio_data
        self.test_data["seg_coc"] = self.test_coc_data
        self.test_data["seg_num"] = self.test_num_segs
        self.test_data["seg_conven"] = self.conven_data

        self.num_classes = num_classes
 
    # _ python에서 underbar method 는 import 불가  
    def _normalize(self, scaling):
        
        '''
        calculate normalization factor from training dataset and apply to
           the whole dataset
        '''
        
        #get data range
        input_range = self._get_data_range()

        #re-arrange array from (N, C, F, T) to (C, -1, F)
        nsegs = self.test_spec_data.shape[0]
        nch   = self.test_spec_data.shape[1]
        nfreq = self.test_spec_data.shape[2]
        ntime = self.test_spec_data.shape[3]
        rearrange = lambda x: x.transpose(1,0,3,2).reshape(nch,-1,nfreq)

        self.test_spec_data  = rearrange(self.test_spec_data)
        
        #re-arrange array form (C , F, T) -> (C , T , F)
        self.test_coc_data = self.test_coc_data.transpose(0,2,1)
        
        #scaler type
        scaler = eval(SCALER_TYPE[scaling])

        for ch in range(nch):
            #get scaling values from training data
            scale_values = scaler.fit(self.test_spec_data[ch])

            self.test_spec_data[ch] = scaler.transform(self.test_spec_data[ch])
            self.test_coc_data[ch] = scaler.transform(self.test_coc_data[ch])
            
    
        
        #Shape the data back to (N,C,F,T)
        rearrange = lambda x: x.reshape(nch,-1,ntime,nfreq).transpose(1,0,3,2)
        self.test_spec_data  = rearrange(self.test_spec_data)
        self.test_coc_data = self.test_coc_data.transpose(0, 2, 1)

        print(f'\nDataset normalized with {scaling} scaler')
        print(f'\tRange before normalization: {input_range}')
        print(f'\tRange after  normalization: {self._get_data_range()}')

    def _get_data_range(self):
        #get data range
        tsmin = np.min(self.test_spec_data)
        dmin = np.min(np.array([tsmin]))
        tsmax = np.max(self.test_spec_data)
        dmax = np.max(np.array([tsmax]))
        
        return [dmin, dmax]

    def _spec_to_rgb(self,data):

        """
        Convert normalized spectrogram to pseudo-RGB image based on pyplot color map
            and apply AlexNet image pre-processing
        
        Input: data
                - shape (N,C,H,W) = (num_spectrogram_segments, 1, Freq, Time)
                - data range [0.0, 1.0]
        """

        #AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
                transforms.Resize(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]) 

        # Get the color map to convert normalized spectrum to RGB
        cm = plt.get_cmap('jet') #brg #gist_heat #brg #bwr

        #Flip the frequency axis to orientate image upward, remove Channel axis
        data = np.flip(data,axis=2)
        
        data = np.squeeze(data, axis=1) 

        data_tensor = list()

        for i, seg in enumerate(data):
            seg = np.clip(seg, 0.0, 1.0)
            seg_rgb = (cm(seg)[:,:,:3]*255.0).astype(np.uint8)
            
            img = Image.fromarray(seg_rgb, mode='RGB')

            data_tensor.append(alexnet_preprocess(img))
        
        return data_tensor


    def _spec_to_gray(self,data):

        """
        Convert normalized spectrogram to 3-channel gray image (identical data on each channel)
            and apply AlexNet image pre-processing
        
        Input: data
                - shape (N,C,H,W) = (num_spectrogram_segments, 1, Freq, Time)
                - data range [0.0, 1.0]
        """

        #AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
                transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]) 

        #Convert format to uint8, flip the frequency axis to orientate image upward, 
        #   duplicate into 3 channels
        data = np.clip(data,0.0, 1.0)
        data = (data*255.0).astype(np.uint8)
        data = np.flip(data,axis=2)
        data = np.moveaxis(data,1,-1)
        data = np.repeat(data,3,axis=-1)
       
        data_tensor = list()
        for i, seg in enumerate(data):
            img = Image.fromarray(seg, mode='RGB')
            data_tensor.append(alexnet_preprocess(img))
            
        return data_tensor  
    

    def get_test_dataset(self):
        return TestDataset(
            self.test_data, num_classes=self.num_classes)
                       

def random_oversample(data, labels):
    print('\tOversampling method: Random Oversampling')
    ros = RandomOverSampler(random_state=0,sampling_strategy='minority')

    n_samples = data.shape[0]
    fh = data.shape[2]
    fw = data.shape[3]
    n_features= fh*fw
        
    data = np.squeeze(data,axis=1)
    data = np.reshape(data,(n_samples, n_features))
    data_resampled, label_resampled = ros.fit_resample(data, labels)
    n_samples = data_resampled.shape[0]
    data_resampled = np.reshape(data_resampled,(n_samples,fh,fw))
    data_resampled = np.expand_dims(data_resampled, axis=1)
    
    return data_resampled, label_resampled


