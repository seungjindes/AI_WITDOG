import argparse
import os
import sys
from functools import partial
import json
import torch
import numpy as np
from PIL import Image

from preprocessing.features_extraction.features_util import extract_features
from preprocessing.data_utils import SERDataset


from tritonclient.utils import InferenceServerException, triton_to_np_dtype
import tritonclient.http as httpclient

class DataPipeline:
    def __init__(self):

        self.params={'window'   : 'hamming',
                'win_length'    : 40,
                'hop_length'    : 10,
                'ndft'          : 800,
                'nfreq'         : 200,
                'nmel'          : 128,
                'segment_size'  : 300
                }
        self.coc_params={
                    'n_filter': 126,
                    'low_lim': 30,
                    'hi_lim': 5000,
                    'sample_factor':1,
                    'nonlinearity': 'power'
                    }
        
        self.batch_size = 1
        self.features = 'logmelspec'
        self.dataset = 'DOG_EMO'
        
    def execute(self, x , sr):
        data_features = extract_features(x,sr, self.features, self.params , self.coc_params)
        ser_dataset = SERDataset(data_features)
        
        # Each data unit is sequence of audio wave
        # Sample rate : 16,000Hz ... 
        test_dataset = ser_dataset.get_test_dataset()

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)       
        test_batch = next(iter(test_loader))

        test_data = {
            'spec': test_batch['seg_spec'],
            'mfcc': test_batch['seg_mfcc'],
            'audio': test_batch['seg_audio'],
            'coc': test_batch['seg_coc'],
            'conven': test_batch['seg_conven']
        }
        
        output_tensor = [test_data[key].numpy().astype('float32') for key in test_data]
       
        return output_tensor


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


FLAGS = None


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 5:
        raise Exception("expecting 5 input, got {}".format(len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception(
            "expecting 1 output, got {}".format(len(model_metadata.outputs))
        )

    if len(model_config.input) != 5:
        raise Exception(
            "expecting 5 input in model configuration, got {}".format(
                len(model_config.input)
            )
        )

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output_metadata.datatype
        )

    return (
        model_config.max_batch_size,
        model_metadata.inputs,
        model_metadata.outputs,
        model_metadata.inputs[0].shape,
        model_metadata.inputs[1].shape,
        model_metadata.inputs[2].shape,
        model_metadata.inputs[3].shape,
        model_metadata.inputs[4].shape,
        model_metadata.outputs[0].shape,
        input_metadata.datatype,
    )

def convert_http_metadata_config(_metadata, _config):
    try:
        from attrdict import AttrDict
    except ImportError:
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)


def requestGenerator(batched_data, input_name, output_name, dtype):

    client = httpclient

    # Set the input data
    inputs = []

    for data , features in zip(batched_data , input_name) :
        inputs.append(client.InferInput(features.name , data.shape, dtype))
        inputs[-1].set_data_from_numpy(data)

    outputs = [client.InferRequestedOutput(output_name[0].name, class_count=4)]

    yield inputs, outputs, "Aidog" , ""


def postprocess(results, output_name):
    """
    Post-process results to show classifications.
    """
    output_array = results.as_numpy(output_name.name)
    # Inclue special handling for non-batching models
    # using max value 
    for result in output_array:
        if output_array.dtype.type == np.object_:
            cls = "".join(chr(x) for x in result).split(":")
        else:
            cls = result.split(":")
        break
    # cls consist ('value' , 'arg')
    if cls[1] == '0':
        obj = {"emo" : "bark"}
    elif cls[1]== '1':
        obj = {"emo" : "whining"}
    elif cls[1] == '2':
        obj = {"emo" : 'growling'}
    elif cls[1] == '3':
        obj = {"emo" : 'howling'}
    else:
        obj = {"emo" : 'unknown'}

    return obj

def get_emotion_using_triton(input_data , url = "172.17.0.1" , port = "8005" , concurrency = 1):

    url = url + ":" + port
   
    # Inference server URL. Default is localhost:8005.
    try:
        triton_client = httpclient.InferenceServerClient(
            url = url,
            verbose = False,
            concurrency=concurrency
        )
    except InferenceServerException as e:
        print("failed to set up trition client" , str(e))


    # model의 메타데이터 확인
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name= "AIdog", model_version=""
        )
    except InferenceServerException as e:
        print("failed to retrieve the model metadata: " + str(e))
        sys.exit(1)

    # model의 config 파일 확인 
    try:
        model_config = triton_client.get_model_config(
            model_name="AIdog", model_version=""
        )
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    # metadata와 config파일을 http형식으로 변환 하기
    model_metadata , model_config = convert_http_metadata_config(model_metadata , model_config)

    max_batch_size ,input_name, output_name, spec_size , mfcc_size, audio_size , coc_size ,conven_size, output_size  , dtype = parse_model(
        model_metadata, model_config
    )

    for inputs, outputs, model_name, model_version in requestGenerator(
        input_data , input_name, output_name, dtype
    ):
        infer_response = triton_client.infer(
            "AIdog",
            inputs,
            model_version = model_version,
            outputs = outputs,
            
        )

    obj = postprocess(infer_response , output_name[0])
    return obj["emo"]


