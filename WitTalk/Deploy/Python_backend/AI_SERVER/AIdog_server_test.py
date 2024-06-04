import argparse
import os
import sys
from functools import partial
import json
import torch
import numpy as np

from preprocessing.features_extraction.features_util import extract_features
from preprocessing.data_utils import SERDataset


from tritonclient.utils import InferenceServerException, triton_to_np_dtype
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient

import librosa

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue
import threading

#EMO_MAP = {'hap': 0, 'sad':1, 'ang':2, 'fea':3}

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


def requestGenerator(batched_data, input_name, output_name, dtype, FLAGS):


    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = []

    for data , features in zip(batched_data , input_name) :
        inputs.append(client.InferInput(features.name , data.shape, dtype))
        inputs[-1].set_data_from_numpy(data)

    outputs = [client.InferRequestedOutput(output_name[0].name, class_count=FLAGS.classes)]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version

 
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
        obj = {"emo" : "happy"}
    elif cls[1]== '1':
        obj = {"emo" : "sad"}
    elif cls[1] == '2':
        obj = {"emo" : 'angry'}
    elif cls[1] == '3':
        obj = {"emo" : 'fear'}
    else:
        obj = {"emo" : 'unknown'}

    return obj

 



        

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", required=False, default=False, help="Enable verbose output")
    parser.add_argument("-a", "--async", dest="async_set", required=False, default=True, help="Use asynchronous inference API")
    parser.add_argument("--streaming", action="store_true", required=False, default=False, help="Use streaming inference API (only available with gRPC protocol)")
    parser.add_argument("-m", "--model-name", type=str, required=False, default="AIdog", help="Name of model")
    parser.add_argument("-x", "--model-version", type=str, required=False, default="", help="Version of model. Default is to use latest version.")
    parser.add_argument("-c", "--classes", type=int, required=False, default=4, help="Number of class results to report. Default is 1.")
    parser.add_argument("-s", "--scaling", type=str, choices=["NONE", "INCEPTION", "VGG"], required=False, default="NONE", help="Type of scaling to apply to image pixels. Default is NONE.")
    parser.add_argument("-u", "--url", type=str, required=False, default="172.17.0.1:8005", help="Inference server URL. Default is localhost:8000.")
    parser.add_argument("-i", "--protocol", type=str, required=False, default="HTTP", help="Protocol (HTTP/gRPC) used to communicate with the inference service. Default is HTTP.")
    FLAGS = parser.parse_args()


    if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")

    try:
        if FLAGS.protocol.lower() == "grpc":
            # Create gRPC client for communicating with the server
            print("create grpc client")
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose
            )
        else:
            # Specify large enough concurrency to handle the
            print("create http client")
            if FLAGS.async_set:
                print("using async")
                concurrency = 20 
            else : concurrency = 1             
            
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency
            )
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)


    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=""
        )
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)


    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version
        )
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)


    if FLAGS.protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config
        )
    
    max_batch_size ,input_name, output_name, spec_size , mfcc_size, audio_size , coc_size ,conven_size, output_size  , dtype = parse_model(
        model_metadata, model_config
    )
    
    ### data load ### 
    ### output is input_spec , input_mfcc , input_audio , input_coc
    pipleline = DataPipeline()

    #response = supabase_client.table('AI_QUEUE').select("*").execute()
    # json_data = response.json()


    folder_path = './test_samples'
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    response_datas = []
    for index, wav_file in enumerate(wav_files):
        file_path = os.path.join(folder_path, wav_file)
        x, sr = librosa.load(file_path, sr=16000)
        response_datas.append({'id' : index  , 'audio_data' : x , 'sr' : sr , 'wav_path' : wav_file})

    # data = json.loads(json_data)
    # response_datas = data["data"]    

 
    requests = []
    infer_responses = []
    result_filenames = []
    request_ids = []
    image_idx = []
    last_request = False
    user_data = UserData() 

    async_requests = []
    sent_count = 0

    for response_data in response_datas:
        print('------------------------------------------------')
        x , sr = np.array(response_data['audio_data']) , response_data['sr']
        wav_name = response_data['wav_path']

        print(wav_name)
        input_data = pipleline.execute(x, sr)
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                input_data , input_name, output_name, dtype, FLAGS
            ):
                sent_count +=1
                if FLAGS.streaming:
                    triton_client.async_stream_infer(
                            FLAGS.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs,
                        )
                elif FLAGS.async_set:
                    if FLAGS.protocol.lower() == "grpc":
                        triton_client.async_infer(
                            FLAGS.model_name,
                            inputs,
                            partial(completion_callback, user_data),
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs,
                        )
                    else:
                        async_requests.append(
                            triton_client.async_infer(
                                FLAGS.model_name,
                                inputs,
                                request_id=str(sent_count),
                                model_version=FLAGS.model_version,
                                outputs=outputs,
                            )
                            )
                else:
                    infer_responses.append(
                        triton_client.infer(
                            FLAGS.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs,
                        )
                    )
        except InferenceServerException as e:
            print("inference failed: " + str(e))
            if FLAGS.streaming:
                triton_client.stop_stream()
            sys.exit(1)

        if not FLAGS.async_set:
            obj = postprocess(infer_responses[-1], output_name[0])
            print("infer result : " , obj["emo"])    




    if FLAGS.streaming:
        triton_client.stop_stream()
        
    if FLAGS.protocol.lower() == "grpc":
        if FLAGS.streaming or FLAGS.async_set:
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                processed_count += 1
                if error is not None:
                    print("inference failed: " + str(error))
                    sys.exit(1)
                # 아래거 수정 필요할지도 
                infer_responses.append(results)
    else:
        if FLAGS.async_set:
            # Collect results from the ongoing async requests
            # for HTTP Async requests.
            for async_request in async_requests:
                infer_responses.append(async_request.get_result())


    if FLAGS.async_set:
        for infer_response in infer_responses: 
            obj = postprocess(infer_response, output_name[0])
            print("infer result : " , obj["emo"])    

    print('Infer Complete..')

 

        
    
