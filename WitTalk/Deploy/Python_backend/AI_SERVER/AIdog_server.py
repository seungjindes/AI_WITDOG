import argparse
import os
import sys
from functools import partial
import json
import torch
import numpy as np
from PIL import Image
import logging
import concurrent.futures

from preprocessing.features_extraction.features_util import extract_features
from preprocessing.data_utils import SERDataset


from tritonclient.utils import InferenceServerException, triton_to_np_dtype
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient


if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

import supabase

SUPABASE_URL = "https://fnjsdxnejydzzlievpie.supabase.co/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZuanNkeG5lanlkenpsaWV2cGllIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwMzIxMzAyOCwiZXhwIjoyMDE4Nzg5MDI4fQ.DBcvEFlnsh3jlLLDWNAE8BIgYaLAhO2sMBwTFvVx23c"
supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_KEY)



EMO_MAP = {'0': "happy", '1': "sad", '2': "angry", '3': "fear"}

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
        # extract feature 수정필요, conven feature 추가 안되있음 
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
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)
            )
        )

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
        model_metadata.outputs[0].shape  ,
        model_metadata.inputs[0].datatype,
        )

def convert_http_metadata_config(_metadata, _config):
    """
    Convert from HTTP API meta data and config object to type obj that can easily use
    AttrDict Import -> AttrDict는 딕셔너리 키를 속성처럼 접근할 수 있게 해주는 유틸리티
    Import Error -> collections , collection.abc를 import하고 모든 이름을 반복하면서, 
    항목을 collection에 할당
    """
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
    inputs = []

    for data , features in zip(batched_data , input_name) :
        inputs.append(client.InferInput(features.name , data.shape, dtype))
        inputs[-1].set_data_from_numpy(data)

    outputs = [client.InferRequestedOutput(output_name[0].name, class_count=FLAGS.classes)]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def postprocess(results, output_name):
    output_array = results[0].as_numpy(output_name.name)
    
    cls = next(("".join(chr(x) for x in result).split(":") for result in output_array \
        if output_array.dtype.type == np.object_), None)
    if cls:
        emo_map = {'0': "happy", '1': "sad", '2': "angry", '3': "fear"}
        return {"emo": emo_map.get(cls[1], "unknown")}
    return {"emo": "unknown"}

def handle_inference_response(response_data, pipeline, input_name, output_name, dtype, FLAGS, triton_client, user_data):
    x, sr = np.array(response_data['audio_data']), response_data['SR']
    input_data = pipeline.execute(x, sr)

    infer_responses = []
    try:
        for inputs, outputs, model_name, model_version in requestGenerator(input_data, input_name, output_name, dtype, FLAGS):
            if FLAGS.streaming:
                triton_client.async_stream_infer(model_name, inputs, request_id=str(response_data['id']), model_version=model_version, outputs=outputs)
            elif FLAGS.async_set:
                if FLAGS.protocol.lower() == "grpc":
                    triton_client.async_infer(model_name, inputs, partial(completion_callback, user_data), request_id=str(response_data['id']), model_version=model_version, outputs=outputs)
                else:
                    infer_responses.append(triton_client.async_infer(model_name, inputs, request_id=str(response_data['id']), model_version=model_version, outputs=outputs).get_result())
            else:
                infer_responses.append(triton_client.infer(model_name, inputs, request_id=str(response_data['id']), model_version=model_version, outputs=outputs))
    except InferenceServerException as e:
        print("Inference failed: " + str(e))
        if FLAGS.streaming:
            triton_client.stop_stream()
        raise

    if FLAGS.streaming:
        triton_client.stop_stream()

    if FLAGS.protocol.lower() == "grpc" and (FLAGS.streaming or FLAGS.async_set):
        processed_count = 0
        while processed_count < len(response_data):
            results, error = user_data._completed_requests.get()
            processed_count += 1
            if error:
                print("Inference failed: " + str(error))
                raise error
            infer_responses.append(results)

    obj = postprocess(infer_responses, output_name[0])
    #supabase_client.table('AI_QUEUE').delete().eq('id', response_data['id']).execute()
    #supabase_client.table('pet_record').insert({'pet_labeling': obj['emo'], 'pet_id': response_data['pet_id']}).execute()
    print("Processed and inserted result for pet_id:"  , 'obj : ' ,obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", required=False, default=False, help="Enable verbose output")
    parser.add_argument("-a", "--async", dest="async_set", action="store_true", required=False, default=False, help="Use asynchronous inference API")
    parser.add_argument("--streaming", action="store_true", required=False, default=False, help="Use streaming inference API (only available with gRPC protocol)")
    parser.add_argument("-m", "--model-name", type=str, required=False, default="AIdog", help="Name of model")
    parser.add_argument("-x", "--model-version", type=str, required=False, default="", help="Version of model. Default is to use latest version.")
    parser.add_argument("-c", "--classes", type=int, required=False, default=4, help="Number of class results to report. Default is 1.")
    parser.add_argument("-s", "--scaling", type=str, choices=["NONE", "INCEPTION", "VGG"], required=False, default="NONE", help="Type of scaling to apply to image pixels. Default is NONE.")
    parser.add_argument("-u", "--url", type=str, required=False, default="172.17.0.1:8005", help="Inference server URL. Default is localhost:8000.")
    parser.add_argument("-i", "--protocol", type=str, required=False, default="HTTP", help="Protocol (HTTP/gRPC) used to communicate with the inference service. Default is HTTP.")
    FLAGS = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if FLAGS.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")

    try:
        if FLAGS.protocol.lower() == "grpc":
            triton_client = grpcclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)
        else:
            concurrency = 20 if FLAGS.async_set else 1
            triton_client = httpclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency)
    except Exception as e:
        logger.error(f"Client creation failed: {str(e)}")
        sys.exit(1)

    try:
        model_metadata = triton_client.get_model_metadata(model_name=FLAGS.model_name, model_version="")
    except InferenceServerException as e:
        logger.error(f"Failed to retrieve the metadata: {str(e)}")
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        logger.error(f"Failed to retrieve the config: {str(e)}")
        sys.exit(1)

    if FLAGS.protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(model_metadata, model_config)

    max_batch_size, input_name, output_name, spec_size, mfcc_size, audio_size, coc_size, conven_size, output_size ,dtype = parse_model(model_metadata, model_config)

    pipeline = DataPipeline()
    user_data = UserData()

    response = supabase_client.table('AI_QUEUE').select("*").execute()
    if response.status_code != 200:
        logger.error("Failed to fetch data from AI_QUEUE")
        sys.exit(1)

    response_datas = response.json().get("data", [])
    if not response_datas:
        logger.info("No data to process")
        sys.exit(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(handle_inference_response, response_data, pipeline, input_name, output_name, dtype, FLAGS, triton_client, user_data)
            for response_data in response_datas
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing inference: {str(e)}")

    logger.info('Inference Complete.')

if __name__ == "__main__":
    main()
