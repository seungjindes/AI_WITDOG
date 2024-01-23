import base64
import json
import numpy as np


import torch 
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils


def score_to_logit(score):
    score = torch.tensor(score)
    logit = F.softmax(score, dim=1)
    return logit

def wrap_json(logit):
    """
    1. tensor logit to list 
    2. make json output 
    """
    obj = {
        "result": np.array(logit).tolist()
    }
    return json.dumps(obj, ensure_ascii=False)

# def vector2cate(logit):
#     emot_map = {'hap': 0, 'sad':1, 'ang':2, 'fea':3}
#     value =torch.argmax(logit)
#     if value == 0:
#         return 'hap'
#     elif value== 1:
#         return 'sad'
#     elif value == 2:
#         return 'ang'
#     elif value == 3:
#         return 'fea'



class TritonPythonModel:
    """
    post processing 

    """

    def execute(self , requests):
        responses = [] 
        for request in requests: 
            predict_score = pb_utils.get_input_tensor_by_name(
                request, "INPUT__0"
            ).as_numpy()

        logit = score_to_logit(predict_score)
        #emotion = vector2cate(logit)
        response = np.array(wrap_json(logit) , dtype = np.object_)
        response = pb_utils.Tensor("result" , response)
        response = pb_utils.inferenceResponse(output_tensors=[response])
        responses.append(response)

        return responses


