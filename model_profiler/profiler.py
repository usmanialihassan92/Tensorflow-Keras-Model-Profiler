# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:45:33 2021

@author: Talha
"""

#from model_profiler.utils import get_available_gpus, get_param, keras_model_memory_usage, count_flops, mem_for_storing_weights
from model_profiler.utils import get_available_gpus, get_param, keras_model_memory_usage, count_flops, mem_for_storing_weights
from tensorflow.python.keras.engine.functional import Functional
import numpy as np
from tabulate import tabulate


Batch_size = 1
units = ['GPU IDs', 'BFLOPs', 'GB', 'Million', 'MB']

Profile = ['Selected GPUs', 'No. of FLOPs', 'GPU Memory Requirement',
           'Model Parameters', 'Memory Required by Model Weights']

flops = 0
mem = 0
param = 0
mem_req = 0

def model_profiler(model, Batch_size, profile=Profile, use_units=units, verbose=0):
    '''
    Parameters
    ----------
    model : a keras/tensorflow compiled or uncompiled model
    Batch_size : an int default to 1.
    Profile : a list of profile characterstics
    use_units : units for those characterstics
    verbose: whether to print out the model profile or not [verbose > 0] will 
            print out the profile
    Returns
    -------
    profile: a ordered pretty string table containing model profile 
            alos prints out the model profile

    '''
    gpus = get_available_gpus()
    ###
    # if you have used a keras/tf built in model than that model will be an object of
    # tensorflow/keras engine functional, meaning one functional layer will be packing
    # all the layers. So, we first need to look inside the functional layer
    ###
    global flops
    global mem
    global param
    global mem_req
    flops = 0
    mem = 0
    param = 0
    mem_req = 0

    recursive_model_calculate_flops(model,use_units)
    
    values = [gpus, flops, mem, param, mem_req]
    
    full_profile = np.concatenate((
                                np.asarray(Profile).reshape(-1,1),
                                np.asarray(values).reshape(-1,1),
                                np.asarray(use_units).reshape(-1,1)
                                )
                            , 1)
    profile = tabulate(
                    np.ndarray.tolist(full_profile),
                    headers = ["Model Profile", "Value", "Unit"],
                    tablefmt="github"
                    )
    if verbose > 0:
        print(profile)
    
    return profile, values

def recursive_model_calculate_flops(model,use_units):
    base_model = []
    for j, layer in enumerate(model.layers):
        if "functional" in str(layer) or "sequential" in str(layer):
            sub_model = model.layers[j]
            recursive_model_calculate_flops(sub_model,use_units)
        else:
            base_model.append(layer)
    global flops
    flops = flops + count_flops(use_units[1], base_model, Batch_size)
    global mem
    mem += keras_model_memory_usage(use_units[2], base_model, Batch_size)
    global param
    param += get_param(use_units[3], base_model)
    global mem_req
    mem_req += mem_for_storing_weights(use_units[4], base_model)
    return
