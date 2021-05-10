#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:56:34 2017
This is a collection of tools used by many other functions in this package
@author: lynnsorensen
"""

import numpy as np
import pickle
import keras

#  I/O: These have been replaced by joblib.


def save_pickle(obj, name):
    try:
        filename = open(name + ".pickle","wb")
        pickle.dump(obj, filename)
        filename.close()
        return(True)
    except:
        return(False)


def load_pickle(filename):
    return pickle.load(open(filename, "rb"))


def getLayerIndexByName(model, layername):
    """
    from https://stackoverflow.com/questions/50151157/keras-how-to-get-layer-index-when-already-know-layer-name
    """
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
    
#