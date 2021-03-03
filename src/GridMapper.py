'''
Created on 23 Mar 2020

@author: Mark Basham
'''
import os
import pickle
import sys

import numpy as np

from matplotlib.pyplot import imread, imsave
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_features(input_dir):
    images = []
    for im in os.listdir(input_dir):
        # Just append the red channel as this is a grayscale image
        images.append(imread(os.path.join(input_dir,im))[:,:,0])
    
    features = np.dstack(images)
    return features

def get_parameter(input_file):
    return imread(input_file)[:,:,0]

def save_parameter(parameter, save_file):
    print(parameter.max())
    print(parameter.min())
    print(parameter.shape)
    print(np.isnan(parameter))
    print(parameter)
    safe = np.copy(parameter)
    safe[safe<0] = 0
    safe[safe>255] = 255
    safe = np.nan_to_num(safe, 255)
    safe = safe.astype(np.uint8)
    print(safe.max())
    print(safe.min())
    print(safe.shape)
    print(np.isnan(safe))
    print(safe)
    print(save_file)
    
    imsave(save_file, safe, cmap='gray', vmin=0, vmax=255)

def train_and_save_predictor(features, parameter, model_file):
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    #reshapes happen here to flatten out the image dimentions, we are intersted only in a list of features and properties.
    reg.fit(features.reshape(-1, features.shape[-1]), parameter.reshape(-1))
    with open(model_file, 'wb') as f:
        pickle.dump(reg, f)

def load_and_apply_prediction(features, model_file):
    reg = None
    with open(model_file, 'rb') as f:
        reg = pickle.load(f)
    # we need to flatten out the image info before predicting
    result = reg.predict(features.reshape(-1, features.shape[-1]))
    #but add it back when returning
    result = result.reshape(features.shape[0:2])
    return result
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dir", help="The directory of the grid you wish to process")
    parser.add_argument('-m','--model_file', dest='model_file', default='model.mod', help='ML model file')
    parser.add_argument('-p','--parameter_file', dest='parameter_file', default=None, help='Parameter file to train against')
    parser.add_argument('-o','--output_file', dest='output_file', default=None, help='create a predicted parameter file and save it here')
    args = parser.parse_args()
    
    print('loading features')
    features = get_features(args.feature_dir)
    
    if args.parameter_file is not None:
        print('training model')
        parameter = get_parameter(args.parameter_file)
        train_and_save_predictor(features, parameter, args.model_file)
        sys.exit(0)
    
    print('estimating parameters')
    prediction = load_and_apply_prediction(features, args.model_file)
    save_parameter(prediction, args.output_file)
    sys.exit(0)
