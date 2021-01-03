#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:35:20 2021

@author: kevjlope
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        help = 'Image webcam')
    parser.add_argument(
        '-m',
        '--model_file',
        help='.tflite model')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help = 'input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type = float,
        help = 'input standard deviation')
    args = parser.parse_args()
    interpreter = tf.lite.interpreter(model_path=args.model_file)
    interpreter.allocate_tensor()
    "Get input and output tensors"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    floating_model = input_details[0]['dtype'] == np.float32
    
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width,height))
    
    input_data = np.expand_dims(img, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data)-args.input_mean)/args.input_std
    interpreter.set_tensor(input_details[0]['index'],input_data)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    scalar_result = int(round(np.asscalar(results)))
    if(scalar_result):
        print('ATENCION: No se ha detectado mascarilla')