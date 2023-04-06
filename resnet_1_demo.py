import sys
import os
import tarfile
import urllib
import zipfile
import librosa
sys.path.append('./kws_streaming')
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# TF streaming
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.models import model_utils
from kws_streaming.layers.modes import Modes

import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf1
import logging
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.train import inference
from kws_streaming.train import test
from kws_streaming.data import input_data
from kws_streaming.data import input_data_utils as du
tf1.disable_eager_execution()

config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf1.Session(config=config)

# general imports
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import scipy as scipy
import scipy.io.wavfile as wav
import scipy.signal
import soundfile as sf
import sounddevice as sd

tf1.reset_default_graph()
sess = tf1.Session()
tf1.keras.backend.set_session(sess)
tf1.keras.backend.set_learning_phase(0)

current_dir = os.getcwd()
MODEL_NAME = 'ds_tc_resnet_numbers'
# MODEL_NAME = 'svdf'
MODELS_PATH = os.path.join(current_dir, "trained_models")
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME + "/")

train_dir = os.path.join(MODELS_PATH, MODEL_NAME)

with tf.compat.v1.gfile.Open(os.path.join(train_dir, 'flags.json'), 'r') as fd:
  flags_json = json.load(fd)

class DictStruct(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)

flags = DictStruct(**flags_json)

model_non_stream_batch = models.MODELS[flags.model_name](flags)
weights_name = 'best_weights'
model_non_stream_batch.load_weights(os.path.join(train_dir, weights_name))
model_non_stream = utils.to_streaming_inference(model_non_stream_batch, flags, Modes.NON_STREAM_INFERENCE)

# tf.keras.utils.plot_model(
#     model_non_stream_batch,
#     show_shapes=True,
#     show_layer_names=True,
#     expand_nested=True)
print(flags.sample_rate)
print(flags.desired_samples)

with open(os.path.join(train_dir, 'labels.txt'), 'r') as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]
 # = ['unknown', 'unknown', 'up', 'down', 'left', 'right', 'unknown']

rms_buffer = [0.1]*20
signal_buffer = np.zeros(int(flags.desired_samples * 1.5))
prev_prediction = ''
while True:
    # Record audio
    signal = sd.rec(flags.desired_samples, samplerate=flags.sample_rate, channels=1)
    sd.wait()
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1, dtype=signal.dtype)
    signal_buffer = signal_buffer[flags.desired_samples:]
    signal_buffer = np.concatenate((signal_buffer, signal))
    start = -1
    window_rms = 0
    for window_start in range(0, len(signal_buffer), int(0.2 * flags.sample_rate)):
        window_rms = np.average(librosa.feature.rms(y=signal_buffer[window_start:int(window_start + 0.2 * flags.sample_rate)]))
        if window_rms > np.average(rms_buffer) * 2:
            start = window_start
            break
        else:
            rms_buffer = rms_buffer[1:]
            rms_buffer.append(window_rms)
    loudness = np.average(librosa.feature.rms(y=signal_buffer))
    peak_loudness = np.max(rms_buffer + [window_rms])

    if start != -1:
        if len(signal_buffer[start:]) >= flags.desired_samples:
            signal_buffer_conditioned = signal_buffer[start:start+flags.desired_samples]
        else:
            signal_buffer_conditioned = np.pad(signal_buffer[start:], (0, flags.desired_samples-len(signal_buffer[start:])), 'constant')
        signal_buffer_conditioned = np.array([signal_buffer_conditioned])
        predictions = model_non_stream.predict(signal_buffer_conditioned)
        predicted_labels = np.argmax(predictions, axis=1)
        confidence = round(predictions[0][predicted_labels[0]], 2)
        if confidence > 4 and labels[predicted_labels[0]] != '_silence_' and labels[predicted_labels[0]] != prev_prediction:
            print(f"{labels[predicted_labels[0]]} confidence={str(confidence)}")
            prev_prediction = labels[predicted_labels[0]]
        else:
            print(f"_silence_ peak={str(round(peak_loudness, 2))}")
            prev_prediction = ''
    else:
        print(f"_silence_ peak={str(round(peak_loudness, 2))}")
        prev_prediction = ''
        