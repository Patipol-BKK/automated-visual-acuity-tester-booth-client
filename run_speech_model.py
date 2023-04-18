import os
import sys
import json
# import tflite_runtime.interpreter as tflite
from kws_streaming.train import inference
import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd

# args = sys.argv[1:]
# model_name = args[0]    
model_name = 'ds_tc_resnet_numbers_longtrain'

tflite_models_folder_path = os.path.join('tflite_models', model_name)
model_path = os.path.join(tflite_models_folder_path, 'model.tflite')

flags_path = os.path.join(tflite_models_folder_path, 'flags.json')
labels_path = os.path.join(tflite_models_folder_path, 'labels.txt')

with open(flags_path, 'r') as fd:
  flags_json = json.load(fd)

class DictStruct(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)

flags = DictStruct(**flags_json)

print(flags.model_name)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_states = []
for s in range(len(input_details)):
  input_states.append(np.zeros(input_details[s]['shape'], dtype=np.float32))

with open(labels_path, 'r') as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

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

        interpreter.set_tensor(input_details[0]['index'], signal_buffer_conditioned.astype(np.float32))

        # run inference
        interpreter.invoke()

        # get output: classification
        predictions = interpreter.get_tensor(output_details[0]['index'])

        predicted_labels = np.argmax(predictions, axis=1)

        # predictions = model_non_stream.predict(signal_buffer_conditioned)
        # predicted_labels = np.argmax(predictions, axis=1)
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
        