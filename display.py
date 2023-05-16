import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import yaml
import math

from optotypes import *
from renderer import *
from utils import *

config_stream = open("config.yml", 'r')
config = yaml.load(config_stream, Loader=yaml.FullLoader)

DISPLAY_WIDTH = config['display']['width']
DISPLAY_HEIGHT = config['display']['height']

ASPECT_RATIO = config['display']['aspect_ratio'].split('_')
ASPECT_RATIO = [int(x) for x in ASPECT_RATIO]

# Display Dimensions in cm
DISPLAY_DIAG_DIM = config['display']['screen_size'] * 2.54
DISPLAY_WIDTH_DIM = ASPECT_RATIO[0] * math.sqrt(
    DISPLAY_DIAG_DIM**2 / (ASPECT_RATIO[0]**2 + ASPECT_RATIO[1]**2))
DISPLAY_HEIGHT_DIM = ASPECT_RATIO[1] * math.sqrt(
    DISPLAY_DIAG_DIM**2 / (ASPECT_RATIO[0]**2 + ASPECT_RATIO[1]**2))

corrects = {}
for logMAR in snellen_dict.keys():
    corrects[logMAR] = []
print(corrects)

# Models

import os
import sys
import json
# import tflite_runtime.interpreter as tflite
from kws_streaming.train import inference
import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd

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

if __name__ == '__main__':
    pygame.init()
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('Visual Acuity Test - Display')

    optotypes = load_optotypes()

    crashed = False
    cur_logMAR = 1
    cur_optotype = 3
    cur_pointed = 0
    update_screen = True
    update_optotypes = True

    predicted_speech = ''
    set_corrected = 0
    current_set_results = []


    test = TestScreen(list(optotypes.values())[cur_optotype], 2, 5, cur_logMAR, display)
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if cur_logMAR < 1:
                        cur_logMAR = round((cur_logMAR + 0.1) * 10) / 10
                        update_screen = True
                        update_optotypes = True
                if event.key == pygame.K_DOWN:
                    if cur_logMAR > -0.3:
                        cur_logMAR = round((cur_logMAR - 0.1) * 10) / 10 
                        update_screen = True
                        update_optotypes = True

                if event.key == pygame.K_LEFT:
                    if cur_optotype > 0:
                        cur_optotype -= 1
                        update_screen = True
                        update_optotypes = True
                if event.key == pygame.K_RIGHT:
                    if cur_optotype < len(optotypes) - 1:
                        cur_optotype += 1
                        update_screen = True
                        update_optotypes = True

                if event.key == pygame.K_SPACE:
                    update_screen = True
                    update_optotypes = True

                if event.key == pygame.K_a:
                    if cur_pointed > -1:
                        cur_pointed -= 1
                    update_screen = True
                if event.key == pygame.K_d:
                    if cur_pointed < len(optotypes) - 1:
                        cur_pointed += 1
                    else:
                        cur_pointed = 0
                    update_screen = True

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
                predicted_speech = labels[predicted_labels[0]]
            else:
                print(f"_silence_ peak={str(round(peak_loudness, 2))}")
                prev_prediction = ''
                predicted_speech = ''
        else:
            print(f"_silence_ peak={str(round(peak_loudness, 2))}")
            prev_prediction = ''
            predicted_speech = ''

        pointed_optotype = [x.name for x in test.line.figures][cur_pointed]
        if predicted_speech != '':
            if predicted_speech == pointed_optotype:
                set_corrected += 1
                current_set_results.append(1)
            else:
                current_set_results.append(0)
            cur_pointed += 1
            update_screen = True

        if cur_pointed >= len(optotypes):
            cur_pointed = 0
            if set_corrected >= 4:
                if cur_logMAR > -0.3:
                    corrects[cur_logMAR].append(set_corrected)
                    cur_logMAR = round((cur_logMAR - 0.1) * 10) / 10 
                    update_screen = True
                    update_optotypes = True
                    set_corrected = 0
            else:
                if cur_logMAR < 1:
                    corrects[cur_logMAR].append(set_corrected)
                    cur_logMAR = round((cur_logMAR + 0.1) * 10) / 10
                    update_screen = True
                    update_optotypes = True
                    set_corrected = 0
            set_corrected = 0
            current_set_results = []

        if update_screen:
            display.fill((255, 255, 255))
            if update_optotypes:
                test = TestScreen(list(optotypes.values())[cur_optotype], 2, 5, cur_logMAR, display)
                update_optotypes = False
            test.render(cur_pointed, current_set_results)
            update_screen = False

        logMARs = list(corrects.keys())
        logMARs.reverse()
        for logMAR in logMARs:
            
            if len(corrects[logMAR]) >= 2:
                if corrects[logMAR][-1] >= 4 and corrects[logMAR][-2] >= 4:
                    print(f'Result = logMAR {logMAR}')
                    exit(0)
        pygame.display.update()