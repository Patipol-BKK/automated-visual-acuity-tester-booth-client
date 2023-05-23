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

if config['testing']['type'] == 'Single Eye Both Sides':
    TEST_NUM = 2
else:
    TEST_NUM = 1

OPTOTYPES_NUM = config['testing']['optotypes_num']

# Display Dimensions in cm
DISPLAY_DIAG_DIM = config['display']['screen_size'] * 2.54
DISPLAY_WIDTH_DIM = ASPECT_RATIO[0] * math.sqrt(
    DISPLAY_DIAG_DIM**2 / (ASPECT_RATIO[0]**2 + ASPECT_RATIO[1]**2))
DISPLAY_HEIGHT_DIM = ASPECT_RATIO[1] * math.sqrt(
    DISPLAY_DIAG_DIM**2 / (ASPECT_RATIO[0]**2 + ASPECT_RATIO[1]**2))

DISTANCE = int(config['testing']['distance'])

corrects = {}
for logMAR in snellen_dict.keys():
    corrects[logMAR] = []

print('Starting...')

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

import cv2
from ffpyplayer.player import MediaPlayer
import threading

def play_intro():
    file_name = "intro.mp4"
    window_name = "window"
    interframe_wait_ms = 23

    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    player = MediaPlayer(file_name)

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while (True):
        ret, frame = cap.read()
        audio_frame, val = player.get_frame()
        if not ret:
            print("Reached end of video, exiting.")
            break

        cv2.imshow(window_name, frame)
        if cv2.waitKey(interframe_wait_ms) & 0x7F == ord('q'):
            print("Exit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()

intro_player = threading.Thread(target=play_intro, args=())

print('Loading Tensorflow Model...')

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

print('Using model:', flags.model_name)

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
    display = pygame.display.set_mode((0,0),pygame.FULLSCREEN)
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

    test = TestScreen(list(optotypes.values())[cur_optotype], DISTANCE, config['testing']['optotypes_num'], cur_logMAR, display)
    msg_renderer = DisplayScreen(DISTANCE, display)
    option_renderer = OptionScreen(display)

    state = 'idle'
    option_selected_index = 0
    option_selection_input = 'idle'

    restart = False
    while not crashed:
        for event in pygame.event.get():
            # Check if user hit escape for closing the program
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    crashed = True
                    pygame.display.quit()
                    pygame.quit()

            # Check if program wants to exit
            if event.type == pygame.QUIT or crashed:
                    crashed = True
                    break

            # Check for user selection 'o' for going into options or proceeding with the test if in main screen
            if state == 'idle':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        state = 'option'
                    else:
                        state = 'intro'

            if  state == 'result':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        state = 'option'
                    else:
                        state = 'idle'

            # Key selection for option screen
            elif state == 'option':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        if option_selected_index > 0:
                            option_selected_index -= 1
                            update_screen = True
                    if event.key == pygame.K_DOWN:
                        if option_selected_index < len(option_renderer.options) + 1:
                            option_selected_index += 1
                            update_screen = True

                    if event.key == pygame.K_LEFT:
                        option_selection_input = 'left'
                        update_screen = True
                    if event.key == pygame.K_RIGHT:
                        option_selection_input = 'right'
                        update_screen = True

                    if event.key==pygame.K_RETURN:
                        if option_selected_index == len(option_renderer.options):
                            option_selected_index = 0
                            option_selection_input = 'idle'
                            state = 'idle'
                            update_screen = True
                        elif option_selected_index == len(option_renderer.options) + 1:
                            restart = True
                            crashed = True
                            pygame.display.quit()
                            pygame.quit()
                            option_renderer.save()
                            continue

            # Key selection in testing screen
            elif state == 'test':
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

        # Check if program is closing
        if crashed:
            break

        # Render main screen
        if state == 'idle':
            msg_renderer.render('Visual Acuity Testing Booth', 'Please put on the goggles to start')
            pygame.display.update()

        # Render option screen
        elif state == 'option':
            option_renderer.render(option_selected_index, option_selection_input)
            option_selection_input = 'idle'
            pygame.display.update()

        # Start intro video
        elif state == 'intro':
            display.fill((255, 255, 255))
            pygame.display.update()
            if config['run_intro']:
                intro_player.start()
                intro_player.join()
                state = 'test'
                wav_file = "beep.mp3"
                player = MediaPlayer(wav_file)
                while True:
                    frame, val = player.get_frame()
                    if val != 'eof' and frame is not None:
                        # Show the frame or perform other tasks if needed
                        pass
                    else:
                        # Audio playback is finished
                        break
            else:
                state = 'test'

        # Render testing screen
        elif state == 'test':

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
                if confidence > 4 and labels[predicted_labels[0]] != '_silence_' and labels[predicted_labels[0]] != '_unknown_' and labels[predicted_labels[0]] != prev_prediction:
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
                if set_corrected >= OPTOTYPES_NUM - 1:
                    if set_corrected >= OPTOTYPES_NUM and cur_logMAR >= 0.2:
                        corrects[cur_logMAR].append(set_corrected)
                        cur_logMAR = round((cur_logMAR - 0.2) * 10) / 10 
                        update_screen = True
                        update_optotypes = True
                        set_corrected = 0

                    elif cur_logMAR > -0.3:
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
                
                test.render(len(optotypes) - 1, current_set_results)
                
                pygame.display.update()
                pygame.time.delay(500)

                update_screen = True
                cur_pointed = 0
                set_corrected = 0
                current_set_results = []

            if update_screen:
                display.fill((255, 255, 255))
                if update_optotypes:
                    test = TestScreen(list(optotypes.values())[cur_optotype], DISTANCE, OPTOTYPES_NUM, cur_logMAR, display)
                    update_optotypes = False
                test.render(cur_pointed, current_set_results)
                # test.render_result(logMAR)
                update_screen = False

            logMARs = list(corrects.keys())
            logMARs.reverse()
            for logMAR in logMARs:
                
                if len(corrects[logMAR]) >= 2:
                    if corrects[logMAR][-1] >= OPTOTYPES_NUM - 1 and corrects[logMAR][-2] >= OPTOTYPES_NUM - 1:
                        state = 'result'
        elif state == 'result':
            msg_renderer.render('Testing Results', f'LogMAR : {logMAR} | Snellen : {snellen_dict[logMAR]}')
            pygame.display.update()
        pygame.display.update()

    print('Exiting...')

    if restart:
        os.system('python main.py')