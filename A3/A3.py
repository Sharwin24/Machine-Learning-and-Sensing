import microphones
from keras.models import load_model
import tensorflow as tf
import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
import pyaudio
from pathlib import Path
import time
import argparse
import wget
import os
from reprint import output
from helpers import Interpolator, ratio_to_db, dbFS, rangemap
import matplotlib.pyplot as plt


# thresholds
PREDICTION_THRES = 0.8  # confidence
DBLEVEL_THRES = -40  # dB

# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Default to mono; will check device capability below
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0
OUTPUT_LINES = 36

###########################
# Model download
###########################


def download_model(url, output):
    return wget.download(url, output)


###########################
# Check Microphone
###########################
print("=====")
print("1 / 2: Checking Microphones... ")
print("=====")

desc, mics, indices = microphones.list_microphones()
if (len(mics) == 0):
    print("Error: No microphone found.")
    exit()

#############
# Read Command Line Args
#############
MICROPHONE_INDEX = 11

# Find description that matches the mic index
mic_desc = ""
for k in range(len(indices)):
    i = indices[k]
    if (i == MICROPHONE_INDEX):
        mic_desc = mics[k]
print("Using mic: %s" % mic_desc)

# Check the supported input channels for the selected device
p = pyaudio.PyAudio()
try:
    device_info = p.get_device_info_by_index(MICROPHONE_INDEX)
    max_input_channels = int(device_info.get('maxInputChannels', 1))
    supported_rate = int(device_info.get('defaultSampleRate', RATE))
    if max_input_channels < 1:
        print("Selected device does not support input channels.")
        exit()
    if CHANNELS > max_input_channels:
        print(
            f"Selected device only supports {max_input_channels} channel(s). Setting CHANNELS = {max_input_channels}")
        CHANNELS = max_input_channels
    if RATE != supported_rate:
        print(
            f"Selected device does not support {RATE}Hz. Using supported rate {supported_rate}Hz instead.")
        RATE = supported_rate
except Exception as e:
    print(f"Could not get device info: {e}")
    exit()
p.terminate()

###########################
# Download model, if it doesn't exist
###########################
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
print("=====")
print("2 / 2: Checking model... ")
print("=====")
model_filename = "models/example_model.hdf5"
ubicoustics_model = Path(model_filename)
if (not ubicoustics_model.is_file()):
    print("Downloading example_model.hdf5 [867MB]: ")
    download_model(MODEL_URL, MODEL_PATH)

# Load the ML model from A2/models
A2_MODEL_PATH = "/home/sharwin/spring_2025/CS496/A2/models/best_small_train_model.h5"
A2_MODEL = Path(A2_MODEL_PATH)
if not A2_MODEL.is_file():
    raise FileNotFoundError(f"Could not find A2 model at {A2_MODEL_PATH}")

##############################
# Load Deep Learning Model
##############################
print("Using deep learning model: %s" % (model_filename))
model = load_model(model_filename, compile=False)
context = ubicoustics.everything
label = {k: context[k] for k in range(len(context))}

# Load A2 Model
A2_model = load_model(str(A2_MODEL), compile=False)

##############################
# Setup Audio Callback
##############################
output_lines = []*OUTPUT_LINES
audio_rms = 0
candidate1 = ("-", 0.0)
candidate2 = ("-", 0.0)

# Prediction Interpolators
NUM_INTERPOLATORS = 31  # 30 for classes + 1 for dB
interpolators = [Interpolator() for _ in range(NUM_INTERPOLATORS)]

# Real-time Waveform setup
plt.ion()
plt.ion()
fig, (ax_wave, ax_bar1, ax_bar2) = plt.subplots(
    nrows=1, ncols=3, figsize=(12, 4),
    gridspec_kw={"width_ratios": [1, 1, 1]}
)
# — Waveform axis (left) —
ax_wave.set_title('Real-time Waveform')
ax_wave.set_xlim(0, CHUNK)
ax_wave.set_ylim(-1, 1)
ax_wave.set_xlabel('Sample Index')
ax_wave.set_ylabel('Amp')
line_wave, = ax_wave.plot(np.zeros(CHUNK), lw=1)

# — Bar chart for “model” (middle) —
ax_bar1.set_title("Model 1 Probs")
ax_bar1.set_xlabel("Probability")
ax_bar1.set_xlim(0, 1.0)
bar_container1 = ax_bar1.barh(
    list(range(len(context))), [0.0]*len(context),
    align='center'
)
ax_bar1.set_yticks(range(len(context)))
ax_bar1.set_yticklabels(
    [ubicoustics.to_human_labels[label[k]] for k in range(len(context))],
    fontsize=6
)
ax_bar1.invert_yaxis()  # so class 0 is on top

# — Bar chart for “A2_model” (right) —
ax_bar2.set_title("A2 Model Probs")
ax_bar2.set_xlabel("Probability")
ax_bar2.set_xlim(0, 1.0)
bar_container2 = ax_bar2.barh(
    list(range(len(context))), [0.0]*len(context),
    align='center'
)
ax_bar2.set_yticks(range(len(context)))
ax_bar2.set_yticklabels(
    [ubicoustics.to_human_labels[label[k]] for k in range(len(context))],
    fontsize=6
)
ax_bar2.invert_yaxis()

# Global buffers for latest waveform + probabilities + inference times
waveform = np.zeros(CHUNK)
latest_probs1 = np.zeros(len(context))
latest_probs2 = np.zeros(len(context))
inference_time1 = 0.0  # ms
inference_time2 = 0.0  # ms


def audio_samples(in_data, frame_count, time_info, status_flags):
  # Audio Input Callback
    global output_lines
    global interpolators
    global audio_rms
    global waveform, latest_probs1, latest_probs2
    global inference_time1, inference_time2
    global candidate1, candidate2

    # Update waveform data and Convert to [-1.0, +1.0]
    np_wav = np.frombuffer(in_data, dtype=np.int16) / 32768.0
    waveform = np_wav.copy()

    # Compute RMS and convert to dB
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)
    interp = interpolators[NUM_INTERPOLATORS - 1]
    interp.animate(interp.end, db, 1.0)

    # Make Predictions
    x = waveform_to_examples(np_wav, RATE)
    if x.shape[0] != 0:
        x = x.reshape(len(x), 96, 64, 1)
        # — Model 1 prediction and timing —
        t0 = time.perf_counter()
        raw1 = model.predict(x, verbose=0)   # raw1.shape == (1, 30)
        inference_time1 = (time.perf_counter() - t0) * 1000.0
        flat1 = raw1[0]                      # now flat1 has shape (30,)
        latest_probs1 = flat1.copy()

        # Find the winning class for Model 1
        m1 = np.argmax(flat1)                # integer between 0 and 29
        candidate1 = (
            ubicoustics.to_human_labels[label[m1]],
            float(flat1[m1])
        )

        # Animate each of the 30 class interpolators with flat1[k]
        for k in range(NUM_INTERPOLATORS - 1):
            interpolators[k].animate(interpolators[k].end, flat1[k], 1.0)

        # — Model 2 (A2_model) prediction and timing —
        t0 = time.perf_counter()
        raw2 = A2_model.predict(x, verbose=0)  # raw2.shape == (1, 30)
        inference_time2 = (time.perf_counter() - t0) * 1000.0
        flat2 = raw2[0]                        # shape (30,)
        latest_probs2 = flat2.copy()

        # Winning class for A2_model
        m2 = np.argmax(flat2)
        candidate2 = (
            ubicoustics.to_human_labels[label[m2]],
            float(flat2[m2])
        )
    return (in_data, pyaudio.paContinue)


##############################
# Main Execution
##############################
while True:
    ##############################
    # Setup Audio
    ##############################
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
                    stream_callback=audio_samples, input_device_index=MICROPHONE_INDEX)

    ##############################
    # Start Non-Blocking Stream
    ##############################
    os.system('cls' if os.name == 'nt' else 'clear')
    print("# Live Prediction Using Microphone: %s" % (mic_desc))
    stream.start_stream()
    while stream.is_active():
        with output(initial_len=OUTPUT_LINES, interval=0) as output_lines:
            while True:
                time.sleep(1.0/FPS)  # 60fps
                # — Update the 30 class bars for Model 1 in console (using interpolators) —
                for k in range(30):
                    interp_k = interpolators[k]
                    val = interp_k.update()
                    bar = "|" * int(val * 100.0)
                    output_lines[k] = f"{ubicoustics.to_human_labels[label[k]]:20s}: {val:.2f} {bar}"

                # — dB level (index 30) —
                interp_db = interpolators[NUM_INTERPOLATORS - 1].update()
                db_val = rangemap(interp_db, -50, 0, 0, 100)
                db_val = max(0, min(100, db_val))
                bar_db = "|" * int(db_val)
                output_lines[30] = f"{'Audio Level':20s}: {interp_db:.1f}dB [{bar_db}]"

                # — Thresholds (index 31) —
                output_lines[31] = f"{'Thresholds':20s}: conf={PREDICTION_THRES:.2f}, db_th={DBLEVEL_THRES:.1f}"

                # — Inference times (index 32 & 33) —
                output_lines[32] = f"{'Model 1 Time':20s}: {inference_time1:6.2f} ms"
                output_lines[33] = f"{'A2 Model Time':20s}: {inference_time2:6.2f} ms"

                # — Final Preds (index 34 & 35) —
                pred_text1 = "-"
                if candidate1[1] > PREDICTION_THRES and interp_db > DBLEVEL_THRES:
                    pred_text1 = candidate1[0].upper()
                output_lines[34] = f"{'Model 1 Pred':20s}: {pred_text1}"

                pred_text2 = "-"
                if candidate2[1] > PREDICTION_THRES and interp_db > DBLEVEL_THRES:
                    pred_text2 = candidate2[0].upper()
                output_lines[35] = f"{'A2 Model Pred':20s}: {pred_text2}"

                # — Update waveform plot (left) —
                line_wave.set_ydata(waveform)
                ax_wave.draw_artist(line_wave)

                # — Update Model 1 bar chart (middle) —
                ax_bar1.clear()
                ax_bar1.set_title("Model 1 Probs")
                ax_bar1.set_xlim(0, 1.0)
                ax_bar1.barh(
                    np.arange(len(latest_probs1)),
                    latest_probs1,
                    align='center'
                )
                ax_bar1.set_yticks(np.arange(len(latest_probs1)))
                ax_bar1.set_yticklabels(
                    [ubicoustics.to_human_labels[label[k]]
                        for k in range(len(latest_probs1))],
                    fontsize=6
                )
                ax_bar1.invert_yaxis()
                ax_bar1.grid(False)

                # — Update A2 model bar chart (right) —
                ax_bar2.clear()
                ax_bar2.set_title("A2 Model Probs")
                ax_bar2.set_xlim(0, 1.0)
                ax_bar2.barh(
                    np.arange(len(latest_probs2)),
                    latest_probs2,
                    align='center'
                )
                ax_bar2.set_yticks(np.arange(len(latest_probs2)))
                ax_bar2.set_yticklabels(
                    [ubicoustics.to_human_labels[label[k]]
                        for k in range(len(latest_probs2))],
                    fontsize=6
                )
                ax_bar2.invert_yaxis()
                ax_bar2.grid(False)

                # — Redraw the entire figure —
                fig.canvas.draw_idle()
                plt.pause(0.001)
