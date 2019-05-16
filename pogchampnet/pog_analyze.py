from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from IPython.display import HTML, display
import cv2
import numpy as np
from keras.models import load_model
from keras.models import Model
from pytube import YouTube
import matplotlib.pyplot as plt
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7


model = load_model('./saved_model.h5')
config = model.get_config()
config.gpu_options.allow_growth = True
model = Model.from_config(config)

def predict(frame):
    frame = cv2.resize(frame, (384, 216))
    # dividing by 255 leads to faster convergence through normalization.
    frame = np.array(frame)/(255)
    frame = np.expand_dims(frame, axis=0)
    r = model.predict(frame)
#     print(np.argmax(r[0]), r[0][np.argmax(r[0])] * 100)
    return np.argmax(r[0])



path_to_clip = './SoOn OW - MY REIN VS XQC 48.mp4'

video = cv2.VideoCapture(path_to_clip)
success, frame = video.read()
fps = int(video.get(cv2.CAP_PROP_FPS))
# video.set(1, 4500)
success, frame = video.read()
scores = []
frames = []
# plt.imshow(frame)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#out = display(progress(0, length), display_id=True)
while success:
    success, frame = video.read()
    if not success:
        break

    frame_num = int(video.get(1))
    print(frame_num)
    #out.update(progress(frame_num, length))
    if frame_num % 10 != 0:
        continue

    r = predict(frame)
    scores.append(r)
    frames.append(video.get(cv2.CAP_PROP_POS_MSEC)//1000)

    if cv2.waitKey(0) == ord('a'):
        break
video.release()

interest_index = []
window_size = 10
last_index = 0
for index in range(len(scores[:-window_size])):
  if sum(scores[index:index+window_size])/window_size > 6:
    if len(interest_index) == 0 or index-1 != last_index:
      interest_index.append(index)
    last_index = index
print('Neural network found ', len(interest_index), 'sequence(s)')

for index in interest_index:
  ffmpeg_extract_subclip(path_to_clip, max(0,frames[index]-5), frames[index]+25, targetname="./highlight"+str(frames[index])+".mp4")


import io
import base64
from IPython.display import HTML
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('./') if isfile(join('./', f))]
data = ''

for file in onlyfiles:
  if file.startswith('highlight'):
    print(file)
    files.download(file)