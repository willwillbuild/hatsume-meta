from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from IPython.display import HTML, display
import cv2
import numpy as np
from keras.models import load_model
from pytube import YouTube
#from google.colab import files
import matplotlib.pyplot as plt
import tensorflow as tf

device_name = tf.test.gpu_device_name()

print('Found GPU at: {}'.format(device_name))


def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

# https://www.youtube.com/watch?v=56PbKMl3ZVM&t=199s
youtube_url = input('Enter the youtube URL of the video you want to use : ')

YouTube(youtube_url).streams.first().download('./')

