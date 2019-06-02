import csv
import cv2
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def main():
    with open(sys.argv[2]) as label_file:
        labels_csv = csv.writer(label_file, delimiter=',')
        

    video_name = sys.argv[1]
    gaming_vid = cv2.VideoCapture(video_name)
    success, img = gaming_vid.read()


if __name__ == "__main__":
    main()