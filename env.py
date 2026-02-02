#start env
# imports
import gym_super_mario_bros
import gym
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np
import cv2
import subprocess as sp
import os

#monitor class--> allows us to record, save, and monitor moves of our agent 
class Monitor:
    def __init__(self, width, height, saved_path):
        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

# check if frame is in the correct format for RL agent
def process_frame(frame):
    if isinstance(frame, np.ndarray) and frame.size > 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))

