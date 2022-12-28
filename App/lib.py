import glob
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib inline
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from tqdm import tqdm

import cv2 as cv
import dlib
import threading
import time
from pygame import mixer

import tkinter as tk
from tkinter import filedialog
from tkinter import Label