import cv2
from easydict import EasyDict as edict

CONFIG = edict()
 
CONFIG.MODE = 'V1'

CONFIG.JSON_PATH  = '../jsons/'
CONFIG.INPUT_PATH = '../input/'

#BGR 
CONFIG.CNN_COLOR     =  (0,255,0)
CONFIG.OUTPUT_DIR    =  "output/"
CONFIG.OCV_COLOR     =  (0,0,255)   
CONFIG.TRK_COLOR     =  (255,0,0)
CONFIG.HRATIO        =  1#frame.shape[0] / HEIGHT
CONFIG.WRATIO        =  1#frame.shape[1] / WIDTH
