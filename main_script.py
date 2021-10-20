import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re
import cv2
import argparse

#BGR 
cnn_color     =  (0,255,0)
opencv_color  =  (0,0,255)   
tracker_color =  (255,0,0)
HRATIO        =  1#frame.shape[0] / HEIGHT
WRATIO        =  1#frame.shape[1] / WIDTH
fourcc        =  cv2.VideoWriter_fourcc(*'XVID')

parser = argparse.ArgumentParser()
parser.add_argument("--name")
value = parser.parse_args()


# name = files.split('.')[0]
name = value.name
print("Creating output ", name+'.mp4')
video_path = 'input/'+name+'.mp4'
json_path  = 'jsons/'+name+'.json'

f = open(json_path).readlines()
data = json.loads('{'+ f[0][10:-2])
# data = json.load(f)
new_df = pd.DataFrame(columns=['FRAME_COUNT', 'OPEN_BOX', 'CNN_BOX','Vertical', 'SPEED', 'POSE', 'SUI', 'SPLASH', 'ACTIND', 'TBBOX', 'ALARM'])


ind = 0
for key in data.keys():
    D = data[key]
    T = json.loads(D['TRACKERS_DATA'])
    for key in list(T.keys()):
        T = json.loads(D['TRACKERS_DATA'])[key]
        new_df.loc[ind] = [str(D['FRAME_COUNT'])] +  [str(D['OPEN_BOX'])] + [str( D['CNN_BOX'])] + [str(T['VERTICAL'])] + [str(T['SPEED'])] + [str(T['POSE'])] + \
                        [str(T['SUI'])] + [str(T['SPLASH'])] + [str(T['ACTIND'])] + [str(T['BBOX'])] + [str(T['ALARM'])]

        ind += 1
    if len(T.keys())==0:
        new_df.loc[ind] = [str(D['FRAME_COUNT'])] +  [str(D['OPEN_BOX'])] + [str( D['CNN_BOX'])]\
                        + [""] + [""] +  [""]  +  [""] + [""] + [""] + [""]+ [""]
        ind += 1
        
del ind

THRESHOLD = 100
new_df['Vertical'] = pd.to_numeric(new_df['Vertical']).clip(upper=THRESHOLD)
new_df['SPEED']    = pd.to_numeric(new_df['SPEED']).clip(upper=THRESHOLD)
new_df['POSE_50']  = pd.to_numeric(new_df['POSE']) * 50 
new_df['SUI_50']   = pd.to_numeric(new_df['SUI']) * 50
new_df['SPLASH']   = pd.to_numeric(new_df['SPLASH']).clip(upper=THRESHOLD)
new_df['ACTIND']   = pd.to_numeric(new_df['ACTIND']).clip(upper=THRESHOLD)

new_df = new_df.reset_index()

video = cv2.VideoCapture(video_path)
analysis_feed  = cv2.VideoWriter("AnalysisOutput/"+ name+ ".mp4",fourcc, 5, (1280,720*2))
FRAME_COUNT = 0
try:
    while video.isOpened():
        FRAME_COUNT += 1
        ret_val, frame = video.read()
        centroid = None
        ## OPENCV 
        arr = new_df.iloc[FRAME_COUNT, 2]
        if len(arr)>3:
            temp = re.findall(r'\d+', arr)
            temp = [t for t in temp if len(t)<=4]
            temp = list(map(int, temp))

            for ind in range(len(temp)//4):
                ind = ind * 4
                bbox = temp[ind] , temp[ind+1] , temp[ind+2] , temp[ind+3] 
                start_point = (  int(bbox[0]*WRATIO ) , int(bbox[1] * HRATIO)  )
                end_point = (int(bbox[2]*WRATIO), int(bbox[3]* HRATIO ) )
                image = cv2.rectangle(frame, start_point, end_point, opencv_color, 1)
                centroid = ( int(bbox[0] +(bbox[2]-bbox[0])/2), int(bbox[1] + (bbox[3]-bbox[1])/2))
                centroid = (int(centroid[0] * HRATIO) , int(centroid[1] * WRATIO))
                image = cv2.circle(frame, centroid, 5, opencv_color, 2)

        ## CNN 
        arr = new_df.iloc[FRAME_COUNT, 3]
        if len(arr)>3:
            temp = re.findall(r'\d+', arr)
            temp = [t for t in temp if len(t)<=4]
            temp = list(map(int, temp))
            for ind in range(len(temp)//4):
                ind = ind * 4
                bbox = temp[ind] , temp[ind+1] , temp[ind+2] , temp[ind+3]
                start_point = (  int(bbox[0]*WRATIO ) , int(bbox[1] * HRATIO)  )
                end_point = (int(bbox[2]*WRATIO), int(bbox[3]* HRATIO ) )
                image = cv2.rectangle(frame, start_point, end_point, cnn_color, 1)
                centroid = ( int(bbox[0] +(bbox[2]-bbox[0])/2), int(bbox[1] + (bbox[3]-bbox[1])/2))
                centroid = (int(centroid[0] * HRATIO) , int(centroid[1] * WRATIO))
                image = cv2.circle(frame, centroid, 5, cnn_color, 2)

        ## TRACKER 
        arr = new_df.iloc[FRAME_COUNT, 10]
        if len(arr)>3:
            temp = re.findall(r'\d+', arr)
            temp = [t for t in temp if len(t)<=4]
            temp = list(map(int, temp))
            for ind in range(len(temp)//4):
                ind = ind * 4
                bbox = temp[ind] , temp[ind+1] , temp[ind+2] , temp[ind+3] 
                start_point = (  int(bbox[0]*WRATIO ) , int(bbox[1] * HRATIO)  )
                end_point = (int(bbox[2]*WRATIO), int(bbox[3]* HRATIO ) )
                image = cv2.rectangle(frame, start_point, end_point, tracker_color, 1)
                centroid = ( int(bbox[0] +(bbox[2]-bbox[0])/2), int(bbox[1] + (bbox[3]-bbox[1])/2))
                centroid = (int(centroid[0] * HRATIO) , int(centroid[1] * WRATIO))
                image = cv2.circle(frame, centroid, 5, tracker_color, 2)



        plot  = new_df[FRAME_COUNT-20:FRAME_COUNT+20].plot("index", [ 'Vertical', 'SPEED', 'POSE_50', 'SUI_50', 'SPLASH', 'ACTIND'], ylim=(0,110), xlabel='FRAME COUNT',figsize=(10, 6))

        vertical = new_df.iloc[FRAME_COUNT,4]
        SPEED    = 0
        POSE     = 0
        SUI      = 0
        SPLASH   = 0
        ACTIND   = 0
        DROWN    = False
        if str(vertical)!='nan':
            SPEED  = new_df.iloc[FRAME_COUNT,5] 
            POSE   = new_df.iloc[FRAME_COUNT,6]
            SUI    = new_df.iloc[FRAME_COUNT,7]
            SPLASH = new_df.iloc[FRAME_COUNT,8]
            ACTIND = new_df.iloc[FRAME_COUNT,9]
            DROWN  = new_df.iloc[FRAME_COUNT,10]
        else:
            vertical = 0


        color  = (0, 255, 0)
        fontscale = 0.5
        thickness = 1
        image = cv2.putText(frame, 'Vertical :' + str(vertical)[:5] , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, 'SPEED   :' + str(SPEED)[:5]  , (20, 70), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, 'POSE    :' + str(POSE)[:5]  , (20, 110), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, 'SUI      :' + str(SUI)[:5]  , (20, 150), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color,  thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, 'SPLASH  :' + str(SPLASH)[:5]  , (20, 190), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, 'ACTIND   :' + str(ACTIND)[:5]  , (20, 240), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, 'DROWN   :' + str(DROWN)[:5]  , (640, 80), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, 'FRAME   :' + str(FRAME_COUNT)  , (640, 130), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)


        image = cv2.putText(frame, "HS_min  = 15  "  , (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, "PO_min  = 1   "  , (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, "SP_max  = 150 "  , (1000, 110), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, "VS_min  = 15  "  , (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color,  thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, "AI_min  = 3.5 "  , (1000, 190), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, "SI_min  = 3.5 "  , (1000, 240), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
        image = cv2.putText(frame, "ACT_MIN  = 2   "  , (1000, 270), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)

        image = cv2.putText(frame, 'CNN BBOX   :' + " GREEEN "  , (20,280), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(frame, 'OPENCV BBOX   :' + "RED" , (20,310), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(frame, 'Tracker BBOX   :' + "BLUE" , (20,330), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)

        plt.axvline(x=FRAME_COUNT)
        fig = plot.get_figure()
        #fig.legend(loc=1)
        fig.savefig("output.png")
        img = cv2.cvtColor(cv2.imread('output.png'), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1280, 720))
        image = cv2.resize(image, (1280, 720))
        im_v = cv2.vconcat([image, img])
        #cv2.imwrite("COMBI.png", im_v)
        analysis_feed.write(im_v)
        del plot,fig,img,im_v

except Exception as e:
    os.remove('output.png')
    print("ERROR ", e)
    analysis_feed.release()
analysis_feed.release()
