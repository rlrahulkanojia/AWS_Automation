#imports
import os
import re
import cv2
import json
import argparse
import traceback
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class Analytics:
    def __init__(self, 
                 name, 
                 video_path = 'input/',
                 analytics_path = 'jsons/',
                 output_path='AnalysisOutput/', 
                 mode=1,
                 plot= False):
        self.name           = name
        self.analytics_path = analytics_path + self.name+'.json'
        self.data           = None
        self.new_df         = None
        self.video          = cv2.VideoCapture(video_path + self.name+'.mp4')
        fourcc              =  cv2.VideoWriter_fourcc(*'mp4v')
        _, frame = self.video.read()
        self.video          = cv2.VideoCapture(video_path + self.name+'.mp4')
        if self.video.isOpened() == False:
            print("Invalid Video")
        self.FRAME_COUNT    = 0
        self.mode           = mode
        
        
        self.CNN_COLOR     =  (0,255,0)   # GREEN
        self.OUTPUT_DIR    =  "output/"
        self.OCV_COLOR     =  (0,0,255)   # RED
        self.TRK_COLOR     =  (255,0,0)   # Blue
        self.HRATIO        =  1#frame.shape[0] / HEIGHT
        self.WRATIO        =  1#frame.shape[1] / WIDTH
    
        self.fontscale     = 3
        self.color         = (0, 255, 0)
        self.thickness     = 2
        self.plot          = plot
        
        if self.plot == True:
            self.analysis_feed  = cv2.VideoWriter(output_path + name+ ".mp4",fourcc, 5, (frame.shape[1],frame.shape[0]*2))
        else:
            self.analysis_feed  = cv2.VideoWriter(output_path + name+ ".mp4",fourcc, 5, (frame.shape[1],frame.shape[0]))
        
    def get_data(self):
        if self.mode == "1":
            json_path  = self.analytics_path
            f = open(json_path).readlines()
            data = json.loads('{'+ f[0][10:-2])
            self.data = data
        elif self.mode == "2":
            with open(self.analytics_path, 'r') as f:
                self.data = json.load(f)
                 
    def create_df(self):

        new_df = pd.DataFrame(columns=['FRAME_COUNT', 'OPEN_BOX', 'CNN_BOX','Vertical', 'SPEED', 'POSE', 'SUI', 'SPLASH', 'ACTIND', 'TBBOX', 'ALARM'])
        ind = 0
        if self.mode == "1":
            all_keys = self.data.keys()
            main_dict = self.data
        elif self.mode == "2":
            all_keys = self.data['Data'].keys()
            main_dict = self.data['Data']
        else:
            print("Mode not supported")
            exit(0)
            
        for key in all_keys:
            D = main_dict[key]
            T_ = json.loads(D['TRACKERS_DATA'])
            
            if len(T_.keys())==0:
                new_df.loc[ind] = [str(D['FRAME_COUNT'])] +  [str(D['OPEN_BOX'])] + [str( D['CNN_BOX'])]\
                                + [""] + [""] +  [""]  +  [""] + [""] + [""] + [""]+ [""]
                ind += 1
                
            else:
                t_bboxes = []
                for key in list(T_.keys()):
                    T = json.loads(D['TRACKERS_DATA'])[key]
                    t_bboxes.append(T['BBOX'])
                new_df.loc[ind] = [str(D['FRAME_COUNT'])] +  [str(D['OPEN_BOX'])] + [str( D['CNN_BOX'])] + [str(T['VERTICAL'])] + [str(T['SPEED'])] + [str(T['POSE'])] + \
                                    [str(T['SUI'])] + [str(T['SPLASH'])] + [str(T['ACTIND'])] + [str(t_bboxes)] + [str(T['ALARM'])]

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

        self.new_df = new_df
        
    def get_fence(self):
        self.fence = np.array(self.data['Meta']["Fence"])
           
    def generate(self):
        color  = (255,255,255)
        fontscale = 0.5
        thickness = 1
        alpha = 0.7
        try:
            #while self.video.isOpened():
            ret_val, frame = self.video.read()
            ret_val, frame = self.video.read()
            for _ in range(int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))):
                self.FRAME_COUNT += 1
                ret_val, frame = self.video.read()
                centroid = None
                ## OPENCV 
                arr = self.new_df.iloc[self.FRAME_COUNT, 2]
                if len(arr)>3:
                    temp = re.findall(r'\d+', arr)
                    temp = [t for t in temp if len(t)<=4]
                    temp = list(map(int, temp))

                    for ind in range(len(temp)//4):
                        ind = ind * 4
                        bbox = temp[ind] , temp[ind+1] , temp[ind+2] , temp[ind+3] 
                        start_point = (  int(bbox[0]* self.WRATIO ) , int(bbox[1] * self.HRATIO)  )
                        end_point = (int(bbox[2]* self.WRATIO), int(bbox[3]* self.HRATIO ) )
                        image = cv2.rectangle(frame, start_point, end_point, self.OCV_COLOR, self.thickness)
                        centroid = ( int(bbox[0] +(bbox[2]-bbox[0])/2), int(bbox[1] + (bbox[3]-bbox[1])/2))
                        centroid = (int(centroid[0] * self.HRATIO) , int(centroid[1] *  self.WRATIO))
                        image = cv2.circle(frame, centroid, 5, self.OCV_COLOR, 2)

                ## CNN 
                arr = self.new_df.iloc[self.FRAME_COUNT, 3]
                if len(arr)>3:
                    temp = re.findall(r'\d+', arr)
                    temp = [t for t in temp if len(t)<=4]
                    temp = list(map(int, temp))
                    for ind in range(len(temp)//4):
                        ind = ind * 4
                        bbox = temp[ind] , temp[ind+1] , temp[ind+2] , temp[ind+3]
                        start_point = (  int(bbox[0]* self.WRATIO ) , int(bbox[1] * self.HRATIO)  )
                        end_point = (int(bbox[2]* self.WRATIO), int(bbox[3]* self.HRATIO ) )
                        image = cv2.rectangle(frame, start_point, end_point, self.CNN_COLOR, self.thickness)
                        centroid = ( int(bbox[0] +(bbox[2]-bbox[0])/2), int(bbox[1] + (bbox[3]-bbox[1])/2))
                        centroid = (int(centroid[0] * self.HRATIO) , int(centroid[1] *  self.WRATIO))
                        image = cv2.circle(frame, centroid, 5, self.CNN_COLOR, 2)

                ## TRACKER 
                TR = 0
                arr = self.new_df.iloc[self.FRAME_COUNT, 10]
                if len(arr)>3:
                    temp = re.findall(r'\d+', arr)
                    temp = [t for t in temp if len(t)<=4]
                    temp = list(map(int, temp))
                    TR = len(temp)//4
                    for ind in range(len(temp)//4):
                        ind = ind * 4
                        bbox = temp[ind] , temp[ind+1] , temp[ind+2] , temp[ind+3] 
                        start_point = (  int(bbox[0]* self.WRATIO ) , int(bbox[1] * self.HRATIO)  )
                        end_point = (int(bbox[2]* self.WRATIO), int(bbox[3]* self.HRATIO ) )
                        image = cv2.rectangle(frame, start_point, end_point, self.TRK_COLOR, self.thickness)
                        centroid = ( int(bbox[0] +(bbox[2]-bbox[0])/2), int(bbox[1] + (bbox[3]-bbox[1])/2))
                        centroid = (int(centroid[0] * self.HRATIO) , int(centroid[1] *  self.WRATIO))
                        image = cv2.circle(frame, centroid, 5, self.TRK_COLOR, 2)
                
                
                overlay = frame.copy()
                overlay = cv2.rectangle(overlay, (10,60), (200,180), (0,0,0), -1)
                
                
                overlay = cv2.putText(overlay, 'Total tracks   : '+ str(TR), (20,80), cv2.FONT_HERSHEY_SIMPLEX, fontscale,color,thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'Tracks(Fence) : '+ str(TR), (20,100), cv2.FONT_HERSHEY_SIMPLEX, fontscale,color,thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'CNN Color    :' + "GREEN"  , (20,120), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'OPENCV Color :' + "RED" , (20,140), cv2.FONT_HERSHEY_SIMPLEX,   fontscale, color, thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'Tracker Color :' + "BLUE"  , (20,160), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)
            
                #### alpha, the 4th channel of the image
                
                
                #### apply the overlay
                frame =  cv2.addWeighted(overlay, alpha, frame, 1 - alpha,0, frame)
                
                if self.plot == True:

                    plot  = self.new_df[self.FRAME_COUNT-20:self.FRAME_COUNT+20].plot("index", [ 'Vertical', 'SPEED', 'POSE_50', 'SUI_50', 'SPLASH', 'ACTIND'], ylim=(0,110), xlabel='FRAME COUNT',figsize=(10, 6))

                    vertical = self.new_df.iloc[self.FRAME_COUNT,4]
                    SPEED    = 0
                    POSE     = 0
                    SUI      = 0
                    SPLASH   = 0
                    ACTIND   = 0
                    DROWN    = False
                    if str(vertical)!='nan':
                        SPEED  = self.new_df.iloc[self.FRAME_COUNT,5] 
                        POSE   = self.new_df.iloc[self.FRAME_COUNT,6]
                        SUI    = self.new_df.iloc[self.FRAME_COUNT,7]
                        SPLASH = self.new_df.iloc[self.FRAME_COUNT,8]
                        ACTIND = self.new_df.iloc[self.FRAME_COUNT,9]
                        DROWN  = self.new_df.iloc[self.FRAME_COUNT,10]
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
                    image = cv2.putText(frame, 'FRAME   :' + str(self.FRAME_COUNT)  , (640, 130), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)


                    image = cv2.putText(frame, "HS_min  = 15  "  , (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    image = cv2.putText(frame, "PO_min  = 1   "  , (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    image = cv2.putText(frame, "SP_max  = 150 "  , (1000, 110), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    image = cv2.putText(frame, "VS_min  = 15  "  , (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color,  thickness,  cv2.LINE_AA)
                    image = cv2.putText(frame, "AI_min  = 3.5 "  , (1000, 190), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    image = cv2.putText(frame, "SI_min  = 3.5 "  , (1000, 240), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    image = cv2.putText(frame, "ACT_MIN  = 2   "  , (1000, 270), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)

                    plt.axvline(x=self.FRAME_COUNT)
                    fig = plot.get_figure()
                    #fig.legend(loc=1)
                    fig.savefig("output.png")
                    img = cv2.cvtColor(cv2.imread('output.png'), cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (frame.shape[1],frame.shape[0]))
                    image = cv2.resize(image, (frame.shape[1],frame.shape[0]))
                    im_v = cv2.vconcat([image, img])
                    #cv2.imwrite("COMBI.png", im_v)
                    self.analysis_feed.write(im_v)
                    plt.close(fig)
                    del plot,fig,img,im_v
                else:
                    self.analysis_feed.write(frame)
                
                
                
                
                if self.FRAME_COUNT == 248:
                    break
        except Exception as e:

            print(traceback.format_exc())
            self.analysis_feed.release()
        self.analysis_feed.release()

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--mode", default=1)

    value  = parser.parse_args()
    mode   = value.mode
    name   = value.name

    v1     = Analytics(name,mode=mode)
    v1.get_data()
    v1.create_df()
    v1.generate()
    if mode == '2':
        v1.get_fence()
    print("Completed")


