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
                 video_path     = 'input/',
                 analytics_path = 'jsons/',
                 output_path    = 'AnalysisOutput/', 
                 OPENCV_OFFSET  = 5,
                 mode           = 1,
                 EVENT_TYPE     = None,
                 plot           = True):
        self.name           = name
        self.analytics_path = analytics_path + self.name+'.json'
        self.data           = None
        self.new_df         = None
        print(video_path + self.name+'.mp4')
        self.video          = cv2.VideoCapture(video_path + self.name+'.mp4')
        fourcc              = cv2.VideoWriter_fourcc(*'mp4v')
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
        self.AIFENCE       = None
        
        if self.plot == True:
            self.analysis_feed  = cv2.VideoWriter(output_path + name+ ".mp4",fourcc, 5, (frame.shape[1],frame.shape[0]*2))
        else:
            self.analysis_feed  = cv2.VideoWriter(output_path + name+ ".mp4",fourcc, 5, (frame.shape[1],frame.shape[0]))
            
            
        # self.OPENCV_NET    = cv2.createBackgroundSubtractorMOG2(history = OPENCV_OFFSET, detectShadows=False) 
        self.EVENT_TYPE    = EVENT_TYPE
        
    def get_data(self):
        if self.mode == "1":
            json_path  = self.analytics_path
            f = open(json_path).readlines()
            data = json.loads('{'+ f[0][10:-2])
            self.data = data
        elif self.mode == "2":
            with open(self.analytics_path, 'r') as f:
                self.data = json.load(f)
                
        self.get_fence()
#         self.create_new_fence()
        try:
            self.create_new_fence()
            self.AIfence3d        = np.zeros((self.AIFENCE.shape[0], self.AIFENCE.shape[1],  3))
            self.AIfence3d[:,:,1] = self.AIFENCE
            self.AIfence3d[:,:,2] = self.AIFENCE
            self.AIfence3d        = self.AIfence3d.astype(np.uint8)
        except:
            self.AIFENCE = None
        
    def create_df(self):

        new_df = pd.DataFrame(columns=['FRAME_COUNT', 'OPEN_BOX', 'CNN_BOX','Vertical', 'SPEED', 'POSE', 'SUI', 'SPLASH', 'ACTIND', 'TBBOX', 'ALARM', 'TID'])
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
                                + [""] + [""] +  [""]  +  [""] + [""] + [""] + [""]+ [""] + [""]
                ind += 1
                
            else:
                t_bboxes = []
                for key in list(T_.keys()):
                    T = json.loads(D['TRACKERS_DATA'])[key]
                    t_bboxes.append(T['BBOX'])
                new_df.loc[ind] = [str(D['FRAME_COUNT'])] +  [str(D['OPEN_BOX'])] + [str( D['CNN_BOX'])] + [str(T['VERTICAL'])] + [str(T['SPEED'])] + [str(T['POSE'])] + \
                                    [str(T['SUI'])] + [str(T['SPLASH'])] + [str(T['ACTIND'])] + [str(t_bboxes)] + [str(T['ALARM'])] + [str(key)]

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
                
                # fgmask          = self.OPENCV_NET.apply(frame)
                # median          = cv2.medianBlur(fgmask, 15)
                # median3d        = np.zeros((median.shape[0], median.shape[1],  3))
                # median3d[:,:,1] = median
                # median3d[:,:,2] = median
                # median3d        = median3d.astype(np.uint8)

                # contours, _     = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # contours        = sorted(contours, key=cv2.contourArea)

                # for contour in reversed(contours[-3:]) :
                #     area        = cv2.contourArea(contour)
                #     rect        = cv2.boundingRect(contour)
                #     x,y,w,h     = rect

                #     if area > 1000: 
                #         frame   = cv2.rectangle(frame, ( int(x), int(y)),(int((x+w)), int((y+h))),  (0,255,0), 2 )
                # frame = cv2.addWeighted(frame, 1, median3d, 0.6, 1)  
                
                # if self.AIFENCE is not None:
                #     frame = cv2.addWeighted(frame, 1, self.AIfence3d, 0.2, 0.5)  
                
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
                overlay = cv2.rectangle(overlay, (10,60), (240,200), (0,0,0), -1)
                
                
                overlay = cv2.putText(overlay, 'Total tracks   : '+ str(TR), (20,80), cv2.FONT_HERSHEY_SIMPLEX, fontscale,color,thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'Tracks(Fence) : '+ str(TR), (20,100), cv2.FONT_HERSHEY_SIMPLEX, fontscale,color,thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'CNN Color     :' + "GREEN"  , (20,120), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'OPENCV Color :' + "RED" , (20,140), cv2.FONT_HERSHEY_SIMPLEX,   fontscale, color, thickness, cv2.LINE_AA)
                overlay = cv2.putText(overlay, 'Tracker Color  :' + "BLUE"  , (20,160), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)
                if self.EVENT_TYPE is not None:
                    overlay = cv2.putText(overlay, 'Event Type :' + str(self.EVENT_TYPE), (20, 180),  cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA)
            
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


                    color     = (255, 255, 255)
                    fontscale = 0.5
                    thickness = 1
                    
                    overlay = frame.copy()
                    overlay = cv2.rectangle(overlay, (10,200), (200,380), (0,0,0), -1)
                    
                    overlay = cv2.putText(overlay, 'Vertical  :' + str(vertical)[:5] , (10, 240), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, 'SPEED   :' + str(SPEED)[:5]  , (10, 260), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, 'POSE    :' + str(POSE)[:5]  , (10, 280), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, 'SUI      :' + str(SUI)[:5]  , (10, 300), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color,  thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, 'SPLASH  :' + str(SPLASH)[:5]  , (10, 320), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, 'ACTIND   :' + str(ACTIND)[:5]  , (10, 340), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, 'DROWN   :' + str(DROWN)[:5]  , (640, 80), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, 'FRAME   :' + str(self.FRAME_COUNT)  , (640, 130), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)


                    overlay = cv2.putText(overlay, "HS_min  = 15  "  , (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, "PO_min  = 1   "  , (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, "SP_max  = 150 "  , (1000, 110), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, "VS_min  = 15  "  , (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color,  thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, "AI_min  = 3.5 "  , (1000, 190), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, "SI_min  = 3.5 "  , (1000, 240), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    overlay = cv2.putText(overlay, "ACT_MIN  = 2   "  , (1000, 270), cv2.FONT_HERSHEY_SIMPLEX, fontscale,   color, thickness,  cv2.LINE_AA)
                    
                    frame =  cv2.addWeighted(overlay, alpha, frame, 1 - alpha,0, frame)
                    
                    plt.axvline(x=self.FRAME_COUNT)
                    fig = plot.get_figure()
                    #fig.legend(loc=1)
                    fig.savefig("output.png")
                    img = cv2.cvtColor(cv2.imread('output.png'), cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (frame.shape[1],frame.shape[0]))
                    frame = cv2.resize(frame, (frame.shape[1],frame.shape[0]))
                    im_v = cv2.vconcat([frame, img])
                    #cv2.imwrite("COMBI.png", im_v)
                    self.analysis_feed.write(im_v)
                    plt.close(fig)
                    del plot,fig,img,im_v
                else:
                    self.analysis_feed.write(frame)
                
                
                
                
                if self.FRAME_COUNT == int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))-1:
                    break
        except Exception as e:
            self.analysis_feed.release()
        self.analysis_feed.release()
    
    def create_new_fence(self):
        MASK                = np.zeros((800,800,3)).astype(np.float32)
        margin              = 40
        coords              = self.data["Meta"]['FenceCords'].replace('"', "")[1:-1].split(',')
        for i in coords:
            y,x             =  i.split('|')
            x               = int(x)*margin; y = int(y)*margin
            MASK[x:x+margin, y:y+margin,:] = 1


        kernel1             = np.ones((20,20), np.uint8)
        kernel2             = np.ones((10,10), np.uint8)
        img_dilation        = cv2.dilate(MASK, kernel1, iterations=1)
        img_erosion         = cv2.erode(img_dilation, kernel2, iterations=1) * 255 

        edged               = cv2.Canny(np.uint8(img_erosion), 0, 255)
        contours, _         = cv2.findContours(edged, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
        MASK                = cv2.drawContours(MASK, contours, -1, (0, 255, 0), thickness=cv2.FILLED)[:,:,1]
        MASK                = cv2.resize(MASK, (1280, 720))

        CASE                = 0
#         return MASK
        for contour in contours:
            approx          = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area            = cv2.contourArea(contour)
            if ((len(approx) > 8) & (area > 30) ):
                CASE = 1

        if CASE == 1:
            self.AIFENCE =  MASK

        else:
            MASK                = np.ones((720,1280,3)).astype(np.float32)[:,:,0]
            self.AIFENCE =  MASK
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--mode", default="1")

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


