This folder contains the Code for Analysis of Events

Main_script requires a single arguments, which states the name of the event video and json with same name.

Structure:

AWS_Automation
|
|--input
   |video.mp4
   |video1.mp4
|--jsons
   |video.json
   |video1.json
| Main_script.py

Command to execute on video.mp4:

    python3 main_script.py --name video   

Command to execute on video1.mp4:
    
    python3 main_script.py --name video1

   