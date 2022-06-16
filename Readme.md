This folder contains the Code for Analysis of Events

Main_script requires a an arguments, which states the name of the event video and json with same name.

Mode is optional argument which is required to pass when code version is v7

Structure:

AWS_Automation
|
|--input
   |video.mp4
   |video1.mp4
|--jsons
   |video.json
   |video1.json
| main.py

Command to execute on video.mp4 following code version <= 6.2

    python3 main.py --name video 

Command to execute on video1.mp4 following code version > 6.2
    
    python3 main.py --name video1 --mode 2

   