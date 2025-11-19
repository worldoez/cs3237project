# C3237 Group Project: UGLL-E

Group Members: Stanley Wijaya, Ella Yovita Suwibowo, Hu Chong Xern, Evan Zhao, Daphne Shaine Wilhelmina

## Code Files for Final Creation

The final creation is composed of 3 main parts
- The main autonomous carrying basket (Known as UGLL-E seperated as UGLL-E Body, UGLL-E Eye, UGLL-E Nose)
- The backpack user wears with an IMU (Known as EVE)
- The April Tag belt users wears on the waist (Known as M-O, tracked by UGLL-E Eye)

<ugll-e_body_thing> contains the ESP32 sketch that mainly controls the motors. <SERVER-ugll-e_body_thing> contains the server that specifies the controls that the motors should make. <SERVER-ugll-e_body_thing> is the main server, it receives data from other two servers to determine the required control.

<ugll-e_nose_thing> contains the ESP32 sketch that handles obstacle avoidance. It interacts with <ugll-e_body_thing> so that the robot knows to stop when an obstacle is detected.

<ugll-e_eye_thing> contains the ESP32 sketch that allows for April Tag detection. <SERVER-ugll-e_eye_thing> contains two files that receive and analyze data from it so that the prediction can be used by <SERVER-ugll-e_body_thing>.

<eve_thing> contains the ESP32 sketch that tracks and sends IMU data. <SERVER-eve_thing> contains the server that would process the data so the prediction can be used by <SERVER-ugll-e_body_thing>.

Do see the <START-SERVER-*.txt> files on how to start the servers. The only complex one is starting the <SERVER-eve_thing>.

Note that wireless and server information may need to be updated in relevant files. The implementation requires all servers and ESP32s to connect to the ESP32Cam's WiFi. Moreover, file paths in files may need to be updated.

The EXPERIMENTS folder contains the raw files created as we slowly brought UGLL-E to life! Some files may have been assisted thanks to the help of GenAI. These files are experiments and is not relevant to the final creation.

## Demo Files for Final Creation

See the ./DEMO folder.

## Master Document for Planning

Google Docs Link: https://docs.google.com/document/d/1S7kgdh2BvyPQz5v5Bl1cbjoKiqJ7__Wd9OBoXAr5L8s/edit?usp=sharing

## IoT System Architecture for Brainstorming 

FigJam Link: https://www.figma.com/board/jSL3bkGkAp7MK5tGulI1Tg/ARCHITECTURE-%7C-CS3237-Group-20-Project?node-id=0-1&t=6BqzkDOl376G2ufz-1

