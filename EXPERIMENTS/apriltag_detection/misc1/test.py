import os
import numpy as np
import cv2
curr_frames = list(os.scandir("captured_frames"))

sorted_curr_frames = []

for i in range(len(curr_frames)):
    if curr_frames[i].name == '.DS_Store':
        continue
    sorted_curr_frames.append(int(curr_frames[i].name.split('_')[1].split('.')[0]))

sorted_curr_frames.sort()

def rename_in_correct_order():
    for i in range(len(sorted_curr_frames)):
        old_file_name = f"captured_frames/frame_{sorted_curr_frames[i]}.jpg"
        new_file_name = f"captured_frames/frame_{i}.jpg"
        # old_file_name = f"captured_frames_2/frame_{sorted_curr_frames[i]}.jpg"
        # new_file_name = f"captured_frames_2/frame_{int(sorted_curr_frames[i]) + 100000000}.jpg"
        os.rename(old_file_name, new_file_name)
        #print(old_file_name, new_file_name)

def play_as_video(fps=30):
    frames = sorted(
        [f for f in os.listdir("captured_frames") if f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    first_frame = cv2.imread(os.path.join("captured_frames", frames[0]))
    height, width, _ = first_frame.shape
    cv2.namedWindow("Video Playback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Playback", width, height)
    
    for frame_name in frames:
        frame_path = os.path.join("captured_frames", frame_name)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue

        cv2.imshow("Video Playback", frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

rename_in_correct_order()
#play_as_video()

# for frame in curr_frames:
#     old_name = frame.name
#     new_name = os.path.join("captured_frames", old_name.split('_', 1)[1])
#     os.rename(os.path.join("captured_frames", old_name), new_name)