from .basic_apriltag_roi_detection import *
import os
import csv

def detect_apriltag_from_cv(cap, detector, is_plot=True):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_apriltag_from_array(gray, detector,is_plot=False)
        
        if is_plot:
            for det in detections:
                corners = det.corners.astype(int)
                for i in range(4):
                    pt1 = tuple(corners[i])
                    pt2 = tuple(corners[(i + 1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

                centre = tuple(det.center.astype(int))
                cv2.putText(frame, str(det.tag_id), centre, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("April tag detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def generate_training_data(detector, image_folder_path, is_save_to_csv=True):
    #image_folder_path = "captured_frames"
    sorted_files = sorted(os.listdir(image_folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    data = []
    for filename in sorted_files:
        #print("File name:", filename)
        joined_filepath = os.path.join(image_folder_path, filename)
        detection = detect_apriltag_from_image(joined_filepath, detector, is_plot=False)
        has_apriltag = len(detection) != 0
        if len(detection) == 0:
            corners = np.array([])
        else:
            corners = detection[0].corners.flatten()
        curr_row = {
            "img_filepath": joined_filepath, 
            "has_apriltag": has_apriltag,
            "corners": corners
        }
        data.append(curr_row)
    
    if is_save_to_csv:
        csv_filename = "apriltag_train_data.csv"
        with open(csv_filename, mode='w', newline='') as csvfile:
            fieldnames = ["img_filepath", "has_apriltag", "corners"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                row["corners"] = ",".join(map(str, row["corners"]))
                writer.writerow(row)
        print(f"Saved {len(data)} rows to {csv_filename}")
        
    return data