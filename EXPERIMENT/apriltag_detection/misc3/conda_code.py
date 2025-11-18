import cv2
import numpy as np
import requests
import time
import queue
import threading
import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
import queue

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = Queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()
    
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()    


URL = "http://192.168.4.1"
cap = cv2.VideoCapture(URL + ":81/stream")

if __name__ == '__main__':
    requests.get(URL + "/control?var=framesize&val={}".format(8))

    while True:
        
        if cap.isOpened():
            ret, frame = cap.read()
            cv2.imshow("Output", frame)

            #imgnp=np.array(bytearray(cap.read()),dtype=np.uint8)
            #im = cv2.imdecode(imgnp,-1)
            bbox, label, conf = cv.detect_common_objects(frame)
            im = draw_bbox(frame, bbox, label, conf)
            cv2.imshow('Output',im)

            key = cv2.waitKey(3)
            
            if key == 27:
                break

    cv2.destroyAllWindows()
    cap.release()
