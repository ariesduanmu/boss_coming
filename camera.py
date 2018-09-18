# -*- coding: utf-8 -*-
import numpy as np
import cv2


def capture_video(output_file="output.avi"):
    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter(output_file, -1, 20.0, (640,480))

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()