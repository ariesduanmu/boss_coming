# -*- coding: utf-8 -*-
import cv2
from recognize_face import recognize_face_from_cv

def save_face_from_webcam(save_video=False, output_file=None):
    cap = cv2.VideoCapture(0)
    if save_video:
        output_file = output_file or "output.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output_file, -1, 20.0, (640,480))
    k = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            for face in recognize_face_from_cv(frame):
                cv2.imwrite("images/image_{}.jpg".format(k), face)
            cv2.imshow('Video', frame)
            k += 1
            if save_video:
                # 是否需要灰度处理
                # video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                video.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    if save_video:
        video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_face_from_webcam(False)

    



