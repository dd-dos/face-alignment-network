import argparse
import logging
import time
from uuid import uuid4

import cv2

from inference import FaceAlignment

logging.getLogger().setLevel(logging.INFO)


def test_retina_landmark(args):
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("sample_rotate.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    fa = FaceAlignment(args)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF
        time_0 = time.time()
        frame = fa.draw_landmarks(frame)
        result.write(frame)
        # frame = fa.get_head_pose(frame)
        infer_time = time.time()-time_0
        logging.info(f"inference time: {infer_time} - fps: {1/infer_time}")

        cv2.imshow("", frame)

        if key == ord("q"):
            break
    
    cap.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")

def cam_cap():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        rd_id = uuid4()
        if key == ord("c"):
            cv2.imwrite("input/{}.jpg".format(rd_id),frame)

        cv2.imshow("", frame)

        if key == ord("q"):
            break


if __name__=="__main__":
    P = argparse.ArgumentParser(description='landmark')
    P.add_argument('--face-detector', type=str, required=True, help='face detector model to use')
    P.add_argument('--detectmodelfile', type=str, required=False, help='face detect model file')
    P.add_argument('--model-path', type=str, required=False, help='model file path')
    P.add_argument('--device', type=str, default='cpu', help='used device: cpu or cuda')
    P.add_argument('--config-path', type=str, required=False, help='model config path')
    P.add_argument('--model-size', type=str, help='model size: TINY, SMALL, MEDIUM or LARGE')
    P.add_argument('--netType', type=str, default='FAN', help='options: fan')
    P.add_argument('--save-heatmap', action="store_true", help="Save heatmap")
    P.add_argument('--strip-heatmap', type=float, default=0, help="Strip value from heatmap")
    P.add_argument('--black-out', action="store_true", help="Black out")

    args = P.parse_args()
    test_retina_landmark(args)
