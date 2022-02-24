from __future__ import print_function
import os
import argparse
import torch
import cv2
from skimage import io
from skimage import color
import numpy as np
import time
import opts
from utils import *
import models
import yaml
import logging
logging.getLogger().setLevel(logging.INFO)

class FaceAlignment:
    def __init__(self, args): 
        self.device = args.device
        face_detector_module = __import__('detection.' + args.face_detector,
                                          globals(), locals(), [args.face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=args.device, path_to_detector=args.detectmodelfile)
        self.face_alignment_net = torch.load(args.model_path, map_location=args.device)
        self.face_alignment_net.to(args.device)
        self.face_alignment_net.eval()

    def get_landmarks(self, image_or_path, detected_faces=None):
        return self.get_landmarks_from_image(image_or_path, detected_faces)

    def get_landmarks_from_image(self, image_or_path, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        if isinstance(image_or_path, str):
            try:
                image = io.imread(image_or_path)
            except IOError:
                print("error opening file :: ", image_or_path)
                return None
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        reference_scale = 200
        if detected_faces is None:
            time_0 = time.time()
            detected_faces = self.face_detector.detect_from_image(image[..., ::-1].copy())
            detected_faces = [det for det in detected_faces if det[-1] >= 0.9]
            # logging.info("detector takes {}".format(time.time()-time_0))
            reference_scale = self.face_detector.reference_scale

        # print("len detected faces: {}".format(len(detected_faces)))
        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        for det in detected_faces:
            expand_ratio=4
            width = det[2] - det[0]
            height = det[3] - det[1]
            det[0] = max(0, det[0]-width/expand_ratio)
            det[1] = max(0, det[1]-height/expand_ratio)
            det[2] = min(image.shape[1], det[2]+width/expand_ratio)
            det[3] = min(image.shape[0], det[3]+height/expand_ratio)

        torch.set_grad_enabled(False)
        landmarks = []
        landmarks_in_crops = []
        img_crops = []

        image = im_to_torch(image)
        
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] -
                 (d[3] - d[1]) / 2.0])
            # center[1] = center[1] + (d[3] - d[1]) * 0.12
            hw = max(d[2] - d[0], d[3] - d[1])
            scale_x = float(hw / reference_scale)
            scale_y = float(hw / reference_scale)

            inp = crop(image, center, [scale_x, scale_y], reference_scale)

            # io.imsave('crop_%s.jpg' % i,im_to_numpy(inp))
            img_crops.append(im_to_numpy(inp))

            inp = inp.to(self.device)
            inp.unsqueeze_(0)

            time_0 = time.time()
            out = self.face_alignment_net(inp)[-1].detach()
            # logging.info("landmark takes {}".format(time.time()-time_0))
            
            out = out.cpu()
            
            pts, pts_img = get_preds_fromhm(out, [center], [[scale_x, scale_y]], [reference_scale])
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            landmarks.append(pts_img.numpy())
            landmarks_in_crops.append(pts.numpy())
        return landmarks, detected_faces, landmarks_in_crops, img_crops

    def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True):
        detected_faces = self.face_detector.detect_from_directory(path, extensions, recursive, show_progress_bar)

        predictions = {}
        for image_path, bounding_boxes in detected_faces.items():
            image = io.imread(image_path)
            preds, detected_faces = self.get_landmarks_from_image(image, bounding_boxes)
            predictions[image_path] = preds

        return predictions

    def draw_landmarks(self, img):
        result = self.get_landmarks(img)
        if result is None:
            return img
            
        preds, detected_faces, preds_in_crops, img_crops = result

        for k,d in enumerate(detected_faces):
            d = d.astype(np.int)
            cv2.rectangle(img,(d[0],d[1]),(d[2],d[3]),(255,255,255))
            landmark = preds[k]
            for i in range(landmark.shape[0]):
                pts = landmark[i]
                cv2.circle(img, (int(pts[0]), int(pts[1])),2,(0,255,0), -1, 2)
                # img = cv2.putText(img,str(i),(pts[0],pts[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
        # cv2.imwrite('./output/{}.jpg'.format(uuid4()),img.astype(np.uint8))
        return img.astype(np.uint8)

    def get_head_pose(self, img: np.ndarray):
        size = img.shape
        try:
            preds, detected_faces, _, _ = self.get_landmarks(img)
        except:
            print("Can not get landmark")
            return img
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        for k, d in enumerate(detected_faces):
            landmark = preds[k]
            image_points = np.array([
                tuple(landmark[30]),
                tuple(landmark[8]),
                tuple(landmark[36]),
                tuple(landmark[45]),
                tuple(landmark[48]),
                tuple(landmark[54])
            ], dtype="double")

            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double"
            )

            dist_coeffs = np.zeros((4,1))
            (success, rotation_vector, translation_vector) = \
                cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,\
                    flags=cv2.cv2.SOLVEPNP_ITERATIVE)
            
            logging.info("> rotation vector: \n{}".format(rotation_vector))
            logging.info("> translation vector: \n{}".format(translation_vector))

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),\
                rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(img, p1, p2, (255,0,0), 2)
        return img
            


if __name__ == '__main__':
    P = argparse.ArgumentParser(description='Predict network script')
    P.add_argument('--face-detector', type=str, required=True, help='face detector model to use')
    P.add_argument('--modelfile', type=str, required=False, help='model file path')
    P.add_argument('--detectmodelfile', type=str, required=False, help='face detect model file')
    P.add_argument('--input', type=str, required=False, help='input image file')
    P.add_argument('--input-folder', type=str, required=False, help='input image folder')
    P.add_argument('--output-folder', type=str, required=True, help='output folder')
    P.add_argument('--device', type=str, default='cpu', help='used device: cpu or cuda')
    args = P.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    fa = FaceAlignment(face_detector=args.face_detector, modelfilename=args.modelfile, facedetectmodelfile=args.detectmodelfile, device=args.device)
    if fa:
        if args.input is not None and args.input_folder is not None:
            raise Exception("Only support 1 kind of input at a time <TECH>")
        if args.input is not None:
            img = np.array(Image.open(path)).transpose(1,2,0)
            img = fa.draw_landmarks(img)
            cv2.imwrite('res.jpg',img)
        else:
            import tqdm
            import glob
            input_path = os.path.join(args.input_folder, "*")
            print(input_path)
            for idx, path in tqdm.tqdm(enumerate(glob.glob(input_path))):
                img = np.array(Image.open(path))
                try:
                    # img = fa.get_head_pose(img)
                    img = fa.draw_landmarks(img).astype(np.uint8)
                    cv2.imwrite(os.path.join(args.output_folder,'{}.jpg'.format(idx)),cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                except:
                    logging.info("exception at {}".format(path))
    else:
        print("FaceAlignment init error!")
