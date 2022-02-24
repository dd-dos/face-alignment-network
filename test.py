import torch
import numpy as np
import cv2

from retinaface_detector.model import RetinaNetDetector

weight_path = 'retinaface_detector/weights/mobilenet0.25_Final.pth'

# detector = torch.jit.load(weight_path)
detector = RetinaNetDetector()
detector.load_weights(weight_path)

img = torch.rand((640, 480, 3)).to(torch.uint8)
_ = detector.forward(img)