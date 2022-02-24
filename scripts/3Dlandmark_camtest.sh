python 3Dlandmark_camtest.py --face-detector "retinaface"\
                             --model-path "cp/best.pth"\
                             --detectmodelfile "detection/retinaface/weights/mobilenet0.25_Final.pth"\
                             --config-path 'config/FAN.yaml' \
                             --model-size 'MEDIUM' \
                             --device 'cpu' # or 'cuda'
