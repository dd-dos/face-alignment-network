python3 train.py --data "datasets/300WLP"\
                 --snapshot 1\
                 --epochs 1000\
                 --workers 4\
                 --train-batch 32\
                 --val-batch 64\
                 --checkpoint "./logs/3D-face-alignment/FAN"\
                 --netType FAN\
                 --config_path './config/FAN.yaml' \
                 --model_size 'MEDIUM' \
                 --pointType 3D\
                 --lr 1e-3\
                 # --pretrained "/content/drive/MyDrive/training/3D-face-alignment/FAN/best.pth"