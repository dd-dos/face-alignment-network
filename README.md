# 3D face landmarks
## Prerequisite
`pip install -r requirements.txt`

## Training
- Dataset: [Google Drive](https://drive.google.com/file/d/11gMaaHPrF_O5UfXFhjp93jGqn4FcMJ0u/view?usp=sharing).
- Training script: [Google Colab](https://colab.research.google.com/drive/1kbMgBG_GL79o5kcVE-LL2q3Injly5Jx1?usp=sharing).
- Remember to replace 300WLP path in:   
`!unzip "/content/drive/MyDrive/raw_data/300WLP.zip" -d "/content/face-alignment/datasets"`  
in training script with your path to 300WLP.zip.
- Run each cell sequentially.

## Testing
- Pretrained FAN: [Google Drive](https://drive.google.com/drive/folders/1SBl2o1g4cCvR7xYKRLx6F4PNleZ7ocds?usp=sharing). Place at `cp`.
- Pretrained retinaface for face detection: [Google Drive](https://drive.google.com/drive/u/4/folders/1vOCVsIt8ThfAHUuzxf9tqMf0PhbOnH20). Place all of them at `detection/retinaface/weights`.
- Test using cam: `sh scripts/3Dlandmark_camtest.sh`.
- Production ready: `cd package & mlchain run`.

