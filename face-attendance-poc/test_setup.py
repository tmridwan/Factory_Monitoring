import cv2
import insightface
import numpy as np

print("OpenCV version:", cv2.__version__)
print("InsightFace imported successfully")

# This downloads the pretrained model on first run (~few hundred MB)
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded successfully")