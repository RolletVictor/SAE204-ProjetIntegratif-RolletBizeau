import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

images = ["/data/mbappe/mbappe.jpg", "/data/obama/obama.jpg", "/data/trump/trump.jpeg"]
labels = [0, 1, 2]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))

label, confiance = recognizer.predict("test.jpeg")

