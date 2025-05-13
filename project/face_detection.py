import cv2
import numpy as np
import matplotlib.pyplot as plt


# Importation de l'image ↓ 

image = cv2.imread("classe_amphi_ga.jpeg")
print(image.shape)

# ---------------------- ↑

# Transformation de l'image en niveau de gris ↓ 

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

# ------------------------------------------- ↑

# Classificateur de cascade pré-entraîné Haar ↓ 

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------------------------- ↑

# Detection des visages ↓

face = face_classifier.detectMultiScale(
    gray_image, minNeighbors=12, minSize=(60,60)
)

# --------------------- ↑

# Création de carrés autour de chaque visage ↓

for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

# ------------------------------------------ ↑

# Resize de l'image ↓ 

height, width = image.shape[0], image.shape[1]

new_width = 1024

scale = new_width / width
new_height = int(height * scale)

image = cv2.resize(image, (1024, new_height), interpolation=cv2.INTER_AREA)

# ----------------- ↑

# Affichage du résultat ↓ 

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------- ↑
