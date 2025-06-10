import cv2

imagePath = 'image2.jpg'

image = cv2.imread(imagePath)

print("Shape original image : ", image.shape)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Shape gray image :     ", image_gray.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    image_gray, minNeighbors=15, minSize=(30, 30)
)

id = 1
for (x, y, w, h) in face:
    cv2.imwrite(f"people_unknown/unknow_{id}.png".format(id), image[y-25:y+(h+20), x-15:x+(w+10)])
    id += 1

    cv2.rectangle(image, (x-15, y-25), (x + (w+10), y + (h+20)), (0, 255, 0), 4)

cv2.imshow("", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

