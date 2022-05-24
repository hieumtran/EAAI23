import cv2
import dlib
import numpy as np
import imutils
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./PyTorch_reg/shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('./data/train_set/images/1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(image_gray)
# remove axes and show image
plt.axis("off")
plt.imshow(image_gray, cmap = "gray")
plt.show()
# image = imutils.resize(image, width=500)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Landmark Detection', image)
# detect faces in the grayscale image
# rects = detector(gray, 1)

# # Detect landmarks for each face
# for rect in rects:
#     # Get the landmark points
#     shape = predictor(gray, rect)
#     # Convert it to the NumPy Array
#     shape_np = np.zeros((68, 2), dtype="int")
#     for i in range(0, 68):
#         shape_np[i] = (shape.part(i).x, shape.part(i).y)
#     shape = shape_np

#     # Display the landmarks
#     for i, (x, y) in enumerate(shape):
#     # Draw the circle to mark the keypoint 
#         cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
# Display the image
cv2.imshow('Landmark Detection', image)