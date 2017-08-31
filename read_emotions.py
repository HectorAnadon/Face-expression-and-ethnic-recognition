import numpy as np
import h5py
import dlib
import cv2

from PIL import Image
import imutils
from imutils import face_utils

SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
IMAGE_SIZE = 50

f = h5py.File('images.h5','r+')

images = np.array(f['data'])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

faceFeatures = []
faces = []
for image in images:

	# Display image
	#print(image.shape)
	#image = Image.fromarray(np.uint8(images[0]*255), 'RGB')
	#image.show()

	image = np.uint8(image * 255)
	rects = detector(image, 0)

	for rect in rects:
		face = image[rect.top():rect.bottom(), rect.left():rect.right()].copy() # Crop from x, y, w, h -> 100, 200, 300, 400
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
		shape = predictor(face, dlib.rectangle(left=0, top=0, right=IMAGE_SIZE, bottom=IMAGE_SIZE))
		poi = []
		for i in range(17,68):
			poi.append([shape.part(i).x, shape.part(i).y])
		faceFeatures.append(poi)
		# VISUALIZE
		# shape = face_utils.shape_to_np(shape)
		# output = face_utils.visualize_facial_landmarks(face, shape)
		# faces.append(output)
		# cv2.imshow("Image", output)
		# cv2.waitKey(0)
	if (len(rects) == 0):
		print("Error: face not recognized")
		# TODO: remove image from h5 dataset
		cv2.imshow("Image", image)
		cv2.waitKey(0)

# Save to disk
faceFeatures = np.stack(faceFeatures)
# faces = np.float64(np.stack(faces))
# faces /= 255.0
print(faceFeatures.shape)
if ("faceFeatures" in f):
	del f['faceFeatures']
if ("faces" in f):
	del f['faces']

y_dset = f.create_dataset('faceFeatures', faceFeatures.shape, dtype='i')
y_dset[:] = faceFeatures

# y_dset = f.create_dataset('faces', faces.shape, dtype='i')
# y_dset[:] = faces

f.close()
