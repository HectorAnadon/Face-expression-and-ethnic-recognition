import cv2
import pickle
import argparse
import numpy as np
import dlib

IMAGE_SIZE=50
SHAPE_PREDICTOR="shape_predictor_68_face_landmarks.dat"

EMOTION = {0: "neutral",
	1: "anger",
	2: "joy",
	3: "sadness",
	4: "fear",
	5: "disgust",
	6: "shame"}

def predict_emotion(name):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

	im = cv2.imread(name, cv2.IMREAD_COLOR)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

	rects = detector(im, 0)

	for rect in rects:
		face = im[rect.top():rect.bottom(), rect.left():rect.right()].copy() # Crop from x, y, w, h -> 100, 200, 300, 400
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
		shape = predictor(face, dlib.rectangle(left=0, top=0, right=IMAGE_SIZE, bottom=IMAGE_SIZE))
		poi = []
		for i in range(17,68):
			poi.append([shape.part(i).x, shape.part(i).y])
		poi = np.array(poi)
	
	poi = poi.reshape(poi.shape[0] * poi.shape[1])



	with open('emotion_classifier.pkl', 'rb') as fid:
	    clf = pickle.load(fid)

	return clf.predict(np.array([poi]))


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to test image")
	args = vars(ap.parse_args())
	result = predict_emotion(args["image"])
	print(EMOTION[result[0]])