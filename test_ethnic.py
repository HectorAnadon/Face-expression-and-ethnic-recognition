from face_network import create_face_network
import cv2
import argparse
import numpy as np
from keras.optimizers import Adam, SGD

ETHNIC = {0: 'Asian', 1: 'Caucasian', 2: "African", 3: "Hispanic"}

def predict_ethnic(name):
	means = np.load('means_ethnic.npy')

	model = create_face_network(nb_class=4, hidden_dim=512, shape=(224, 224, 3))
	model.load_weights('weights_ethnic.hdf5')

	im = cv2.imread(name, cv2.IMREAD_COLOR)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
	im = cv2.resize(im, (224, 224))
	im = np.float64(im)
	im /= 255.0
	im = im - means

	return model.predict(np.array([im]))


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to test image")
	args = vars(ap.parse_args())

	result = predict_ethnic(args["image"])
	print(result)
	print(ETHNIC[np.argmax(result)])