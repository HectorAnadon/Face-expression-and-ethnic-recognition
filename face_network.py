import numpy as np

from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace


def create_face_network(nb_class=2, hidden_dim=512, shape=(224, 224, 3)):
	# Convolution Features
	model = VGGFace(include_top=False, input_shape=shape)
	last_layer = model.get_layer('pool5').output
	x = Flatten(name='flatten')(last_layer)
	x = Dense(hidden_dim, activation='relu', name='fc6')(x)
	x = Dense(hidden_dim, activation='relu', name='fc7')(x)
	out = Dense(nb_class, activation='softmax', name='fc8')(x)
	custom_vgg_model = Model(model.input, out)

	print(custom_vgg_model.summary())
	return custom_vgg_model


