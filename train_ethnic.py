from keras.optimizers import SGD
import h5py
from face_network import create_face_network
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

train_split = 0.7

f = h5py.File('images.h5', 'r') 
X_data = np.array(f['dataResized'])
y_data = np.array(f['ethnic'])

#One-hot
y_data = to_categorical(y_data, 4)

# Split into training and validation sets
num_images = len(y_data)
p = np.random.permutation(num_images)
X_data = X_data[p]
y_data = y_data[p]


X_train = X_data[0:int(round(train_split*num_images))]
y_train = y_data[0:int(round(train_split*num_images))]
X_test = X_data[int(round(train_split*num_images))+1:-1]
y_test = y_data[int(round(train_split*num_images))+1:-1]
# Zero center
means = np.mean(X_train, axis = 0)
X_train -= means
X_test -= means
# Save means (for testing)
np.save('means_ethnic.npy',means)

opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
checkpoint = ModelCheckpoint('weights_ethnic.hdf5', monitor='val_acc', verbose=1, save_best_only=True,
								 save_weights_only=True, mode='max')
model = create_face_network(nb_class=4, hidden_dim=512, shape=(224, 224, 3))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train,
	batch_size=32,
	epochs=10,
	verbose=1,
	callbacks=[checkpoint],
	validation_data=(X_test, y_test),
	shuffle=True,
	class_weight=None,
	sample_weight=None,
	initial_epoch=0)
