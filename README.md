# Face expression and ethnic recognition

This project creates two image models: for face expression recognition and for ethnic classification. The dataset used for training is the [Montreal Set of facial displays of emotion (MSFDE)](http://www.psychophysiolab.com/msfde/terms.php). The images are taken from 32 participants (African, Asian, Caucasian and Hispanic). Each person shows different expressions (neutral, anger, joy, sadness, fear, disgust and shame) with different intensity of the emotion obtaining a total of 580 training images.

## Face expression recognition
For this purpose, the face is isolated of the image and reshape to a fixed size. 
Then, the position of the points of interest of the face (nose, eyes, eyebrows and mouth) have been extracted obtaining a 102 dimensionality vector. SVM is applied in order to create boundaries between the seven classes. 72% accuracy is achieved in the validation set (20%).

## Ethnic recognition
A deep learning approach has been used to tackle this problem. The vgg face architecture explained in this [paper](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) has been used implemented in a Keras [library](https://github.com/rcmalli/keras-vggface). This deep network has been pretrained in 200 million images and eight million unique identities, then, removing the last two fully connected layers, retrained for this problem using stochastic gradient descent with Nesterov momentum. It obtains 95% accuracy in the validation set (30%).

Both approachs may be feasible to test in realtime as they can process around 5 frames per second.


## How to run the code:

### For making the dataset

$ python read_images.py

$ python read_emotions.py

### For expression recognition

$ python train_emotion.py

$ python test_emotions.py -i <image-path>

### For ethnic recognition

$ python train_ethnic.py

$ python test_ethnic.py -i <image-path>

