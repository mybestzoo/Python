import os
import h5py
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import cv2



def save_bottlebeck_features(path, nb_train_samples):
    data_dir = path
    #nb_train_samples = nb_train_samples
    # dimensions of our images.
    img_width, img_height = 50, 50
	
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists('vgg16_weights.h5'), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File('vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=1,
            class_mode=None,
            shuffle=False)
    tic = time.time()
    bottleneck_features = model.predict_generator(generator, nb_train_samples)
    tac = time.time()
    print 'Feature extraction time:', (tac-tic)
    return bottleneck_features

def save_bottlebeck_features2(im):
    #nb_train_samples = nb_train_samples
    # dimensions of our images.
    img_width, img_height = 50, 50
	
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists('vgg16_weights.h5'), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File('vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
	
	#prepare image
    img = cv2.resize(im,(img_width, img_height)).astype(np.float32)
    cv2.imshow("frame",img)
    #img = img.transpose((2,0,1))
    #img = np.expand_dims(img, axis=0)
	
    tic = time.time()
    bottleneck_features = model.predict(img)
    tac = time.time()
    print 'Feature extraction time:', (tac-tic)
    return bottleneck_features	
	
def train_SVM(train_data,nb_train_samples):
	X = np.zeros((nb_train_samples,512))
	for i in range(nb_train_samples):
		X[i,:] = train_data[i,:,0,0]
	train_labels = np.array([0] * (nb_train_samples / 4) + [1] * (nb_train_samples / 4) + [2] * (nb_train_samples / 4) + [3] * (nb_train_samples / 4))
	
	model = SVC(kernel='poly', gamma=1, C=1)
	
	tic = time.time()
	model.fit(X, train_labels)
	tac = time.time()
	
	print 'Training time:', (tac-tic)
	return model
	
	
def validate_SVM(model, validation_data, nb_validation_samples):
	X = np.zeros((nb_validation_samples,512))
	for i in range(nb_validation_samples):
		X[i,:] = validation_data[i,:,0,0]
	validation_labels = np.array([0] * (nb_validation_samples / 4) + [1] * (nb_validation_samples / 4) + [2] * (nb_validation_samples / 4) + [3] * (nb_validation_samples / 4))
	prediction = model.predict(X)
	print prediction
	print 'Confusion matrix:', confusion_matrix(validation_labels,prediction)

"""
frame = cv2.imread('001.jpg')	

bb_r,h,bb_c,w = 532,50,290,50

im = frame[bb_c:bb_c+w,bb_r:bb_r+h]
cv2.imshow("frame",frame[bb_c:bb_c+w,bb_r:bb_r+h])

features = save_bottlebeck_features2(im)

plt.plot(features[0,:,0,0])
plt.show()
"""

# TRAIN THE MODEL
# extract features from both training and validation data
bottleneck_features_train = save_bottlebeck_features('data/train', 400)

#model = train_FC(bottleneck_features_train, 200)
model = train_SVM(bottleneck_features_train,400)

#------------------------------------------------------------------------------------------------------------------------

cap = cv2.VideoCapture('film1.avi')
# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1288,728))

frameid = 1

while(1):
	ret ,frame = cap.read()
	
	if ret == True:
		
		frameid += 1
	
		# parse file and extract measurements
		infile = open("boxes1.txt", "r")
		lines = infile.readlines()
		for i in xrange(len(lines)):
			line = lines[i].strip()
			fields = line.split(" ")
			if int(fields[0]) == frameid and len(fields) == 1 and frameid%1 == 0:
				print frameid
				j=i+1
				while len(lines[j])>4:
					l = lines[j].strip()
					f = l.split(" ")
					r,h,c,w = int(f[0]),int(f[2])-int(f[0]),int(f[1]),int(f[3])-int(f[1])
					q = max([w,h])
					im = frame[c:c+q,r:r+q]
					cv2.imwrite("buffer/buffer/buffer.jpg", frame[c:c+q,r:r+q])
					#features = save_bottlebeck_features(im)
					features = save_bottlebeck_features('buffer', 1)
					X = np.zeros((1,512))
					X[0,:] = features[0,:,0,0]
					prediction = model.predict(X)
					if prediction == 0:
						cv2.rectangle(frame, (r, c), (r+h, c+w), (255,0,0), 3)
					elif prediction == 1:
						cv2.rectangle(frame, (r, c), (r+h, c+w), (0,255,0), 3)
					elif prediction == 2:
						cv2.rectangle(frame, (r, c), (r+h, c+w), (0,0,255), 3)
					else:
						cv2.rectangle(frame, (r, c), (r+h, c+w), (0,0,0), 3)
					j += 1
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		#cv2.imshow("frame",frame)
		
		# write the frame
		out.write(frame)
		
		key = cv2.waitKey(50) & 0xff
	
		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break
	else:
		break
		
cv2.destroyAllWindows()
cap.release()

"""
# setup initial location of window
r1,h1,c1,w1 = 300,200,630,240
r2,h2,c2,w2 = 295,70,505,70
r3,h3,c3,w3 = 300,70,870,80
r4,h4,c4,w4 = 280,80,950,150
"""
