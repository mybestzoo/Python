# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
'''
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created object/ and background/ subfolders inside train/ and validation/
- put the object pictures from frame 1 in data/train/object
- put the object pictures from frame 10 in data/validation/object
- put the background pictures from frame 1 in data/train/dogs
- put the background pictures from frame 10 in data/validation/dogs
In summary, this is our directory structure:
```
data/
    train/
        object1/
            object1001.jpg
            object1002.jpg
            ...
        object2/
            object2001.jpg
            object2002.jpg
            ...
    validation/
        object1/
            object1001.jpg
            object1002.jpg
            ...
        object2/
            object2001.jpg
            object2002.jpg
            ...
```
'''
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

def save_bottlebeck_features(path, nb_train_samples, save_as_name):
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
    #np.save(open(save_as_name, 'w'), bottleneck_features)
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

# extract features from both training and validation data
bottleneck_features_train = save_bottlebeck_features('data/train', 400, 'bottleneck_features_train.npy')
bottleneck_features_validation = save_bottlebeck_features('data/validation', 400, 'bottleneck_features_validation.npy')

#model = train_FC(bottleneck_features_train, 200)
model = train_SVM(bottleneck_features_train,400)

#validate_FC(model, bottleneck_features_validation, 220)
validate_SVM(model, bottleneck_features_validation, 400)
