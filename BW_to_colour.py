
from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.layers.convolutional import UpSampling2D
#from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image 
import theano
theano.config.optimizer="None"
from sklearn.cross_validation import train_test_split
#os.chdir("/media/nandan/local disk F")
import cv2
import os
from PIL import Image
#x = np.zeros(shape=(64,64))
num=43
m,n=96,96
path1="imagenet1"
num_image = os.listdir(path1)
epsilon = 0.001                         #used in loss function MSE

# loading images and saving it in its B&W and Colour part
def load_images_from_folder(folder):
    images = []
    gray = []
    colour = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), )
        img = cv2.resize(img,(m,n))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,u,v  = cv2.split(img)
        u_v = cv2.merge((u,v))
        if img is not None:
        	gray.append(l[:])
        	colour.append(u_v[:])
        	images.append(img[:])
    return gray,colour,images

#Finding the mean square error
def MSE(y_pred, y_out):
	u_loss = np.add((y_pred[:,:,0]-y_out[:,:,0])**2)
	v_loss = np.add((y_pred[:,:,1]-y_out[:,:,1])**2)
	loss = epsilon*( u_loss + v_loss)
	return loss


x_gray, y_colour, image1=load_images_from_folder(path1)
x_gray=np.array(x_gray)
y_colour=np.array(y_colour)
y_colour = y_colour*0.196078431
y_colour = y_colour.astype(int)

x_gray=x_gray.reshape((num,m,n,1))
y_colour=y_colour.reshape((num,m,n,2))
y_colour=np.array(y_colour)

test = x_gray[9]
test = test.reshape((1,m,n,1))

uv_1 = y_colour[:,:,:,0]
a = np.amin(uv_1)
print(a)
uv_1 = uv_1.reshape((num, m, n, 1))

batch_size= 32
#nb_classes = 5      #no. of classes of output
nb_epoch=20          # no. of iteration
nb_filters=64
np_pool=2
nb_conv=3


#Defining the Model
model = Sequential()

model.add(Convolution2D(nb_filters,nb_conv, strides=1, padding ='same' ,activation='relu', input_shape=(m,n,1)))
model.add(Convolution2D(nb_filters,nb_conv, strides=1, padding ='same' ,activation='relu')) 
model.add(MaxPooling2D(pool_size=(np_pool,np_pool)))


model.add(Convolution2D(2*nb_filters,nb_conv, strides=1, padding ='same' ,activation='relu')) 
model.add(Convolution2D(2*nb_filters,nb_conv, strides=1, padding ='same' ,activation='relu')) 
model.add(MaxPooling2D(pool_size=(np_pool,np_pool)))


model.add(Convolution2D(4*nb_filters,nb_conv, strides=1, padding ='same' ,activation='relu')) 
model.add(Convolution2D(4*nb_filters,nb_conv, strides=1, padding ='same' ,activation='relu')) 
model.add(MaxPooling2D(pool_size=(np_pool,np_pool)))

model.add(UpSampling2D(2))
model.add(Conv2DTranspose(4*nb_filters,nb_conv, strides=1, padding ='same', activation='relu'))
model.add(Conv2DTranspose(4*nb_filters,nb_conv, strides=1, padding ='same', activation='relu'))

model.add(UpSampling2D(2))
model.add(Conv2DTranspose(2*nb_filters,nb_conv, strides=1, padding ='same', activation='relu'))
model.add(Conv2DTranspose(2*nb_filters,nb_conv, strides=1, padding ='same', activation='relu'))

model.add(UpSampling2D(2))
model.add(Conv2DTranspose(nb_filters,nb_conv, strides=1, padding ='same', activation='relu'))
model.add(Conv2DTranspose(nb_filters,nb_conv, strides=1, padding ='same', activation='relu'))
model.add(Conv2DTranspose(2, nb_conv, strides=1, padding ='same', activation='relu'))

# model.add(Flatten())
# model.add(Dense(50))
# model.add(Activation('relu'))
# model.add(Dense(1))

model.summary()
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='rmsprop', loss='MSE', metrics=['accuracy'])
model.fit(x_gray, y_colour, epochs=10, batch_size=120,validation_split=0.2 ) 
# model.save_weights('weights')

prediction = model.predict(test)
test = test[0,:,:,0]
print(test.shape)
print(prediction.shape)
u_pred = prediction[0,:,:,0]
u_pred = u_pred*255/50
v_pred = prediction[0,:,:,1]
v_pred = v_pred*255/50

data1= test.reshape((m,n,1))
data2= u_pred.reshape((m,n,1))
data3= v_pred.reshape((m,n,1))

data11 = np.concatenate((data1,data2,data3),axis = 2)
print(data11.shape)
img = Image.fromarray(data11, 'RGB')
img2 = Image.fromarray(image1[9],'RGB')
img.save('abs.jpeg')
img2.save('abs2.jpeg')
# img.show()

