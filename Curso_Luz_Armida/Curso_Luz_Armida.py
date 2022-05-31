

# Convolutional Neural Network is a Deep Learning algorithm specially designed
# for working with Images and videos. It takes images as inputs, extracts and
# learns the features of the image, and classifies them based on the learned features.

# CNN has various filters, and each filter extracts some information from the image such as
# edges, different kinds of shapes (vertical, horizontal, round), and then all of these
# are combined to identify the image.

# The CNN model works in two steps: feature extraction and Classification

# Feature Extraction is a phase where various filters and layers are applied to the images
# to extract the information and features out of it and once it’s done it is passed on
# to the next phase i.e Classification where they are classified based on the target variable of the problem.

#importing the required libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
	print("start my CNN")

	# loading data
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# reshaping data
	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

	# checking the shape after reshaping
	print(X_train.shape)
	print(X_test.shape)

	# normalizing the pixel values
	X_train = X_train / 255
	X_test = X_test / 255

	# defining model
	model = Sequential()
	# adding convolution layer
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	# adding pooling layer
	model.add(MaxPool2D(2, 2))
	# adding fully connected layer
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	# adding output layer
	model.add(Dense(10, activation='softmax'))
	# compiling the model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fitting the model
	model.fit(X_train, y_train, epochs=10)

	#evaluting the model
	model.evaluate(X_test,y_test)
pass