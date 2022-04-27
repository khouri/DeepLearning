import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CallBack(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epochs, logs={}):
		if(logs.get('accuracy') >= 0.95):
			print("I have reached the expected accuracy!")
			self.model.stop_training = True


def get_model_architecture():

	# First, the images are much larger—300 × 300 pixels—so more layers may be needed.
	# Second, the images are full color, not grayscale, so each image will have three
	# channels instead of one. Third, there are only two image types, so we have a binary
	# classifier that can be implemented using just a single output neuron, where it
	# approaches 0 for one class and 1 for the other
	model = Sequential([
						# notice how we stack several more convolutional layers.
						# We do this because our image source is quite large, and we want, over time,
						# to have many smaller images, each with features highlighted
						#Remember that this is because our input image is 300 × 300 and it’s
						# in color, so there are three channels
						Conv2D(16, (3, 3), activation = tf.nn.relu, input_shape=(300, 300, 3)),
						MaxPooling2D(2, 2),
						Conv2D(32, (3, 3), activation = tf.nn.relu),
						MaxPooling2D(2, 2),
						Conv2D(64, (3, 3), activation = tf.nn.relu),
						MaxPooling2D(2, 2),
						Conv2D(64, (3,3), activation = tf.nn.relu),
						MaxPooling2D(2,2),
						Conv2D(64, (3, 3), activation = tf.nn.relu),
						MaxPooling2D(2, 2),
						Flatten(),
						Dense(128, activation = tf.nn.relu),
						# and we can get a binary classification with just a single
						# neuron if we activate it with a sigmoid function
						Dense(1, activation = tf.nn.sigmoid)
						])

	return(model)
pass

# here we will download using ImageDataGenerator
def get_data_train():

	training_dir = 'horse-or-human/training/'
	train_datagen = ImageDataGenerator(rescale=1 / 255)
	train_generator = train_datagen.flow_from_directory(training_dir,
														target_size=(300, 300),
														class_mode='binary')

	return(train_generator)
pass

def train_model():

	model = get_model_architecture()
	(train_images, train_label), (test_images, test_label) = get_data_train()

	model.compile(optimizer = 'adam',
				  loss= 'sparse_categorical_crossentropy',
				  metrics = ['accuracy'])

	callback = CallBack()
	model.fit(train_images,
			  train_label,
			  epochs = 50,
			  callbacks = [callback])

	model.evaluate(test_images, test_label)
	model.save('saved_models/modelCH03.h5')

	return(model)
pass


if __name__ == '__main__':

	print("Chapter 03_horses_and_humans")
	# trained_model = train_model()
	# modelCH03 = tf.keras.models.load_model('saved_models/modelCH03.h5')
	# (train_images, train_label), (test_images, test_label) = get_data_train()

	# print(modelCH03.summary())
	from download_horses_andhumans import download_data

	download_data()

pass


#In this section we’ll explore a more complex scenario than the Fashion MNIST classifier.
# We’ll extend what we’ve learned about convolutions and convolutional neu‐ ral networks to try
# to classify the contents of images where the location of a feature isn’t always in the same place.
# I’ve created the Horses or Humans dataset for this purpose.

# An interesting side note is that these images are all computer-generated. The theory is that features
# spotted in a CGI image of a horse should apply to a real image. You’ll see how well this works later in this chapter.



