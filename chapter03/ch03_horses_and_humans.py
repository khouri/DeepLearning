import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop
from chapter03.Data_Generator import Data_Generator
from chapter03.Download_File_From_Web import Download_File_From_Web


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

def create_data_dir():

	train_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
	train_file_name = "horse-or-human.zip"
	training_dir = 'horse-or-human/training/'
	web_dowloader = Download_File_From_Web(train_url, train_file_name)
	web_dowloader.download_data_to(training_dir)

	val_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
	val_file_name = "validation-horse-or-human.zip"
	validation_dir = 'horse-or-human/validation/'
	web_dowloader = Download_File_From_Web(val_url, val_file_name)
	web_dowloader.download_data_to(validation_dir)

	return(training_dir, validation_dir)
pass

def train_model(model, train_data_generator, val_data_generator):

	saved_model_path = """saved_models/model_horses_humans.h5"""
	model.compile(loss = 'binary_crossentropy',
				  optimizer = RMSprop(lr=0.001),
				  metrics = ['accuracy'])

	history = model.fit_generator(train_data_generator,
								  epochs = 15,
								  validation_data = val_data_generator)

	model.save('saved_models/model_horses_humans.h5')

	return(saved_model_path)
pass


if __name__ == '__main__':

	print("Chapter 03_horses_and_humans")
	training_dir, validation_dir = create_data_dir()

	model = get_model_architecture()
	train_generator = Data_Generator().get_data_generator(training_dir)
	val_generator = Data_Generator().get_data_generator(validation_dir)

	saved_model_path = train_model(model, train_generator, val_generator)

	# model = tf.keras.models.load_model('saved_models/model_horses_humans.h5')
pass


#In this section we’ll explore a more complex scenario than the Fashion MNIST classifier.
# We’ll extend what we’ve learned about convolutions and convolutional neu‐ ral networks to try
# to classify the contents of images where the location of a feature isn’t always in the same place.
# I’ve created the Horses or Humans dataset for this purpose.

# An interesting side note is that these images are all computer-generated. The theory is that features
# spotted in a CGI image of a horse should apply to a real image. You’ll see how well this works later in this chapter.

# Note how, by the time the data has gone through all the convolutional and pooling layers, it ends
# up as 7 × 7 items. The theory is that these will be activated feature maps that are relatively simple,
# containing just 49 pixels. These feature maps can then be passed to the dense neural network to match them
# to the appropriate labels.