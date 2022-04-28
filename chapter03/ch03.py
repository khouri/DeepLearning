from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class CallBack(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epochs, logs={}):
		if(logs.get('accuracy') >= 0.95):
			print("I have reached the expected accuracy!")
			self.model.stop_training = True

def get_model_architecture():

	# To convert this to a convolutional neural network, we simply use convolutional layers
	# in our model definition. We’ll also add pooling layers.
	# In this case, we want the layer to learn 64 convolutions. It will randomly initialize
	# these, and over time will learn the filter values that work best to match the input
	# val‐ ues to their labels. The (3, 3) indicates the size of the filter.

	# Do note, however, that because Conv2D layers are designed for multicolor images, we’re specifying
	# the third dimension as 1, so our input shape is 28 × 28 × 1. Color images will typically have a 3
	# as the third parameter as they are stored as values of R, G, and B.
	model = Sequential([
						keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, input_shape = (28, 28, 1)),
						keras.layers.MaxPooling2D(2,2),
						keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
						keras.layers.MaxPooling2D(2, 2),
						keras.layers.Flatten(),
						Dense(128, activation = tf.nn.relu),
						keras.layers.Dense(10, activation=tf.nn.softmax)
						])

	return(model)
pass

def get_data():

	data = tf.keras.datasets.fashion_mnist
	(train_images, train_label), (test_images, test_label) = data.load_data()

	# There are a few things to note here. Remember earlier when I said that the input
	# shape for the images had to match what a Conv2D layer would expect, and we updated it
	# to be a 28 × 28 × 1 image? The data also had to be reshaped accordingly. 28 × 28 is the
	# number of pixels in the image, and 1 is the number of color channels. You’ll typi‐ cally
	# find that this is 1 for a grayscale image or 3 for a color image, where there are three
	# channels (red, green, and blue), with the number indicating the intensity of that color.
	# So, prior to normalizing the images, we also reshape each array to have that extra dimension.
	# The following code changes our training dataset from 60,000 images, each 28 × 28
	# (and thus a 60,000 × 28 × 28 array), to 60,000 images, each 28 × 28 × 1:
	train_images = train_images.reshape(60000, 28, 28, 1)
	train_images = train_images / 255.0
	test_images = test_images.reshape(10000, 28, 28, 1)
	test_images = test_images / 255.0

	return((train_images, train_label), (test_images, test_label))
pass

def train_model():

	model = get_model_architecture()
	(train_images, train_label), (test_images, test_label) = get_data()

	model.compile(optimizer = 'adam',
				  loss= 'sparse_categorical_crossentropy',
				  metrics = ['accuracy'])

	callback = CallBack()
	model.fit(train_images,
			  train_label,
			  epochs = 50,
			  callbacks = [callback])

	model.evaluate(test_images, test_label)
	model.save('saved_models/model.h5')

	return(model)
pass

if __name__ == '__main__':

	print("Chapter 03")
	# trained_model = train_model()
	modelCH03 = tf.keras.models.load_model('saved_models/modelCH03.h5')
	(train_images, train_label), (test_images, test_label) = get_data()

	print(modelCH03.summary())

	# classifications = model.predict(test_images)
	# print(classifications[0])
	# print(test_label[0])
pass

# In the chapter 02:
# Your neural network was trained on small monochrome images that each contained
# only a single item of clothing, and that item was centered within the image
# To take the model to the next level, you need to be able to detect features in images
# So, for example, instead of looking merely at the raw pixels in the image, what if we
# could have a way to filter the images down to constituent elements? Matching those elements,
# instead of raw pixels, would help us to detect the contents of images more effectively.

# Consider the Fashion MNIST dataset that we used in the last chapter— when detecting a shoe,
# the neural network may have been activated by lots of dark pixels clustered at the bottom of
# the image, which it would see as the sole of the shoe. But when the shoe is no longer centered
# and filling the frame, this logic doesn’t hold.

# One method to detect features comes from photography and the image processing methodologies
# that you might be familiar with. If you’ve ever used a tool like Photo‐ shop or GIMP to sharpen
# an image, you’re using a mathematical filter that works on the pixels of the image. Another word
# for these filters is a convolution, and by using these in a neural network you will create
# a convolutional neural network (CNN).

# In this chapter you’ll learn about how to use convolutions to detect features in an image.
# You’ll then dig deeper into classifying images based on the features within. We’ll explore
# augmentation of images to get more features and transfer learning to take preexisting features
# that were learned by others, and then look briefly into opti‐ mizing your models using dropouts.

# A convolution is simply a filter of weights that are used to multiply a pixel with its neighbors
# to get a new value for the pixel.

# These examples also show that the amount of information in the image is reduced, so we can potentially
# learn a set of filters that reduce the image to features, and those features can be matched to labels
# as before. Previously, we learned parameters that were used in neurons to match inputs to outputs.
# Similarly, the best filters to match inputs to outputs can be learned over time

# Pooling
# Pooling is the process of eliminating pixels in your image while maintaining the semantics of the
# content within the image


# Let’s first take a look at the Output Shape column to understand what is going on here. Our first layer
# will have 28 × 28 images, and apply 64 filters to them. But because our filter is 3 × 3, a 1-pixel border
# around the image will be lost, reducing our overall information to 26 × 26 pixels.

# each maxpooling 2x2 reduces 75% de area of the image

#Each convolution is a 3 × 3 filter, plus a bias. Remember earlier with our dense layers, each layer was Y = mX + c,
# where m was our parameter (aka weight) and c was our bias? This is very similar, except that because the
# filter is 3 × 3 there are 9 parameters to learn. Given that we have 64 convolutions defined, we’ll have 640 overall
# parame‐ ters (each convolution has 9 parameters plus a bias, for a total of 10, and there are 64 of them).


#The MaxPooling layers don’t learn anything, they just reduce the image, so there are no learned
# parameters there—hence 0 being reported.

#The next convolutional layer has 64 filters, but each of these is multiplied across the previous 64 filters,
# each with 9 parameters. We have a bias on each of the new 64 filters, so our number of parameters
# should be (64 × (64 × 9)) + 64, which gives us 36,928 parameters the network needs to learn.

# By the time we get through the second convolution, our images are 5 × 5, and we have 64 of them. If we
# multiply this out we now have 1,600 values, which we’ll feed into a dense layer of 128 neurons. Each neuron
# has a weight and a bias, and we have 128 of them, so the number of parameters the network will learn is
# ((5 × 5 × 64) × 128) + 128, giving us 204,928 parameters.
# Our final dense layer of 10 neurons takes in the output of the previous 128, so the number of parameters
# learned will be (128 × 10) + 10, which is 1,290.

