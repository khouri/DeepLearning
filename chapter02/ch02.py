import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class CallBack(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epochs, logs={}):
		if(logs.get('accuracy') >= 0.95):
			print("I have reached the expected accuracy!")
			self.model.stop_training = True

#If you remember, in Chapter 1 we had a Sequential model to specify that we had many layers.
# It only had one layer, but in this case, we have multiple layers.
def get_model_architecture():

	# The first, Flatten, isn’t a layer of neurons, but an input layer specification.
	# Our inputs are 28 × 28 images, but we want them to be treated as a series of numeric val‐ ues,
	# like the gray boxes at the top of Figure 2-5. Flatten takes that “square” value (a 2D array)
	# and turns it into a line (a 1D array).

	# The next one, Dense, is a layer of neurons, and we’re specifying that we want 128 of them. This is the
	# middle layer shown in Figure 2-5. You’ll often hear such layers described as hidden layers. Layers that are
	# between the inputs and the outputs aren’t seen by a caller, so the term “hidden” is used to describe them.
	# We’re asking for 128 neurons to have their internal parameters randomly initialized. Often the question
	# I’ll get asked at this point is “Why 128?”

	# This is entirely arbitrary—there’s no fixed rule for the number of neurons to use. As you design the
	# layers you want to pick the appropriate number of values to enable your model to actually learn.
	# More neurons means it will run more slowly, as it has to learn more parameters. More neurons could
	# also lead to a network that is great at recognizing the training data, but not so good at recognizing
	# data that it hasn’t previously seen (this is known as overfitting, and we’ll discuss it later in this chapter).
	# On the other hand, fewer neurons means that the model might not have sufficient parameters to learn.
	model = keras.Sequential([
								keras.layers.Flatten(input_shape = (28, 28)),
								keras.layers.Dense(128, activation = tf.nn.relu),
								keras.layers.Dense(10, activation=tf.nn.softmax)
							 ])

	return(model)
pass

def get_data():
	# Keras has a number of built-in datasets that you can access with a single line of code like this.
	# In this case you don’t have to handle downloading the 70,000 images—split‐ ting them into training
	# and test sets, and so on—all it takes is one line of code. This methodology has been improved upon
	# using an API called TensorFlow Datasets, but for the purposes of these early chapters, to reduce
	# the number of new concepts you need to learn, we’ll just use tf.keras.datasets.
	data = tf.keras.datasets.fashion_mnist
	(train_images, train_label), (test_images, test_label) = data.load_data()

	# Python allows you to do an operation across the entire array with this notation. Recall that all of
	# the pixels in our images are grayscale, with values between 0 and 255. Dividing by 255 thus ensures
	# that every pixel is represented by a number between 0 and 1 instead. This process is called
	# normalizing the image
	train_images = train_images / 255.0
	test_images = test_images / 255.0

	return((train_images, train_label), (test_images, test_label))
pass

def train_model():

	model = get_model_architecture()
	(train_images, train_label), (test_images, test_label) = get_data()

	# Choosing loss and optimizer is an artwork more than science
	# Our item of clothing will belong to 1 of 10 categories of clothing, and thus using a
	# categorical loss function is the way to go. Sparse categorical cross entropy is a good choice.

	# The same applies to choosing an optimizer. The adam optimizer is an evolution of the stochastic
	# gradient descent (sgd) optimizer we used in Chapter 1 that has been shown to be faster and more efficient.
	model.compile(optimizer = 'adam',
				  loss= 'sparse_categorical_crossentropy',
				  metrics = ['accuracy'])

	callback = CallBack()
	model.fit(train_images,
			  train_label,
			  epochs = 50,
			  callbacks = [callback])

	model.evaluate(test_images, test_label)
	model.save('saved_models/modelCH02.h5')

	return(model)
pass


if __name__ == '__main__':
	print("Chapter 02")
	trained_model = train_model()
	modelCH02 = tf.keras.models.load_model('saved_models/modelCH02.h5')
	(train_images, train_label), (test_images, test_label) = get_data()

	# You’ll notice that the classification gives us back an array of values.
	# These are the val‐ ues of the 10 output neurons. The label is the actual label for the
	# item of clothing, in this case 9. Take a look through the array—you’ll see that some of
	# the values are very small, and the last one (array index 9) is the largest by far.
	# These are the probabilities that the image matches the label at that particular index.
	# So, what the neural network is reporting is that there’s a 91.4% chance that the item
	# of clothing at index 0 is label 9. We know that it’s label 9, so it got it right.
	classifications = modelCH02.predict(test_images)
	print(classifications[0])
	print(test_label[0])
pass

#In Chapter 1, you saw a very simple scenario where a machine was given a set of X and Y values, and
# it learned that #the relationship between these was Y = 2X – 1. This was done using a very simple
# neural network with one layer and #one neuron.
#Each of our images is a set of 784 values (28 × 28) between 0 and 255.
# They can be our X. We know that we have 10 #different types of images in our dataset, so let’s con‐ sider
# them to be our Y. Now we want to learn what the #function looks like where Y is a function of X.

# Differences between numbers x images classification:
# Given that we have 784 X values per image, and our Y is going to be between 0 and 9, it’s pretty clear
# that we cannot do Y = mX + c as we did earlier.
# But what we can do is have several neurons working together. Each of these will learn
# parameters, and when we have a combined function of all of these parameters work‐ ing together


# You might notice that there’s also an activation function specified in that layer. The activation function
# is code that will execute on each neuron in the layer. TensorFlow supports a number of them, but a very
# common one in middle layers is relu, which stands for rectified linear unit. It’s a simple function that
# just returns a value if it’s greater than 0. In this case, we don’t want negative values being passed to the
# next layer to potentially impact the summing function, so instead of writing a lot of if-then code, we can
# simply activate the layer with relu.

#Finally, there’s another Dense layer, which is the output layer. This has 10 neurons, because we have 10 classes.
# #Each of these neurons will end up with a probability that the input pixels match that class, so our job is to
# determine which one has the highest value. We could loop through them to pick that value, but the softmax
# activation function does that for us.


# The network is doing much better with the training data, but it’s not necessarily a better model.
# In fact, the divergence in the accuracy numbers shows that it has become overspecialized to the
# training data, a process often called overfitting.