import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
	print("chapter 03")

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


