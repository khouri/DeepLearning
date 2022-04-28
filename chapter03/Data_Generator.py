from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Data_Generator():

	# here we will download using ImageDataGenerator
	def get_data_generator(self, data_path):

		train_datagen = ImageDataGenerator(rescale=1 / 255)
		data_generator = train_datagen.flow_from_directory(data_path,
															target_size=(300, 300),
															class_mode='binary')

		return(data_generator)
	pass

pass