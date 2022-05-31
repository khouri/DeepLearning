import tensorflow as tf

class CallBack(tf.keras.callbacks.Callback):

	def on_epoch_end(self, epochs, logs={}):
		if(logs.get('accuracy') >= 0.95):
			print("I have reached the expected accuracy!")
			self.model.stop_training = True
	pass
pass
