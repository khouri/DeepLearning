import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# TensorFlow is an open source platform for creating and using machine learning mod‐ els.
# It implements many of the common algorithms and patterns needed for machine learning
if __name__ == '__main__':

    print(tf.__version__)
    print("Hello Keras")

    # When using TensorFlow, you define your layers using Sequential.
    # Inside the Sequential, you then specify what each #layer looks like.
    # We only have one line inside our Sequential, so we have only one layer
    # “Dense” means a set of fully (or densely) connected neurons, which is what you can see in Figure 1-18
    # where every #neuron is connected to every neuron in the next layer.
    # Finally, when you specify the first layer in a neural network (in this case, it’s our only layer),
    # you have to tell #it what the shape of the input data is.
    # In this case our input data is our X, which is just a single value, so we specify that that’s its shape.
    l0 = Dense(units=1, input_shape=[1])
    model = Sequential([l0])

    # In a scenario such as this one, the computer has no idea what the relationship between X and Y is.
    #Armed with this knowledge, the computer can then make another guess.That’s the job of the optimizer.
    # This is where the heavy calculus is used, but with TensorFlow, that can be hidden from you.
    # You just pick the appropriate optimizer to use for differ‐ ent scenarios.
    # In this case we picked one called sgd, which stands for stochastic gradi‐ ent descent—a complex
    # mathematical function that, when given the values, the previous guess, and the results
    # of calculating the errors ( or loss) on that guess, can
    # then generate another one. Over time, its job is to minimize the loss, and by so
    # doing bring the guessed formula closer and closer to the correct answer.
    model.compile(optimizer = 'sgd',
                  loss = 'mean_squared_error')

    # Next, we simply format our numbers into the data format that the layers expect.
    # In Python, there’s a library called #Numpy that TensorFlow can use, and here we put our numbers
    # into a Numpy array to make it easy to process them
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # The learning process will then begin with the model.fit command, like this:
    # You can read this as “fit the Xs to the Ys, and try it 500 times.”
    model.fit(xs, ys, epochs=10)

    # Our last line of code then used the trained model to get a prediction like this
    print(model.predict([10.0]))
    
    # I can print out the values (or weights) that the layer learned.
    print("Here is what I learned: {}".format(l0.get_weights()))
pass