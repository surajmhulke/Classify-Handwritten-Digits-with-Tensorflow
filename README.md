# Classify-Handwritten-Digits-with-Tensorflow

One of the capabilities of deep learning is image recognition, The “hello world” of object recogniton for machine learning and deep learning is the MNIST dataset for handwritten digit recognition.
![image](https://github.com/surajmhulke/Classify-Handwritten-Digits-with-Tensorflow/assets/136318267/f6cb2216-ccf3-4d88-9551-59eca7e0f0f4)


Sample Digits from MNIST dataset


Description of the MNIST Handwritten Digit.
The MNIST Handwritten Digit is a dataset for evaluating machine learning and deep learning models on the handwritten digit classification problem, it is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

#  Import the TensorFlow library
import tensorflow as tf # Import tensorflow library
import matplotlib.pyplot as plt # Import matplotlib library
Create a variable named mnist, and set it to an object of the MNIST dataset from the Keras library and we’re gonna unpack it to a training dataset (x_train, y_train) and testing dataset (x_test, y_test):

 
mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data

# Preprocess the data
To make sure that our data was imported correctly, we are going to plot the first image from the training dataset using matplotlib:

plt.imshow(x_train[0], cmap="gray") # Import the image
plt.show() # Plot the image
![image](https://github.com/surajmhulke/Classify-Handwritten-Digits-with-Tensorflow/assets/136318267/f9a974e7-8f18-49aa-a0d6-15c09634814e)

# Image from MNIST dataset
Before we feed the data into the neural network we need to normalize it by scaling the pixels value in a range from 0 to 1 instead of being from 0 to 255 and that make the neural network needs less computational power:

# Normalize the train dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)

# Normalize the test dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)

#  Build the model
Now, we are going to build the model or in other words the neural network that will train and learn how to classify these images.

It worth noting that the layers are the most important thing in building an artificial neural network since it will extract the features of the data.

First and foremost, we start by creating a model object that lets you add the different layers.

Second, we are going to flatten the data which is the image pixels in this case. So the images are 28×28 dimensional we need to make it 1×784 dimensional so the input layer of the neural network can read it or deal with it. This is an important concept you need to know.

Third, we define input and a hidden layer with 128 neurons and an activation function which is the relu function.

And the Last thing we create the output layer with 10 neurons and a softmax activation function that will transform the score returned by the model to a value so it will be interpreted by humans.

#Build the model object
model = tf.keras.models.Sequential()

# Add the Flatten Layer
model.add(tf.keras.layers.Flatten())

# Build the input and the hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Build the output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

Compile the model
Since we finished building the neural network we need to compile the model by adding some few parameters that will tell the neural network how to start the training process.

First, we add the optimizer which will create or in other word update the parameter of the neural network to fit our data.

Second, the loss function that will tell you the performance of your model.

Third, the Metrics which give indicative tests of the quality of the model.

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
Train the model
We are ready to train our model, we call the fit subpackage and feed it with the training data and the labeled data that correspond to the training dataset and how many epoch should run or how many times should make a guess.

model.fit(x=x_train, y=y_train, epochs=5) # Start training process
![image](https://github.com/surajmhulke/Classify-Handwritten-Digits-with-Tensorflow/assets/136318267/27a7e0e7-e85a-4be9-948f-3be3a69e0e14)

# Training Process
Evaluate the model
Let’s see how the model performs after the training process has finished.
![image](https://github.com/surajmhulke/Classify-Handwritten-Digits-with-Tensorflow/assets/136318267/41a85507-46c2-4e60-bda4-69843378ff60)

# Evaluate the model performance
test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
# Print out the model accuracy 
print('\nTest accuracy:', test_acc)
![image](https://github.com/surajmhulke/Classify-Handwritten-Digits-with-Tensorflow/assets/136318267/ee73d332-bd49-422d-bf2c-7ec4d88115cf)

Evaluating the Model Performance
It shows that the neural network has reached 97.39% accuracy which is pretty good since we train the model just with 5 epochs.

Make predictions
Now, we will start making a prediction by importing the test dataset images.

predictions = model.predict([x_test]) # Make prediction
We are going to make a prediction for numbers or images that the model has never seen before. 

For instance, we try to predict the number that corresponds to the image number 1000 in the test dataset:

print(np.argmax(predictions[1000])) # Print out the number

Prediction
![image](https://github.com/surajmhulke/Classify-Handwritten-Digits-with-Tensorflow/assets/136318267/4811b3a2-ddc9-45a1-b7be-cbcf7d35e8c0)

As you see, the prediction is number nine but how we can make sure that this prediction was true? well, we need to plot the image number 1000 in the test dataset using matplotlib:

plt.imshow(x_test[1000], cmap="gray") # Import the image
plt.show() # Show the image
![image](https://github.com/surajmhulke/Classify-Handwritten-Digits-with-Tensorflow/assets/136318267/d3a6e6a3-9ad0-443e-8511-28aed3c63f9b)

The Correct Prediction
Congratulations, The prediction was correct and that being said that our model works correctly and well for classifying Handwritten-images.
