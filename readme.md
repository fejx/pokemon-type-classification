# Pokemon Type classification

This is a project to experiment with [Tensorflow](tensorflow.org) in [Python](python.org). Using Tensorflow, a model is trained that is able to predict the type of a Pokemon by examining it’s sprite (= picture of the Pokemon). The model implemented can only differentiate between Pokemon of the types grass and water.

## What is a Pokemon type?

If you never played Pokemon, all you need to know is that the type of a Pokemon simply is a property assigned to it, which may or may not play a role in the Pokemon`s appearance. Therefore, an image classifier can be used to detect this property.

## How to run

* Get Jupyter Notebook from [here](jupyter.org/install.html).
* Navigate to repo in terminal
* Run `jupyter notebook`
* Run each script files from 1 to 3 to reproduce the whole process by yourself. Downloading the sprites may take some time, but there is a fancy progress bar you can watch in the meantime!

## Project structure

The project consists of three distinct files:

* `1 Download Pokemon sprites.ipynb` contains a script to fetch all Pokemon sprites with specific types
* `2 Prepare Data.ipynb` contains a script to prepare the sprites for training
* `3 Model training.ipynb` contains a script to do the actual training and evaluation

## Findings during training

* At first, I tried to use AlexNet because I did not know any better. As it turns out, it is useless for this classification problem because it has nothing with the classification problem AlexNet solves. All it did was slow down the training process significantly.

* After that, I tried the following network:

	```python
	Sequential([
	    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 4)),
	    MaxPooling2D(),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Flatten(),
	    Dense(512, activation='relu'),
	    Dense(2, activation='sigmoid')
	])
	```

	which resulted in a score of `0.913` and was way faster to train.

* Initially I split the training and testing data using  the `validation_split` parameter of the [ImageDataGenerator](tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) class. But that turned out to be too much automatism for me because I could not see what is happening in the background anymore. Because I wanted to split the data beforehand, I had to split the data by myself into training and testing sets.

* I was confused by the terminology of `validation` in the Tensorflow API. The training dataset should be used to calculate the losses during training, while the validation dataset is only used to validate the score of the network (*which was not done in this project*). This means that the validation dataset is not required during training. Due to the fact that the method `fit_generator` accepts a parameter called `validation_data`, I believe that what Tensorflow calls *validation* is actually the *test* dataset.