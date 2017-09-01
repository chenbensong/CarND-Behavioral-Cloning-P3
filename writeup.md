#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/cnn-architecture.png "Model Visualization"
[image2]: ./examples/center.jpg "center"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I implemented the [Nvidia End-to-End Deep Learning mode](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in keras(model.py lines 103-117). The model consists of 8 hidden layers (3 5x5 convolution layers, with 2x2 stride, two 3x3 convolution layers with 1x1 stride and 3 fully connected layers with 100, 50, and 10 neurons respectively).

The model uses RELU as activation function in each layer to introduce nonlinearity , and the data is first cropped to focus on the bottom half of the image (code line 104) and normalized in the model using a Keras lambda layer (code line 105). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 130-134). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 121).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, generating data by mirroring the image and control, and generating data from off center camera images.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the  [Nvidia End-to-End Deep Learning mode](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because it performed well in real world scenario and should work even better in a CG setting. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the model achieved a good error rate on both training and validation data. 

Then I run the simulator to see how well the car was  driving around track one. The car barely drove through the bridge and immediately drove off the track. I thought there was not enough training data so tried to generate more training data by running the simulator in training mode.  

However, the additional data did not improve the model's performance in simulator. I then spent a lot of time testing different tweaks, but finally found that I used BGR color space in training and RGB color space in testing in simulator. The model worked flawlessly after I fixed this bug.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 104-118) consisted of a convolution neural network with the following layers and layer sizes 3 layers of 5x5 convolution layers, with 2x2 stride, two 3x3 convolution layers with 1x1 stride and 3 fully connected layers with 100, 50, and 10 neurons respectively.

Here is a visualization of the architecture (copied from [Nvidia Blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/))

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer towards center when it was off. These images show what a recovery looks like starting :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would generate more data. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I over-sampled driving through sharp turns by recording a lot of examples on how to drive through a sharpe turn.

I also utilized images recorded from both left and right off center cameras to generate more training samples by adding a small correction angle to the steering angle recored (model.py #53-56).

After the collection process, I had 87888 number of data points.


I finally randomly shuffled the data set and put 80% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by training and validation error stopped to decrease. I used an adam optimizer so that manually training the learning rate wasn't necessary.
