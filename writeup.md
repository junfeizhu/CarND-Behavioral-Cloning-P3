# Behavioral Cloning Project

## Model Architecture 
My model has 5 convolution layers and 4 fully connectd layers. Before the first convolution layer, the image was cropped 60 pixels from the top and 20 pixels from the bottom. The model architecture was shown below:

|               | Layer     |
|---            |---        |
|               |Input       | 
|  Layer 1      | Convolution 5x5, output 24 channels |
|               | Relu       |              
|               |subsample    |                  
|   Layer 2     | Convolution 5x5, output 36 channels |
|               | Relu       |              
|               |subsample    |                    
|   Layer 3     | Convolution 5x5, output 48 channels |
|               | Relu       |               
|               |subsample    |  
|   Layer 4     | Convolution 3x3, output 64 channels |
|               | Relu       |               
|   Layer 5     | Convolution 3x3, output 64 channels |
|               | Relu       |               
|   Layer 6    | Fully connected, output 100 nodes |
|               |Relu    |   
|   Layer 7    | Fully connected, output 50 nodes |
|               |Relu    |  
|   Layer 8    | Fully connected, output 10 nodes |
|               |Relu    |  
|   Layer 9    | Fully connected, output 1 node |


I didn't use a dropout layer in the model. Instead, to prevent overfitting, I only trained the model for 3 epochs. The model used an adam optimizer, so the learning rate was not tuned manually

## Creating the training and validation dataset
I used the dataset provided by Udacity. Dataset was splitted into 80% training set and 20% validation set. For each set, I used all three camera images. For the images taken from the left camera, I added 0.2 to the angle. For the images taken from the left camera, I subtracted 0.2 from the angle. Then I flipped all the images and timed the angle by -1. After this data augmentation. I actually have 5 times more data than just the original center images.


## Model testing
I used the simulator to test the model and the result was very good. The car was always in the center of the lane. Some images are shown below:

![image1](https://raw.githubusercontent.com/junfeizhu/CarND-Behavioral-Cloning-P3/master/example_images/2018_04_08_14_28_37_543.jpg)

![image2](https://raw.githubusercontent.com/junfeizhu/CarND-Behavioral-Cloning-P3/master/example_images/2018_04_08_14_29_40_713.jpg)

![image3](https://raw.githubusercontent.com/junfeizhu/CarND-Behavioral-Cloning-P3/master/example_images/2018_04_08_14_30_40_680.jpg)

![image4](https://raw.githubusercontent.com/junfeizhu/CarND-Behavioral-Cloning-P3/master/example_images/2018_04_08_14_31_05_150.jpg)