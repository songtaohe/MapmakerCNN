# MapmakerCNN

Run the example:

We put the satellite image and gps image for a 512 meters by 512 meters region in the 'example' folder.

You may run the apply_model_slidingwindow.py file. It will read those images from the 'example folder'.
The output will be the 'output.png'. 

'apply_model_slidingwindow.py' applies the model to the input image with a 2D sliding window. So, your input data could have any size.

The input size of the model here is 256x256, and the output size of the model is 512x512. 
So, the resolution of 'output.png' should be 1 time larger than the input.




Input data:

The resolution of input data should be 1 meter per pixel.

For the gps input, the value of each pixel represents how many gps traces are passing through the corresponding 1 meter by 1 meter cell.

Suppose there are N gps traces passing through a pixel, then the value of this pixel is:  int(log(N * 32)*4)
This value should be smaller than 255.









