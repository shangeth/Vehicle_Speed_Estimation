# Vehicle Speed Estimation

## Task
Given a frame/video of dashboard camera, the task is to predict the Speed of the vehicle.
![](Images/img1.png)
![](Images/img2.png)

## Task Insights
### Convolutional Neural Network:
When we think about this problem in Deep Learning perspective, the first thought is to use a Convolutional Neural Network to extract the features of each frame and use fully connected layers to regress the output(speed).
But the data is not completely useful for this task as the sky or the car dash is not contributing for estimating the speed of the vehicle.
![](
So we crop out the sky and the car dash.


