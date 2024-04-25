# Siamese-NN-Using-One-Shot-Learning

## Purpose of this work:
Enabling students to experiment with building a convolutional neural net, using it on a real-world dataset and problem. In addition to practical knowledge in the “how to” of building the network, an additional goal is the integration of useful logging tools for gaining better insight into the training process. Finally, the students are expected to read, understand and (loosely) implement a scientific paper.

link to the paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## Task: Facial Recognition

## Table of Contents:
1. Data Analysis
2. Project Structure
3. Model Architecture
4. Model Initialization
5. Hyperparameters
6. Stopping Conditions
7. Experiments
8. Results and Evaluation
9. Conclusions

## Part 1: Data Analysis
The raw data we received as part of the assignment is actually compressed images inside a zip file and txt files. After reading the readme file of the Labeled Faces in the dataset and extracting the files, the data can be described as follows:

- The data is actually the images, sized 250x250 pixels, so they are divided by people. Each person has one or more images, which are numbered starting from 0001 (0002, 0003 and so on).
- The training and test sets are actually defined by the txt files.

Structure of the txt file:
On the first line - a number describing the number of records in the set of each type - match (matching images of the same person) and mismatch (two different images of different people).
After that, there are records according to the above number for each type, so that the structure of each record is:

A record describing a match:
person_name img1_num img2_num

A record describing a mismatch: 
person1_name img1_num person2_name img2_num

Therefore, we performed the following preprocessing on the data:
- Resized the image to 105 x 105 pixels. In addition, for the training set we also added the image channels: 1.
- Converted the images into matrices containing the pixel values - the images will enter the model we build later in this format.

We will note a few basic features of the data:
- Total number of images: 13233
- Number of records in the training set: 2200  
- Number of matching image records: 1100
- Number of non-matching image records: 1100
- Number of records in the test set: 1000
- Number of matching image records: 500
- Number of non-matching image records: 500

Here are examples of the data, from the training set:

- A record describing a match between two images (same person in the image):
Aaron_Peirsol 1 2

![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/20f77b15-8b31-48a8-aecd-be857d023641)
![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/8f582d56-0773-403e-8807-070dd055923d)

- A record describing a mismatch between two images (2 different people in each image):
AJ_Cook 1 Marsha_Thomason 1  

![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/9f7a3a1c-8c8f-432a-9e76-caf50ba24f76)
![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/894b19dc-0a4a-436f-8abc-033cdfe99e76)

## Part 2: Project Structure
This part of the report is intended to give an overview of the files and folders in the project. We will expand on their contents later.

Folders:
NOTE - we didnt attached this directories.
data directory - A folder containing the images themselves, the txt files we downloaded and the pkl files we created (which contain the images after preprocessing).
logs directory - In addition to runtime prints and saving results and graphs, we chose to add a folder that will save log records at certain important points so that we can trace back and try to understand the process the model went through.
results directory - A folder that will contain the results of the experiments and the graphs describing the training and testing processes.

Files:

load_data.py - This file is responsible for loading the data, preprocessing it, and then saving the data in pkl files. It also contains an initial analysis of the data.

logger.py - Contains initialization of the logger's logging settings according to the format we chose (wherever we chose to use the logging option, we initialized it using this file).

requirements.txt - A file with the requirements of the packages we used, so that if someone wants to run the project on their machine, they can make sure they have all the required packages.

siamese_network.py - A class that implements the model of the network, also contains weight initialization, building the CNN architecture, and performing forward as written in the paper.

trainer.py - Contains a class built for training and testing. We chose to separate the actual training stage from running the experiments in order to create order, clarity, and convenience (we did not want the experiment run file to be too cluttered or vice versa). It is important to note that in this file, we load the data from the pkl files, and the prediction on the test set is also performed.

run_experiments.py - A file for running experiments. It contains a "grid search" that runs combinations of hyperparameters of our choice.

## Part 3: Model Architecture
The model can be divided into two parts that connect during the forward phase. The first part is the convolutional network, and the second is the "prediction" network. In general, our implementation is based on a single convolutional network, which is run twice during prediction to obtain two temporary outputs, and then a distance calculation is performed between these outputs, and the result of the calculation is input into the second part of the network - the "prediction" network.

We logically divided the structure of the convolutional network into "blocks":

- Block 1:
    - Input size: 105x105 pixel images. Each image has a single channel.
    - Components: First, there is a convolutional layer with 64 kernels, where each kernel is 10x10.
    After the kernel operation, 96x96 pixel images are obtained (multiplied by the number of kernels - 64).
    Finally, there is a normalization layer, an activation layer (ReLU), and a max pooling layer.
    - Output size: 48x48 pixel images, 64 channels per image.

- Block 2:
    - Input size: 48x48 pixel images, 64 channels per image (output size of Block 1).
    - Components: First, there is a convolutional layer with 128 kernels, where each kernel is 7x7.
    After the kernel operation, 42x42 pixel images are obtained (multiplied by the number of kernels - 128).
    Finally, there is a normalization layer, an activation layer (ReLU), and a max pooling layer.
    - Output size: 21x21 pixel images, 128 channels per image.

- Block 3:
    - Input size: 21x21 pixel images, 128 channels per image (output size of Block 2).
    - Components: First, there is a convolutional layer with 128 kernels, where each kernel is 4x4.
    After the kernel operation, 18x18 pixel images are obtained (multiplied by the number of kernels - 128).
    Finally, there is a normalization layer, an activation layer (ReLU), and a max pooling layer.
    - Output size: 9x9 pixel images, 128 channels per image.

- Block 4:
    - Input size: 9x9 pixel images, 128 channels per image (output size of Block 3).
    - Components: First, there is a convolutional layer with 256 kernels, where each kernel is 4x4.
    After the kernel operation, 6x6 pixel images are obtained (multiplied by the number of kernels - 256).
    Finally, there is a normalization layer, an activation layer (ReLU), and no pooling layer.
    - Output size: 6x6 pixel images, 256 channels per image.

- Block 5:
    The last block in the convolutional network is a channel flattening block (flatten), where the flattening creates a vector of size 6*6*256 = 9216. We connect a dense layer of size 4096 to this layer, and then we perform a sigmoid activation.

Explanation of how the convolutional network integrates with the "prediction" network as part of the Siamese network model we built, and an explanation of the forward function:

The input is essentially doubled - meaning two images.

Components: Each input is separately entered into the convolutional network, and the output is saved separately for each.
Then, using torch.abs, the "distance" (L1 distance) between these vectors is calculated (they are of the same size - 4096 on 1).
After that, the L1_distance vector enters the "prediction" network.

Output: The probability of a match between the input images.

Note: We wrapped all the blocks and layers with nn.Sequential.

## Part 4: Model Initialization
Model initialization refers to two aspects: setting the random seed and initializing the weights.

- In order to fix the seed, we defined a value for it in a field of the model class. In addition, we ran the setup_seeds function, which initializes all the random generators for all the packages we use in the model: torch, random, numpy random.

- For weight initialization, we used the setup_weights function which is called in the constructor. This function initializes weights based on the layer type: Conv/Linear. Initialization of convolutional layers is performed according to what is written in the paper, but we slightly modified the initial weights of the Linear layers in order to achieve better performance based on the runs we performed.

## Part 5: Hyperparameters 
Fixed Hyperparameters:
- Validation set size: We chose to use a common size of 20%. In order to allow the model to train "more", we tried a set size of 15%, but we saw that there was no significant difference in the results.
- Optimizer: We chose to use Adam. We performed a few individual experiments with an SGD optimizer, but we got consistently worse results for it.
- Loss function: Binary Cross Entropy - according to the paper.

Variable Hyperparameters:  

- Batch size: We chose to examine sizes of 16 and 32 as opposed to the common (32 and 64), because the amount of data we have for training is relatively small (2200 records) and we wanted to allow the model to train more times in each epoch.

- Learning rate: We used learning rates of 0.001 and 0.005.
It is important to note that we used the StepLR scheduler object, where every 5 epochs we set the learning rate to be 95% of its current value: new_lr = old_lr * 0.95. This was with the idea that as we approach the optimum, we want the learning to be slower and more gradual and less volatile (volatility makes it difficult to converge).

- Regularization coefficient: Out of concern for converging to overfitting, we thought to try using regularization. However, since the amount of data for training is small, we must also avoid underfitting, so we chose small values for this coefficient: [0.0, 0.0001, 0.0005]. 

- Number of epochs: We chose to run different experiments for 15, 25, 50 epochs with the idea that on one hand we want the model to have enough time to train, and on the other hand, we don't want it to overtrain. This is especially since the amount of training data is small, and then the likelihood that in each epoch the model will validate on data it has already seen increases, and thus the risk of overfitting increases.

- Dropout rate: Out of concern for converging to overfitting, we thought to use the option of dropping out some of the information passing through the network. However, as mentioned, the number of records for training is not large, so this concern is not significant, and accordingly, we chose to examine small values for this parameter [0.0, 0.2].

## Part 6: Stopping Conditions

As part of the training stage, we implemented a mechanism for early stopping of the training process, which works as follows:
We defined a field in the trainer class called patience, which defines the number of consecutive epochs in which no improvement is seen in the validation accuracy.

After each epoch, we checked whether there was an improvement in this metric, and if there was indeed an improvement, we reset the counter that is getting closer to 0 as there are no improvement steps.
When the counter reaches 0, the training process stops.
if validation accuracy improved (better than best epoch)
    then reset patience counter
else patience counter-=1

As part of running the runs throughout the work, we saw that this condition stops the training process too early, before the training reaches its peak performance, and thereby hurts the quality of the model.
It is possible that a more complex stopping condition would succeed in achieving better results along with shortening the training process time.

## Part 7: Experiments
During the work, we tried to build our code as modular as possible so that we could try as many parameters as possible. In practice, given the large number of parameters, we did not perform experiments for all combinations (it is possible to reach over 400 combinations when each parameter has 3 values to check).
It is important to note that we ran all the experiments on the GPU available on one of our personal computers. This fact added to the fact that it was not possible to check all combinations in relevant times.

Therefore, as a first step, we performed a general experiment after thinking and selecting parameters based on the results of individual runs we performed until this stage:

<img width="516" alt="image" src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/317141b1-d2f3-41e4-95a6-036bea9d1071">

After this general experiment, we continued to perform more focused experiments, after first narrowing down and fixing some parameters:

Based on the results of the first experiment, we decided to fix the learning rate to 0.005.
We decided to reduce the set of epoch options to [25, 50]. Our hypothesis was that 15 epochs are not enough for the model to learn well enough.

At this stage, we ran an experiment that focused on finding the optimal value for the regularization coefficient from the set of options [0.0, 0.0001, 0.0005].

<img width="653" alt="image" src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/f5a84b8d-7283-4f06-b302-5e032c0fbc2a">

The best results were obtained for experiments 1, 2 and 6. We will save the parameter values of these experiments and expand on the performance of the resulting models later.

At this stage, we decided to "put aside" experiment #6 and its parameters, because we got similar results, but more stable ones, for experiments 1 and 2, where the only difference is in the number of epochs. 

Therefore, we set the regularization coefficient to 0.

In the next stage, we decided to examine additional parameters: batch size and dropout rate. We examined the following values for these parameters:
batch size = [16, 32]  
dropout rate = [0.0, 0.2]

Also, since in the previous experiment we got similar results for 50 and 25 epochs (experiments 1 and 2 in the previous table), we decided to perform the next experiment for 25 epochs only.

<img width="654" alt="image" src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/c74b96fa-97a5-448b-b5d3-a4fc053c66b4">

First, we can see that applying dropout in this experiment at a value of 0.2 worked well (experiments 2 and 4). We were surprised to see that a batch size of 16 brought the best results in this experiment (#2). It should be noted that in previous experiments we performed (before the first general experiment), we did not identify that this batch size brings better results than batch size = 32.

We will save the parameter values of experiment #2 and expand on the performance of this model later.

## Part 8: Results and Evaluation
We will present the five best performing models we obtained according to the Test Accuracy results.

**Model 1 - Test Accuracy = 0.743:**

* Epochs: 50
* Batch size: 32
* Learning rate: 0.005
* Regularization lambda: 0.0005
* Dropout rate: 0
* Train Time: 13.2~ minutes
* Training Loss: 0.418
* Validation Loss: 0.571
* Validation Accuracy: 0.709
* Test Time: 8.56 seconds
* Test Loss: 25.7

<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/4358f4a7-a3ed-450b-a266-1bf48aa48c83.jpg" width="400" height="300">
<br>
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/c15b4e7a-be4b-4ca0-913c-dafd275dca2f.jpg" width="400" height="300">
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/4ef28a54-ae92-4040-8af5-815a426c9f37.jpg" width="400" height="300">
<br>
<br>
<br>
<br>

**Model 2 - Test Accuracy = 0.732:**

* Epochs: 25
* Batch size: 32
* Learning rate: 0.005
* Regularization lambda: 0
* Dropout rate: 0
* Train Time: 6.43~ minutes
* Training Loss: 0.565
* Validation Loss: 0.586
* Validation Accuracy: 0.699
* Test Time: 8.77 seconds
* Test Loss: 26.8

<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/3b20d992-e8be-47b6-b8d0-3822e85561ed.jpg" width="400" height="300">
<br>
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/3b932395-3d14-4b61-a74b-8eca6ba373a6.jpg" width="400" height="300">
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/3895246a-e00c-4d2e-8047-a75426e91a17.jpg" width="400" height="300">
<br>
<br>
<br>
<br>

**Model 3 - Test Accuracy = 0.739:**
* Epochs: 50
* Batch size: 32
* Learning rate: 0.005
* Regularization lambda: 0
* Dropout rate: 0
* Train Time: 13.1~ minutes
* Training Loss: 0.48
* Validation Loss: 0.577
* Validation Accuracy: 0.703
* Test Time: 8.61 seconds
* Test Loss: 26.1

<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/6d5cf68e-f971-4a8b-aaa8-5c519e687aff.jpg" width="400" height="300">
<br>
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/a24dee08-2b75-44cd-aae6-26c258a1bcdf.jpg" width="400" height="300">
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/15fb7746-c853-4304-8207-3cbb90f68e00.jpg" width="400" height="300">
<br>
<br>
<br>
<br>

**Model 4 - Test Accuracy = 0.749:**

* Epochs: 25
* Batch size: 16
* Learning rate: 0.005
* Regularization lambda: 0
* Dropout rate: 0.2
* Train Time: 8.18~ minutes
* Training Loss: 0.578
* Validation Loss: 0.585
* Validation Accuracy: 0.725
* Test Time: 8.84 seconds
* Test Loss: 25.1

<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/09a8be4c-c8e6-4a91-ae06-2a87ba5e98e8.jpg" width="400" height="300">
<br>
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/99c5664a-b28a-4fb6-a603-2310965e1cdc.jpg" width="400" height="300">
<img src="https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/8986e673-406f-487e-9997-b3758f859e88.jpg" width="400" height="300">
<br>
<br>
<br>
<br>

**In addition, we performed a comparison of training run times (not test since they are very short) between the models:**

![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/a04f13e7-de25-466b-81db-ae725b19201c)

As expected, we can see that models 1 and 3, which ran for 50 epochs, had a training time approximately twice that of models 2 and 4. It is also interesting to see that model 4, the last one, ran longer than model 2 - we speculate that the reason is related to the fact that the number of batches in each epoch is actually doubled (batch size in model 4 is 16, half of that in model 2). It is possible that the dropout rate also has an effect on the model's training time.

So far, we can summarize that model 4 has the best performance. Therefore, we will examine certain parameters of this model in depth in order to see their effects on model performance, in the hope of even improving the performance.

**Batch Size Effect:**

![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/2835996b-926b-46a7-97fc-ec043dc6e65f)

In this experiment, we fixed all the parameters except for the batch size. We hypothesized that too small a size would lead to poorer results because in such a state there would be frequent changes in the weights, and these updates would be based on learning from a relatively small amount of data. Indeed, when running the model with a batch size of 8, we obtained a good but not optimal result. In contrast, we were surprised to get an optimal result for batch size = 64, whereas in previous runs we performed as part of the assignment, we did not get good results for this size.

**Learning Rate Effect:**

![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/0bfd08df-1476-41a1-a283-631218ed7f7a)

In this experiment, we decided to examine 4 values for the learning rate, which are divided into 2 orders of magnitude. First, we can see that the learning rate in the order of magnitude of {10}^{-2} brought better results than in the order of magnitude of {10}^{-3}. We hypothesize that the reason for this is that a learning rate that is too small makes it difficult for the model to reach the optimum - the steps are too small in each learning step.

**Number of Epochs Effect:**

![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/0dbbd237-3b4f-43e7-8d5e-3f9ffedacee6)

In this experiment, we examined the model's accuracy as a function of the number of epochs in training the model. Our hypothesis was that the more we increase the number of epochs, the better results we will get, and indeed that's what happened. However, it appears that for a doubled amount of epochs (50 versus 25), the model's improvement was minimal. We assume this is the case because the model approached overfitting, and most likely if we had continued to 75 and 100 epochs, the results may have dropped significantly since the model would not have been able to generalize well.

**Selecting the Model:**

After examining parameters on model 4, we decided that it is not stable enough, as we obtained different results in additional runs of this model, so it is possible that we got a random result in the previous runs. 

In summary, we decided to choose model 1, which achieved a test accuracy of 0.743, despite not achieving the best result (compared to 0.749 of model 4). We chose this as our optimal model due to the stability the model exhibited in its accuracy level over many experiments we performed, and in addition, the difference in accuracy level between it and model 4 is not significant.
We performed an additional run for model 1, which we chose as the best model. We will present a confusion matrix for it to illustrate its results and performance:
**Model 1 Parameters:**
* 25 epochs
* Batch size 32
* Learning rate 0.005
* Regularization coefficient 0
* Dropout percentage 0

![image](https://github.com/ShaharBenIshay/Siamese-NN-Using-One-Shot-Learning/assets/93884611/bb7a277b-7f61-42a9-9b69-94a96efaeae3)


## Part 9: Conclusions

1. We faced a complex task, and consequently, the results we obtained were not satisfactory - none of our models managed to achieve an accuracy level above 0.75. While there may be numerous reasons for this, we surmise that the primary cause is the small amount of data available for training, which constitutes the main reason for the difficulty in performing the task.

2. Regarding the number of epochs, we recommend maintaining a quantity of 25 during training, unless the user has significantly more computational power available. If the user has such capability, increasing the number of epochs to 50 or even more could potentially yield better results. However, we observed that a significant increase in the number of epochs beyond 25 did not contribute substantially to improving the model's accuracy.

3. Regarding the batch size, we had some deliberation on the conclusion. Ultimately, we decided to recommend a size of 32 rather than 16, despite obtaining a slightly better result with a size of 16. The reason we chose this recommendation is that we wanted a more stable and less random model. Additionally, we wanted a certain balance between the batch size and the amount of data for training, since after each batch in the training process, the weights are updated, and we would want to perform a sufficient number of updates to reach the optimum, but not perform too many updates, thus "missing" the optimum.

4. During the course of our work, we observed that the learning rate could have a small value, but not too small, as the model needs to take relatively significant steps at the beginning of the learning process and progressively smaller steps over time. We managed to implement this approach using the scheduler, which decreased the learning rate every 5 epochs, allowing for a gradual approach as we neared the optimum.

5. We recommend keeping the regularization coefficient at 0 or a value very close to 0. We observed that a significant increase in this coefficient's value resulted in a "random" model, and we assume that for this model, the learning process is already highly complex, and therefore, adding "difficulty" to the training caused the model to fail completely.

6. We conclude that it is preferable not to use dropout for a similar reason as in conclusion 5. The model is already being trained on a complex task, so making the task even more difficult causes the model to over-generalize to the point where it is essentially not learning.

7. Due to time constraints, we did not perform significant preprocessing on the input images, such as cropping or rotation. It is possible that such preprocessing could have further improved the performance of our model. Our recommendation to ourselves, had we continued to investigate this topic and develop the model, would be to start with this task.
