# Adversarial Regularization for Convolution Filters

## General info
This project is our approach of solving overfitting problem in case when data set is pretty small using Convolutional Neural Network (CNN).
The repository is built on top of PyTorch `1.10.0`


## Details
Our Model structure is following: it has 3 convolution layers of different size made by using `Conv2d` method.
General CNN model looks the following way:


<img src="https://github.com/Arius1404/Adversarial-Regularization-for-Convolution-Filters/blob/main/imgs/CNN.png" width="800"/>

In our case we slightly simplified CNN model by removing pooling since our goal does not require dimension reduction.
As you can see every convolution layer has different number of kernels and its' size. These values for our model are printed below (check the table).

#### Number of kernels and its size in every layer
| Layer ID      | Number of kernels | Kernel size | 
| ----------- | ----------- | ----------- |
| 1      | 192 (3 x 64)       |7 x 7 |
| 2   | 8192 (64 x 128)        |5 x 5 |
| 3   | 32768 (128 x 256)        |3 x 3 

## Getting started
We decided, that it would be more convenient for you to run this script via Google Colab, since it provide users with sufficient both GPU and CPU resources, and, moreover allows to download required datasets with only one line of code.
You have got the opportunity to see our masterpiece and try to run it without pulling it from [Github](https://github.com/Arius1404/Adversarial-Regularization-for-Convolution-Filters). It is as easy as a pie! Just follow these steps:
- You have to make several folders on your Google Drive. The path must be the following `/ML_project/by_channel_layer_n`, where n should be 0, 3, 6
- After that follow [this](https://colab.research.google.com/drive/1E_HRN-isNgKG10ujKAst31DpydhtjURv#scrollTo=cQz8Kb_iqdbI) link and run cells consequently
- Do not forget to give reqired permission for Colab to connect to your Google Drive.
- Enjoy and have fun!

### Datasets

During the project we used CIFAR-100 and CIFAR-10 datasets. This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.
More about [CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

## Results
As the output we would get kernels with evenly distributed heatmaps like this one


<img src="https://github.com/Arius1404/Adversarial-Regularization-for-Convolution-Filters/blob/main/imgs/3-0.png" width="800"/>
