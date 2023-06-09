# Preparing the data
- I am using the [Chinese MNIST](https://www.kaggle.com/datasets/gpreda/chinese-mnist) dataset from Kaggle.
- It contains 15 Chinese hand written number: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000000.
- Each number contains 1000 examples, with a resolution of 64x64 pixels.
- I have remove the `100`, `1000`, `10000` and `100000000` in `chinese_mnist.csv` for better presentation, and save it as `chinese_mnist_0-10.csv`.
- Then run the `prepareData.m` to generate the `final.mat`.
- It read in all the image files and use the `chinese_mnist_0-10.csv` to map the actual value.
- All images are then transformed into a row vecotr and append into the X as a row.
- The folder structure now should be like this.  
  ![this](folder-structure.png).
- The `final.m` stores two variables:
  - X is a (11x1000) x (64x64) matrix where 11 is Chinese numbers, 1000 is the examples of each numbers, 64x64 is the dimension of the image.
  - y is a (11x1000) x 1 column vector, which is the actual value of the images.

# Start the prediction
- I am using the nueral network for this assignment to predict the image value.
- 20 examples are extracted from the X for testing, and the rest are for trainning.
- I have one hidden layer for the network, I use the `fmincg` for the cost gradient calculation.
- Run `final.m`.
- It will try to calculate the optimal `Theta` from the **trainning** dataset and then use it to predict the **testing** dataset.
- It presents the cost history diagram, the accuracy of the tranning dataset and finally the result of the prediction and its accuracy.

# Final though
The accuracy doesn’t quite satify with my inital setup, then I try to play with the parameters `lambda`, `number of iterations` and `hidden layer units`, and the computation times, CPU resources and result starts to vary.
I have achieve a better prediction result by increasing the hidden layer units, the number of iterations and decreasing the lambda (just a litter bit).
Beside trying out different parameters, I think the next thing I can do is to work on the image preprocess step or try out other different network.