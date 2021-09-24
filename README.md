# Give And Take: Building and Optimizing Neural Network Models for Charity

## Purpose
The point of this project was to build, train and optimize neural network models using Python, Pandas and TensorFlow, among other things, in order to help a (fictional) charitable organization determine which projects were worthy of investment. It involved wrangling data, splitting it into testing and training sets, and fitting and tweaking
a predictive model to achieve better accuracy.

## Results
* The target variable for the model is a binary classification -- whether previous donations produced "successful" outcomes or not. The definition of "success" isn't given, but it's denoted by either a '1' or '0' in this dataset.
* The features include most of the other variables in the data, including application types, affiliations, use case, type of organization, income amounts, and how much money is being requested.
* For the purposes of modeling the data, identification numbers (EIN) and Organization name (NAME) were neither targets, nor features. There are probably scenarios we could imagine where the name of an organization might hint at something about its potential, and we might want to keep that data, especially if we delve into language processing and features. But for this project, that was not a consideration.

### Compiling, Training and Evaluating
* I began by choosing two hidden layers, with 8 and 5 neurons, respectively, as well as two activation functions -- relu for the hidden layers, and sigmoid for the output layer. There were roughly 43 features used as inputs, but I wanted to begin with a relatively small number of neurons/layers so I could gradually increase the complexity of the model if I needed.

![image](https://user-images.githubusercontent.com/1015285/134708283-f4633db7-3532-4b10-9b44-4c66486d6a98.png)

* The model's first pass achieved a little less than 73% accuracy on the testing data. That's respectable, but not quite up to the desired 75% level. 

![image](https://user-images.githubusercontent.com/1015285/134708503-4388a875-688a-40a8-9af5-9c80630a19df.png)
 
* In order to try and dial up the accuracy, I changed a few of the inputs -- loosening the paramaters on the variable application_type by binning uniques less than 156 instead of 528. I also doubled the number of neurons in each attempt (from 8 and 5 to 16 and 10; 16 and 10 to 32 and 20, etc.) and added a third hidden layer with 4 neurons. I then doubled that number for the third attempt at optimization, and changed the activation function in the second hidden layer from relu to 'tanh.'

* Unfortunately, the closest I got to achieving 75% accuracy on the test data was in my first model, before optimization began. All my attempts achieved a little less than 73%, but the one with the fewest neurons and hidden layers was slightly higher than the rest (72.73%).

## Recommendation

Because increasing complexity didn't seem to make the model more accurate (indeed, it decreased in accuracy), I would recommend trying a simple logistic regression first, to establish a baseline with a more interpretable model. If the accuracy isn't as high as desired, then maybe attempt a Random Forest Classifier, and afterward, perhaps a very simple neural network.


