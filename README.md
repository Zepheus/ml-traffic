# Trafic Sign Recognition

This project will execute the training and prediction of traffic signs, based on the Kaggle competition [Here](https://inclass.kaggle.com/c/traffic-sign-recognition).

It uses both a convolutional neural network (**Lasagne**) and Logistic Regression (**scikit-learn**) in combination with feature extraction through **scikit-image**. The models were trained on an NVidia GTX 960 with 2GB of memory.

## Executable files:
The executable files are listed below with their respective functionality:

* haar_importances.py:
	- This file will calculate the importances of the different haar configurations.
	- All importances will be written to the file "haarImportance.txt" in the current directory.
	- The importances are sorted according their importance in descending order.
	- This means that the most important configuration will be on the first line of the file.
	- This file can later be used by the haar_feature.
* main.py:
	- This file will train the model on all the train given and predict the results of the test images given.
	- These results are all written to a file named 'result.csv' in the current directory.
	- The given train and test images are specified in the python file. Also the used model and features are given in the code file.
* meta_parameter_estimators.py
	- This file will test metaparameters of features based on the error_rate.
	- Which parameters, features are trained with which trainer is specified in the file itself.
