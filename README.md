# Objectives

The learning objectives of this project are to:
1. build a neural network for multilabel classification problem. Given a text, predict whether any of the following emotions are present:

admiration  
amusement  
gratitude
love
pride
relief
remorse

2. practice tuning model hyper-parameters

# CodaBench Competition

This project was a part of class level competition on Codabench.

The competition website is:

https://www.codabench.org/competitions/1557/?secret_key=5e170c5e-5ea3-4ddb-a3c2-401e50a7fc80

You can visit that site and read all the details of the competition, which
include the task definition, how your model will be evaluated, the format of
submissions to the competition, etc.


# Write your code

You should design a neural network model to perform the task described on the
CodaBench site.
Your code should train a model on the provided training data and tune
hyper-parameters on the provided development data.
Your code should be able to make predictions on either the development data
or the test data.
Your code should package up its predictions in a `submission.zip` file,
following the formatting instructions on CodaBench.

You must create and train your neural network in the Keras framework that we
have been using in the class.
You should train and tune your model using the training and development data
that you downloaded from the CodaBench site.


# Test your model predictions on the development set

During the development phase of the competition, the CodaBench site will expect
predictions on the development set.
To test the performance of your model, run your model on the development data,
format your model predictions as instructed on the CodaBench site, and upload
your model's predictions on the "My Submissions" tab of the CodaBench site.


# Submit your model predictions on the test set

When the test phase of the competition begins (consult the CodaBench site for
the exact timing), the instructor will update the CodaBench site to expect
predictions on the test set, rather than predictions on the development set.
The instructor will also release the unlabeled test set on CodaBench as
described above under "Download the Data".
To test the performance of your model, download the test data, run your model on
the test data, format your model predictions as instructed on the CodaBench
site, and upload your model's predictions on the "My Submissions" tab of the
CodaBench site.

During the test phase, you are allowed to upload predictions only once.
This is why it is critical to debug any formatting problems during the
development phase.
 
