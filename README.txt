##### ml_classification 1.0 ####

The script is meant to be executed on Python 3.8.
Please install all the packages indicated in the file requirements.txt before running.

####

This python script aims to apply different Machine Learning classification algorithms on a simple dataset. A prediction will be made whether a hypothetical loan case will be paid off or not. 

####

Historical dataset from previous loan applications are loaded from CSV files, the data is then cleaned, and different classification algorithm are applied on the data. The following algorithms are used to build the models:
- k-Nearest Neighbour
- Decision Tree
- Support Vector Machine
- Logistic Regression

The results about the accuracy of each classifier are then reported using the following metrics when these are applicable:
- Jaccard index
- F1-score
- LogLoass

####

© Copyright Nikita Volkov 2021