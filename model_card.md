# Model Card
This project intends to develop a classification model based on public information from Census Bureau data. Code will be written to monitor the model performance on various data slices. Finally, it will be deployed using the FastAPI package.

## Model Details
This model has been created using Random Forest Classifier on the dataset.

## Intended Use
This model was created for a class project and should be considered general use for exploratory purposes only.

## Training Data
The data for this model was acquired from a US Census Bureau dataset provided by the organization hosting this project. It contains fields such as: age, work class, education, marital status, occupation, relationship, race, sex, capital gain and loss, hours worked per week, native country, and salary.

## Evaluation Data
The evaluation was done on a subset of the data with a test size of 20%.

## Metrics
Precision: 0.7472
Recall: 0.6340 
F1: 0.6860

## Ethical Considerations
Because this is publicly available data from the Census Bureau, bias should be considered minimal. No other ethical considerations are noted.

## Caveats and Recommendations
This model was created as part of a class project and should not be used in any professional setting. This is an entry-level educational endeavor, therefore some errors may be present.