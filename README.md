# Titanic Survival Prediction

This project implements a machine learning model to predict survival on the Titanic using the famous Titanic dataset. The model uses Random Forest Classifier to make predictions based on passenger information.

## Project Structure

- `main.py` - Entry point of the application that trains the model
- `train.py` - Contains the model training logic using RandomForestClassifier
- `clean_data.py` - Handles data preprocessing and cleaning
- `Titanic-Dataset.csv` - The dataset containing Titanic passenger information
- `requirements.txt` - List of Python dependencies

## Features Used

The model uses the following features from the dataset:
- Passenger Class (encoded as one-hot vectors)
- Sex (encoded as binary: 0 for female, 1 for male)
- Other relevant passenger information

Features dropped during training:
- Ticket
- Name
- PassengerId
- Embarked
- Cabin

## Data Preprocessing

The data preprocessing steps include:
1. One-hot encoding of Passenger Class
2. Binary encoding of Sex (male/female)
3. Removal of unnecessary columns
4. Handling of the dataset using pandas DataFrame

## Model

- Algorithm: Random Forest Classifier
- Number of estimators: 5000
- Training split: 80% training, 20% testing
- Accuracy: 85%+ on the test set

## Usage

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model:
```bash
python main.py
```

The program will output the model's accuracy score on the test set.

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib

## Note

The model uses a Random Forest Classifier with 5000 estimators to ensure robust predictions. The data cleaning process includes handling categorical variables through one-hot encoding and binary encoding for optimal model performance.