import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Load data
dataset = pd.read_csv("heart.csv")
X = dataset.drop(['output'], axis=1)
y = dataset['output']

# Define the target variable and features
target = 'output'
features = [col for col in dataset.columns if col != target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
