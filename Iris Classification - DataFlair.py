# DataFlair Iris Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] # As per the iris dataset information

# Load the data
df = pd.read_csv('E:/Study/GIKI BAI Course Material/Fourth Semester BAI/CS205 L/Lab 7 and 8/iris.data', names=columns)

df.head()

# Some basic statistical analysis about the data
df.describe()

# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')

# Seperate features and target  
data = df.values
X = data[:,0:4]
Y = data[:,4]

# Calculate avarage of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25

# Plot the avarage
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()

# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

# Add random noise to the input data
np.random.seed(42)  # for reproducibility
noise_factor = 0.5  # adjust the noise level as needed

X_noisy = X + np.random.normal(scale=noise_factor, size=X.shape)

# Split the noisy data to train and test dataset
X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(X_noisy, Y, test_size=0.2)

# Support vector machine algorithm with noisy data
svn_noisy = SVC()
svn_noisy.fit(X_train_noisy, y_train_noisy)

# Predict from the test dataset with noisy data
predictions_noisy = svn_noisy.predict(X_test_noisy)

from sklearn.metrics import accuracy_score
# Calculate the accuracy with noisy data
accuracy_noisy = accuracy_score(y_test_noisy, predictions_noisy)
print("Accuracy with noisy data:", accuracy_noisy)



# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

#model.predict(X_new)