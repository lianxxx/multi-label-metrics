from mlmetrics import mlmetrics
import numpy as np

y_true = np.array([[1,0,1,1],[0,1,0,0],[0,0,0,0],[0,1,1,0]]) # true value
y_pred = np.array([[0,0,1,1],[1,1,0,1],[0,1,0,0],[1,1,1,1]]) # predicted value

# creating the object and passing the true and predicted
metric = mlmetrics(y_true, y_pred)

# accuracy
print(metric.accuracy())
