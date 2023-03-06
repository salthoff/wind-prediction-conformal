""" Containing class for doing random forest quantile regression

"""

import numpy as np
from quantile_forest import RandomForestQuantileRegressor as qrf

class RanForestQuantile():
    def __init__(self, num_trees = 100, window_length = np.inf):
        self.window_length = window_length
        self.system = None
        self.input = None
        self.label = None
        self.system = qrf(n_estimators=num_trees)

    def calibrate(self, data, forecast, label):
        label = np.squeeze(label)
        if self.label != None:
            self.data = np.r_[self.data, data]
            self.label = np.r_[self.label, label]
        else:
            self.data = data
            self.label = label

        if len(self.label) > self.window_length:
            self.system.fit(X=self.data[-self.window_length:,:],y=self.label[-self.window_length:])
        else:
            self.system.fit(X=self.data, y=self.label)

    def predict(self, data, forecast, confidence = 0.95):
        pred = self.system.predict(X = data.reshape(1, -1), quantiles = [(1-confidence)/2, 1-(1-confidence)/2])
        return pred

