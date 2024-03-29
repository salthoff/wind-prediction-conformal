""" Containing class for doing random forest quantile regression

"""

import numpy as np
from quantile_forest import RandomForestQuantileRegressor as qrf

class RanForestQuantile():
    """
    Class for using the Quantile regression forest

    Attributes
    ---------
    window length: int
        Length of data to be used by the system
    input:
        Saved input data
    label:
        Saved labels associated with the data
    system:
        The QRF object
    
    Methods
    -------
    calibrate(data, forecast, label)
        Calibrates the system
    predict(data, forecast, length_distr = 200, ymin = 0, ymax = 100)
        Predicts the CDF from data and forecast
    """
    def __init__(self, num_trees = 100, window_length = np.inf):
        self.window_length = window_length
        self.input = None
        self.label = np.array([])
        self.system = qrf(n_estimators=num_trees)

    def calibrate(self, data, forecast, label):
        label = np.squeeze(label)
        if self.label.size != 0:
            self.data = np.r_[self.data, data]
            self.label = np.r_[self.label, label]
        else:
            self.data = data
            self.label = label

        if len(self.label) > self.window_length:
            self.system.fit(X=self.data[-self.window_length:,:],y=self.label[-self.window_length:])
        else:
            self.system.fit(X=self.data, y=self.label)

    def predict(self, data, forecast, length_distr = 200, ymin = 0, ymax = 100):
        pred = np.squeeze(self.system.predict(X = data.reshape(1, -1), quantiles = np.linspace(1/length_distr, 1-1/length_distr, num=length_distr-2).tolist()))
        if pred[0]< ymin:
            pred[pred < ymin] = ymin
            pred = np.r_[pred, ymax]
            pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        else:
            pred = np.r_[ymin, pred, ymax]
        return np.squeeze(pred)

