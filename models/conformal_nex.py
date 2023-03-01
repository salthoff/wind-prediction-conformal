"""Containing class for doing conformal prediction with weighted residuals

"""

import numpy as np

class Conformal_nex():
    def __init__(self, forget_factor):
        self.ff = forget_factor
        self.weigths = None
        self.residuals = None
    
    def calibrate(self,data, modeloutput, label):
        if self.weigths != None:
            self.weigths = np.r_[np.power(self.ff, range(len(modeloutput)+len(self.weigths),len(self.weigths)+1,-1)), self.weigths]
            self.residuals = np.r_[self.residuals,np.abs(label-modeloutput)]
        else:
            self.weigths = np.power(self.ff, range(len(modeloutput),0,-1))
            self.residuals = np.abs(label-modeloutput)

        
        
        
    def predict(self,data, modeloutput, confidence =0.95):
        weights_cal = self.weigths / (np.sum(self.weigths) + 1)
      
        if(np.sum(weights_cal) >= confidence):
            ordR = np.argsort(self.res)
            ind_thres = np.min(np.where(np.cumsum(weights_cal[ordR])>=confidence))             
            cal_thres = np.sort(self.res)[ind_thres]
        else:
            cal_thres = np.inf
        
        yPI = [modeloutput - cal_thres, modeloutput + cal_thres]
        return yPI