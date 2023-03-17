"""Containing class for doing conformal prediction with weighted special conformity scores

"""

import numpy as np

class Conformal_nxg():
    def __init__(self, forget_factor, resid_factor, input_factor, num_input_vars):
        self.ff = forget_factor
        self.resid_factor = resid_factor
        self.input_factor = input_factor
        self.num_input_vars = num_input_vars
        self.cs = np.array([])
        self.weigths = np.array([])
    
    def calibrate(self, data, forecast, label):
        if len(forecast) > 1:
            forecast = np.squeeze(forecast)
            label = np.squeeze(label)
        else:
            forecast = forecast.flatten()
            label = label.flatten()

        if self.cs.size != 0:
            self.weigths = np.r_[np.power(self.ff, range(len(forecast)+len(self.weigths)-1,len(self.weigths)-1,-1)), self.weigths]
            input_vars = np.split(data, self.num_input_vars, axis = 1)
            variance = np.empty((data.shape[0],self.num_input_vars))
            for i in range(self.num_input_vars):
                variance[:,i] = np.var(input_vars[i],axis=1)
            self.cs = np.r_[self.cs,  (- variance @ self.input_factor.T + self.resid_factor*np.abs(label-forecast))]
        else:
            self.weigths = np.power(self.ff, range(len(forecast)-1,-1,-1))
            input_vars = np.split(data, self.num_input_vars, axis = 1)
            variance = np.empty((data.shape[0],self.num_input_vars))
            for i in range(self.num_input_vars):
                variance[:,i] = np.var(input_vars[i],axis=1)
            
            self.cs = - variance @ self.input_factor.T + self.resid_factor*np.abs(label-forecast)
            
        
        
        
    def predict(self,data,forecast, length_distr = 200, ymin = 0, ymax = 100):
        weights_cal = self.weigths / (np.sum(self.weigths) + 1)
        variance = np.var(np.split(data,self.num_input_vars),axis=1)
        num_i = np.min([len(weights_cal)//2,length_distr//2])
        pred = forecast
        for i in range(num_i - 1):
            confidence = (i+1)/num_i
            if(np.sum(weights_cal) >= confidence):
                ordR = np.argsort(self.cs)
                ind_thres = np.min(np.where(np.cumsum(weights_cal[ordR])>=confidence))             
                cal_thres = (np.sort(self.cs)[ind_thres] + variance @ self.input_factor.T)/self.resid_factor
                if cal_thres < 0:
                    raise Exception('Resulting conformity is negative, input_factor ' + str(self.input_factor) + ' is probabily too large.')
                pred = np.r_[forecast - cal_thres,pred, forecast + cal_thres]
            else:
                pred = np.r_[ymin,pred, ymax]
            
        if pred[0]< ymin:
            pred[pred < ymin] = ymin
            pred = np.r_[pred, ymax]
            pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        else:
            pred = np.r_[ymin, pred, ymax]
            if len(pred) != length_distr:
                pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        
        return np.squeeze(pred)