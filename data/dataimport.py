import xarray as xr
import pandas as pd
import scipy as sc
import numpy as np
from data.datasplitter import DataSplitter

def dataimport():
    inputdirs = ['./data/sepensemble.nc','./data/oktensemble.nc', './data/novensemble.nc', './data/decensemble.nc', './data/janensemble.nc']
    forecastdirs = ['./data/sepforecast.nc','./data/oktforecast.nc', './data/novforecast.nc', './data/decforecast.nc', './data/janforecast.nc']
    measurementdir = ['./data/matdata.csv']

    datsplitter = DataSplitter(inputdirs,forecastdirs,measurementdir)

    splits= [pd.to_datetime('2022-09-15'), pd.to_datetime('2022-11-15 18:00:00'), pd.to_datetime('2023-01-23 18:00:00')]
    inputvars = ['x_wind_10m','y_wind_10m']
    ens_num = 30
    data, fcdata, msdata, fcdates = datsplitter.data_split(splits,inputvars,['wind_speed_10m'],['Vindhastighet'],24, ensemble_number=ens_num)


    for s in range(len(splits)-1):
        delS = np.array([],dtype=int)
        for p in range(data[s].shape[0]):
            if np.sum(np.isnan(data[s][p,:])) > 0.75*data[s].shape[1]:
                delS = np.append(delS,p)
                continue
            elif np.sum(np.isnan(fcdata[s][p,:])) > 0:
                delS = np.append(delS,p)
                continue
            elif np.sum(np.isnan(msdata[s][p,:])) > 0:
                delS = np.append(delS,p)
                continue
        
            elif np.sum(np.isnan(data[s][p,:])) > 0:
                for i in range(len(inputvars)):
                    data[s][p,i*ens_num:(i+1)*ens_num][np.argwhere(np.isnan(data[s][p,i*ens_num:(i+1)*ens_num]))] = np.random.choice(data[s][p,i*ens_num:(i+1)*ens_num][np.argwhere(~np.isnan(data[s][p,i*ens_num:(i+1)*ens_num]))].flatten(),(len(np.argwhere(np.isnan(data[s][p,i*ens_num:(i+1)*ens_num]))),1))
                #data[s][p,np.argwhere(np.isnan(data[s][p,:]))] = np.random.choice(data[s][p,~np.isnan(data[s][p,:])],(len(data[s][p,np.argwhere(np.isnan(data[s][p,:]))]),1))

    
        data[s] = np.delete(data[s], delS, axis=0)
        fcdata[s] = np.delete(fcdata[s], delS, axis=0)
        msdata[s] = np.delete(msdata[s], delS, axis=0)
        fcdates[s] = np.delete(fcdates[s], delS, axis=0)
        #fcdata[s] = np.squeeze(fcdata[s])
        #msdata[s] = np.squeeze(msdata[s])
    return data, fcdata, msdata, fcdates