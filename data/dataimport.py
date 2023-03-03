import xarray as xr
import pandas as pd
import scipy as sc
import numpy as np
from datasplitter import DataSplitter

def dataimport():
    inputdirs = ['../data/sepensemble.nc','../data/oktensemble.nc', '../data/novensemble.nc', '../data/decensemble.nc', '../data/janensemble.nc']
    forecastdirs = ['../data/sepforecast.nc','../data/oktforecast.nc', '../data/novforecast.nc', '../data/decforecast.nc', '../data/janforecast.nc']
    measurementdir = ['../data/matdata.csv']

    datsplitter = DataSplitter(inputdirs,forecastdirs,measurementdir)

    splits= [pd.to_datetime('2022-09-15'), pd.to_datetime('2022-11-30 18:00:00'), pd.to_datetime('2022-12-31 18:00:00'), pd.to_datetime('2023-01-23 18:00:00')]
    data, fcdata, msdata, fcdates = datsplitter.data_split(splits,['x_wind_10m','y_wind_10m'],['wind_speed_10m'],['Vindhastighet'],24)


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
                data[s][p,np.argwhere(np.isnan(data[s][p,:]))] = np.random.choice(data[s][p,~np.isnan(data[s][p,:])],len(data[s][p,np.argwhere(np.isnan(data[s][p,:]))]))

    
        data[s] = np.delete(data[s], delS, axis=0)
        fcdata[s] = np.delete(fcdata[s], delS, axis=0)
        msdata[s] = np.delete(msdata[s], delS, axis=0)
        fcdates[s] = np.delete(fcdates[s], delS, axis=0)
    return data, fcdata, msdata, fcdates