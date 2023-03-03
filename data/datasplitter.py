import xarray as xr
import pandas as pd
import scipy as sc
import numpy as np


class DataSplitter:
    def __init__(self,inputdirs, forecastdirs, measuredirs):
        self.inputdirs = inputdirs
        self.forecastdirs = forecastdirs
        self.measuredirs = measuredirs
    
        
    def data_split(self, splits, input_vars, forecast_vars, label_vars, time_horizon = 24, ensemble_number = 30, include_previous_forecasts = False, time_of_day = 12, frequency = 24 ):
        inputdata, fcdata = self.read_data()
        
        dateranges =[]
        forecastdates = []
        for i in np.arange(1,len(splits)):
            if splits[i-1].hour + 6 > time_of_day:
                dateranges.append(pd.date_range(splits[i-1] + pd.Timedelta(hours=time_of_day + (24-(splits[i-1].hour))) ,splits[i], freq = str(frequency)+'h'))
            else:
                dateranges.append(pd.date_range(splits[i-1] + pd.Timedelta(hours=np.max([6,time_of_day-splits[i-1].hour ])) ,splits[i], freq = str(frequency)+'h'))
            forecastdates.append(dateranges[-1] + pd.Timedelta(hours=time_horizon))
        
        measuredata = pd.read_csv(self.measuredirs[0], sep=';')
        measuredata['Datetime'] = pd.to_datetime(measuredata['Datum'] + ' ' + measuredata['Tid (UTC)'])
        measuredata = measuredata.set_index('Datetime')
        measuredata = measuredata.drop(['Datum','Tid (UTC)'], axis=1)
        
        time_hz = int(time_horizon/12 - 1)
        if include_previous_forecasts:
            num_prev_forecast = 2 - time_hz
        
        
        input_arrays = []
        ms_arrays = []
        fc_arrays = []
                
        for dr in dateranges: 
            if include_previous_forecasts:
                input_arr = np.empty((dr.size,len(input_vars)*ensemble_number*(num_prev_forecast+1)))                
            else:
                input_arr = np.empty((dr.size,len(input_vars)*ensemble_number))
            input_arr[:] = np.NaN
                
            fc_arr = np.empty((dr.size,len(forecast_vars)))
            fc_arr[:] = np.NaN
            ms_arr = np.empty((dr.size,len(label_vars)))
            ms_arr[:] = np.NaN
            i = 0
            for date in dr:
                for var in range(len(input_vars)):
                    if include_previous_forecasts:
                        for j in range(num_prev_forecast+1):
                            try:
                                input_arr[i,j * ensemble_number + var * (num_prev_forecast+1) * ensemble_number :(j+1) * ensemble_number + var * (num_prev_forecast+1) * ensemble_number] = inputdata[input_vars[var]].sel(forecast_reference_time = date - j*pd.Timedelta(hours=12), time = time_hz+j, ensemble_member = range(0,ensemble_number)).values
                            except:
                                pass
                    else:
                        try:
                            input_arr[i,var*ensemble_number:(var+1)*ensemble_number] = inputdata[input_vars[var]].sel(forecast_reference_time = date, time = time_hz, ensemble_member = range(0,ensemble_number)).values
                        except:
                            pass
                    
                for var in range(len(forecast_vars)):
                    try:
                        fc_arr[i,var] = fcdata[forecast_vars[var]].sel(forecast_reference_time = date, time = time_hz).values
                    except:
                        pass
                for var in range(len(label_vars)):
                    try:
                        ms_arr[i,var] = measuredata.loc[date + pd.Timedelta(hours=time_horizon),label_vars[var]]
                    except:
                        pass
                i += 1
                        
            input_arrays.append(input_arr)
                
            fc_arrays.append(fc_arr)
            ms_arrays.append(ms_arr)
            
            
        return input_arrays, fc_arrays, ms_arrays, forecastdates
            
        
    
    
    def read_data(self):
        inputdata = xr.open_dataset(self.inputdirs[0])
        for dir in np.arange(1,len(self.inputdirs)):
            temp_data = xr.open_dataset(self.inputdirs[dir])
            inputdata = xr.concat([inputdata,temp_data],'forecast_reference_time')
        inputdata = inputdata.squeeze()
        
        fcdata = xr.open_dataset(self.forecastdirs[0])
        for dir in np.arange(1,len(self.forecastdirs)):
            temp_data = xr.open_dataset(self.forecastdirs[dir])
            fcdata = xr.concat([fcdata,temp_data],'forecast_reference_time')
        fcdata=fcdata.squeeze()
            
        return inputdata, fcdata
        
        
        