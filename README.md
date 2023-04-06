## Wind prediction using conformal

All data in the data directory are gathered under the Creative Commons 4.0 BY license. Ensemble and forecast data are provided by MET Norway while the measurements in ´matdata.csv´ are provided by SMHI.

Certain attributes and columns have been removed from the SMHI data, the measurements have not been altered with.

The data was gathered during the following dates:

1. 07-02-23
    - `oktensemble.nc`, `novensemble.nc`, `decensemble.nc`, `janensemble.nc`
2. 08-02-23
    - `oktforecast.nc`, `novforecast.nc`, `decforecast.nc`, `janforecast.nc`
3. 09-03-23
    - `matdata.csv`, `febensemble.nc`, `marensemble.nc`, `aprensemble.nc`, `mayforecast.nc`, `mayensemble.nc`, `junforecast.nc`, `junensemble.nc`, `julforecast.ne`, `julensemble.nc`, `augforecast.nc`, `augensemble.nc`, `sepforecast.nc`, `sepensemble.nc`
4. 10-03-23
    - `aprforecast.nc`, `marforecast.nc`, `febforecast.nc`, `jan22forecast.nc`, `jan22ensemble.nc`

The following libraries are required for executing the code
- numpy
- crepes
- quantile_forest
- pandas
- xarray
- matplotlib
