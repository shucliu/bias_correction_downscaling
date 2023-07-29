#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
description     Auxiliary functions for Bias-corrected downscaling
authors		Before 2022: original developments by Roman Brogli
                Since 2022:  upgrade to PGW for ERA by Christoph Heim
                Since 2023:  modify the code for bias-corrected downscaling bu Shuchang Liu
"""
##############################################################################
import os, math
import xarray as xr
import numpy as np
from numba import njit
from datetime import datetime,timedelta
from constants import CON_RD,CON_G,CON_MW_MD
from settings import (
    i_debug,
    i_use_xesmf_regridding,
    file_name_bases,
    TIME_GCM, LEV_GCM, HLEV_GCM, LON_GCM, LAT_GCM,
    TIME_DELTA, PLEV_DELTA, LON_DELTA, LAT_DELTA,
)
if i_use_xesmf_regridding:
    import xesmf as xe

## TODO DEBUG
#from matplotlib import pyplot as plt
##############################################################################


##############################################################################
##### ARBITRARY FUNCTIONS
##############################################################################
def dt64_to_dt(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    source: 
      https://gist.github.com/blaylockbk/1677b446bc741ee2db3e943ab7e4cabd
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)


##############################################################################
##### PHYSICAL COMPUTATIONS
##############################################################################
def specific_humidity_to_vapor_pressure(hus, pa):
    """
    Compute vapor pressure
    https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    """
    vapp = hus * pa / (CON_MW_MD + 0.378 * hus)
    return(vapp)

def vapor_pressure_to_specific_humidity(vapp, pa):
    """
    Compute vapor pressure
    https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    """
    hus = CON_MW_MD * vapp / (pa - (1 - CON_MW_MD) * vapp)
    return(hus)

def saturation_vapor_pressure_water_or_ice(pa, ta, water=True):
    """
    Compute saturation vapor pressure over water or ice
    according to ECMWF IFS documentation (7.93).
    """
    T0 = 273.16     # K
    if water:
        a1 = 611.21 # Pa
        a3 = 17.502 # 1
        a4 = 32.19  # K
    else: # over ice
        a1 = 611.21 # Pa
        a3 = 22.587 # 1
        a4 = -0.7   # K
    vapp_sat_x = a1 * np.exp(a3 * (ta - T0) / (ta - a4))
    return(vapp_sat_x)

def saturation_vapor_pressure_water_and_ice(pa, ta):
    """
    Compute saturation vapor pressure according to ECMWF IFS documentation (7.92).
    """
    T0 = 273.16     # K
    Ti = 250.16     # K
    alpha = xr.full_like(ta, np.nan)
    alpha = xr.where(ta >= T0, 1, alpha)
    alpha = xr.where(ta <= Ti, 0, alpha)
    alpha = xr.where((ta < T0) & (ta > Ti), np.power((ta - Ti)/(T0 - Ti), 2.), alpha)
    vapp_sat = (
        alpha * saturation_vapor_pressure_water_or_ice(pa, ta, water=True) +
        (1 - alpha) * saturation_vapor_pressure_water_or_ice(pa, ta, water=False)
    )
    return(vapp_sat)

def specific_to_relative_humidity(hus, pa, ta):
    """
    Compute relative humidity from specific humidity
    following the implementation in the IFS model.
    """
    hur = (
        specific_humidity_to_vapor_pressure(hus, pa) /
        saturation_vapor_pressure_water_and_ice(pa, ta)
    ) * 100
    return(hur)

def relative_to_specific_humidity(hur, pa, ta):
    """
    Compute relative humidity from specific humidity
    following the implementation in the IFS model.
    """
    vapp = hur / 100 * saturation_vapor_pressure_water_and_ice(pa, ta)
    hus = vapor_pressure_to_specific_humidity(vapp, pa)
    return(hus)

def integ_geopot(pa_hl, zgs, ta, hus, level1, p_ref):
    """
    Integrate GCM geopotential from surfce to a reference pressure
    level p_ref.
    """
    ## take log half-level pressure difference (located at full levels)
    # make sure pressure is not exactly zero because of ln
    pa_hl = pa_hl.where(pa_hl > 0, 0.0001)
    dlnpa = np.log(pa_hl).diff(
                dim=HLEV_GCM, 
                label='lower').rename({HLEV_GCM:LEV_GCM})

    # create geopotential array and fill with surface geopotential
    phi_hl = zgs.expand_dims(dim={HLEV_GCM:level1}).copy()

    # compute virtual temperature
    tav = ta * (1 + 0.61 * hus)

    ## integrate over model half levels
    for l in sorted(tav[LEV_GCM].values, reverse=True):
        # geopotential at full level
        phi_hl.loc[{HLEV_GCM:l}] = (
                phi_hl.sel({HLEV_GCM:l+1}) +
                (CON_RD * tav.sel({LEV_GCM:l}) * dlnpa.sel({LEV_GCM:l}))
        )

            
    phi_hl = phi_hl.transpose(TIME_GCM, HLEV_GCM, LAT_GCM, LON_GCM)

    ## integrate from last half level below reference pressure
    ## up to reference pressure
    # determine level below reference pressure
    p_diff = pa_hl - p_ref
    p_diff = p_diff.where(p_diff >= 0, np.nan)
    try:
        ind_ref_star = p_diff.argmin(dim=HLEV_GCM)
    except ValueError :
        raise ValueError("p_ref locally lies below the surface. Please set a lower reference pressue (p_ref_inp) in settings.py")
    hl_ref_star = p_diff[HLEV_GCM].isel({HLEV_GCM:ind_ref_star})
    # get pressure and geopotential of that level
    p_ref_star = pa_hl.sel({HLEV_GCM:hl_ref_star})
    phi_ref_star = phi_hl.sel({HLEV_GCM:hl_ref_star})

    # finally interpolate geopotential to reference
    # pressure level
    phi_ref = (
            phi_ref_star -
            (CON_RD * tav.sel({LEV_GCM:hl_ref_star-1})) * 
            (np.log(p_ref) - np.log(p_ref_star))
    )

    # remove multi-dimensional coordinates
    if HLEV_GCM in phi_ref.coords:
        del phi_ref[HLEV_GCM]
    if LEV_GCM in phi_ref.coords:
        del phi_ref[LEV_GCM]
    if PLEV_DELTA in phi_ref.coords:
        del phi_ref[LEV_GCM]

    return(phi_ref)


##############################################################################
##### CLIMATE DELTA COMPUTATION AND INTERPOLATION
##############################################################################
def load_delta(delta_input_dir, var_name, gcm_date_time, 
               target_date_time=None,
               name_base=file_name_bases['SCEN-CTRL']):
    """
    Load a climatology bias and if target_date_time is given,
    interpolate it to that date and time of the year.
    """
    ## full climatology bias (either daily or monthly)
    full_delta = xr.open_dataset(os.path.join(delta_input_dir,
                            name_base.format(var_name)))

    ## remove leap year february 29th if in delta
    leap_day = None
    for tind,dt64 in enumerate(full_delta.time.values):
        dt = dt64_to_dt(dt64)
        if (dt.month == 2) and (dt.day == 29):
            leap_day = dt64
    if leap_day is not None:
        full_delta = full_delta.drop_sel(time=leap_day)

    ## if climatology bias should be interpolated to a specific time
    if target_date_time is not None:
        # replace delta year values with year of current target_date_time
        for i in range(len(full_delta.time)):
            full_delta.time.values[i] = dt64_to_dt(
                        full_delta.time[i]).replace(
                                year=target_date_time.year)

        # find time index of climatology bias before target time
        # (and implement periodicity if necessary)
        is_before = (full_delta.time.values <= 
                    np.datetime64(target_date_time))

        # target time is within year 
        # --> periodicity not necessary
        if np.sum(is_before) > 0:
            ind_before = np.argwhere(is_before)[-1].squeeze()
            before = full_delta.isel(time=ind_before)

        # target time is before the first delta time step 
        # --> periodicity necessary
        else:
            ind_before = -1
            before = full_delta.isel(time=ind_before)
            before.time.values = dt64_to_dt(
                        before.time).replace(
                                year=target_date_time.year-1)

        # find time index of climatology bias after target time
        # (and implement periodicity if necessary)
        is_after =(full_delta.time.values >= 
                    np.datetime64(target_date_time)) 

        # target time is within year 
        # --> periodicity not necessary
        if np.sum(is_after) > 0:
            ind_after = np.argwhere(is_after)[0].squeeze()
            after = full_delta.isel(time=ind_after)

        # target time is after the last delta time step 
        # --> periodicity necessary
        else:
            ind_after = 0
            after = full_delta.isel(time=ind_after)
            after.time.values = dt64_to_dt(
                        after.time).replace(
                                year=target_date_time.year+1)

        # if target time is exactly contained in climatology bias
        # just take that value (from "before" which is arbitrary)
        if ind_before == ind_after:
            delta = before[var_name].expand_dims(dim='time', axis=0)

        # if interpolation is necessary, concate "before" and 
        # "after" and interpolate 
        else:
            full_delta = xr.concat([before, after], dim='time')
            # interpolate in time and select variable
            delta = full_delta[var_name].interp(time=target_date_time, 
                                        method='linear', 
                                    ).expand_dims(dim='time', axis=0)

        # make sure time is in the same format as in GCM file
        # GCM has "seconds since xyz" while delta has np.datetime64
        delta['time'] = gcm_date_time

    ## if full climatology bias should be returned without 
    ## time interpolation
    else:
        delta = full_delta[var_name]

    return(delta)


### TODO DEBUG START
#def load_delta_old(delta_inp_path, var_name, gcm_date_time, 
#               delta_date_time=None, name_base='{}_delta.nc'):
#    """
#    Load a climatology bias and if delta_date_time is given,
#    interpolate it to that date and time of the year.
#    """
#
#    def hour_of_year(dt): 
#        beginning_of_year = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
#        return(int((dt - beginning_of_year).total_seconds() // 3600))
#    name_base = name_base.split('.nc')[0] + '_{:05d}.nc'
#    diff_time_step = int(hour_of_year(delta_date_time)/3)
#    delta = xr.open_dataset(os.path.join(delta_inp_path,
#            name_base.format(var_name, diff_time_step)))[var_name]
#    # make sure time is in the same format as in laf file
#    delta['time'] = gcm_date_time
#
#    return(delta)
### TODO DEBUG STOP

def load_delta_interp(delta_input_dir, var_name, target_P,
                    gcm_date_time, target_date_time,
                    ignore_top_pressure_error=False):
    """
    Does the following:
        - load a climatology bias
        - for specific variables (ta and hur) also load surface
          climatology bias,
          as well as CTRL surface pressure. This is to extend
          the 3D climatology biass with surface values which makes
          the interpolation to the GCM model levels more precise.
        - vertically interpolate climatology biass to GCM model levels
    """
    delta = load_delta(delta_input_dir, var_name, 
                        gcm_date_time, target_date_time)

    ## for specific variables also load climatology bias for surface
    ## values and the historical surface pressure.
    if var_name in ['ta','hur']:
    #if var_name in []:
        sfc_var_name = var_name + 's'
        delta_sfc = load_delta(delta_input_dir, sfc_var_name, 
                            gcm_date_time, target_date_time)
        ps_hist = load_delta(delta_input_dir, 'ps', 
                            gcm_date_time, target_date_time,
                            name_base=file_name_bases['CTRL'])
    else:
        delta_sfc = None
        ps_hist = None

    # interpolate climatology bias onto GCM model levels
    delta = vert_interp_delta(delta, target_P, delta_sfc, ps_hist,
                            ignore_top_pressure_error)
    return(delta)


def replace_delta_sfc(source_P, ps_hist, delta, delta_sfc):
    """
    In the 3D climatology biass, replace the value just below
    the surface by the surface climatology bias value and insert
    it at CTRL surface pressure. This improves the precision
    of the climatology biass during interpolation to the GCM model levels.
    All 3D climatology bias values below the historical surface pressure
    are set to the surface value (constant extrapolation). This is
    because within the orography the DELTA climatology bias is assumed
    to be incorrect.
    """
    out_source_P = source_P.copy()
    out_delta = delta.copy()
    if ps_hist > np.max(source_P):
        sfc_ind = len(source_P) - 1
        out_source_P[sfc_ind] = ps_hist
        out_delta[sfc_ind] = delta_sfc
    elif ps_hist < np.min(source_P):
        raise ValueError()
    else:
        sfc_ind = np.max(np.argwhere(ps_hist > source_P))
        out_delta[sfc_ind:] = delta_sfc
        out_source_P[sfc_ind] = ps_hist
    return(out_source_P, out_delta)


def vert_interp_delta(delta, target_P, delta_sfc=None, ps_hist=None,
                       ignore_top_pressure_error=False):
    """
    Vertically interpolate climatology bias onto GCM model levels.
    If delta_sfc and ps_hist are given, surface values will
    be inserted into the 3D climatology bias at the height of
    the surface pressure. This gives a more precise interpolation.
    Climate delta values below the surface are set to the surface
    climatology bias because below the surface, the DELTA climatology bias
    is considered unreliable and thus constant extrapolation
    seems more reasonable.
    """

    # sort delta dataset from top to bottom (pressure ascending)
    delta = delta.reindex(
                {PLEV_DELTA:list(reversed(delta[PLEV_DELTA]))})

    # create 4D source pressure with DELTA pressure levels
    source_P = delta[PLEV_DELTA].expand_dims(
                    dim={LON_DELTA:delta[LON_DELTA],
                         LAT_DELTA:delta[LAT_DELTA],
                         TIME_DELTA:delta[TIME_DELTA]}).transpose(
                                TIME_DELTA, PLEV_DELTA, LAT_DELTA, LON_DELTA)

    ## if surface values are given, replace them at the
    ## level of the surface pressure
    if delta_sfc is not None:
        source_P, delta = xr.apply_ufunc(
                replace_delta_sfc, source_P, 
                ps_hist, 
                delta, delta_sfc,
                input_core_dims=[[PLEV_DELTA],[],[PLEV_DELTA],[]],
                output_core_dims=[[PLEV_DELTA],[PLEV_DELTA]],
                vectorize=True)
        source_P = source_P.transpose(TIME_DELTA, PLEV_DELTA, LAT_DELTA, LON_DELTA)
        delta = delta.transpose(TIME_DELTA, PLEV_DELTA, LAT_DELTA, LON_DELTA)

    # make sure all arrays contain the required dimensions
    if source_P.dims != (TIME_DELTA, PLEV_DELTA, LAT_DELTA, LON_DELTA):
        raise ValueError()
    if delta.dims != (TIME_DELTA, PLEV_DELTA, LAT_DELTA, LON_DELTA):
        raise ValueError()
    if target_P.dims != (TIME_GCM, LEV_GCM, LAT_GCM, LON_GCM):
        raise ValueError()

    # make sure there is no extrapolation at the model top
    # unless these levels are anyways not important for the user
    # and she/he manually sets ignore_top_pressure_error=True
    if np.min(target_P) < np.min(source_P):
        if not ignore_top_pressure_error:
            raise ValueError('GCM top pressure is lower than '+
                             'climatology bias top pressure. If you are ' +
                             'certain that you do not need the data ' +
                             'beyond to upper-most pressure level of the ' +
                             'climatology bias, you can set the flag ' +
                             '--ignore_top_pressure_error and re-run the ' +
                             'script.')
                             

    # run interpolation
    delta_interp = interp_logp_4d(delta, source_P, target_P,
                        extrapolate='constant')
    return(delta_interp)


def interp_logp_4d(var, source_P, targ_P, extrapolate='off',
                   time_key=None, lat_key=None, lon_key=None):
    """
    Interpolate 3D array in vertical (pressure) dimension using the
    logarithm of pressure.
    extrapolate:
        - off: no extrapolation
        - linear: linear extrapolation
        - constant: constant extrapolation
        - nan: set to nan
    """
    if extrapolate not in ['off', 'linear', 'constant', 'nan']:
        raise ValueError('Invalid input value for "extrapolate"')

    # set default values (GCM values) for dimension keys
    if time_key is None: time_key = TIME_GCM
    if lat_key is None: lat_key = LAT_GCM
    if lon_key is None: lon_key = LON_GCM

    #print(var.shape)
    #print(source_P.shape)
    #print(targ_P.shape)


    targ = xr.zeros_like(targ_P)
    tmp = np.zeros_like(targ.values)
    interp_1d_for_timelatlon(var.values,
                np.log(source_P).values,
                np.log(targ_P).values, 
                tmp,
                len(targ_P[time_key]), len(targ_P[lat_key]),    
                len(targ_P[lon_key]),
                extrapolate)
    targ.values = tmp
    return(targ)

@njit()
def interp_1d_for_timelatlon(orig_array, src_p, targ_p, interp_array,
                        ntime, nlat, nlon, extrapolate):
    """
    Vertical interpolation helper function with numba njit for 
    fast performance.
    Loop over time lat and lon dimensions and interpolate each column
    individually
    extrapolate:
        - off: no extrapolation
        - linear: linear extrapolation
        - constant: constant extrapolation
        - nan: set to nan
    """
    for time_ind in range(ntime):
        for lat_ind in range(nlat):
            for lon_ind in range(nlon):
                src_val_col = orig_array[time_ind, :, lat_ind, lon_ind]
                src_p_col = src_p[ time_ind, :, lat_ind, lon_ind]
                targ_p_col = targ_p[time_ind, :, lat_ind, lon_ind]
                
                if src_p_col[-1] < src_p_col[0]:
                    raise ValueError('Source pressure values must be ascending!')
                if targ_p_col[-1] < targ_p_col[0]:
                    raise ValueError('Target pressure values must be ascending!') 

                # call 1D interpolation function for current column
                interp_col = interp_extrap_1d(src_p_col, src_val_col, 
                                            targ_p_col, extrapolate)
                interp_array[time_ind, :, lat_ind, lon_ind] = interp_col


@njit()
def interp_extrap_1d(src_x, src_y, targ_x, extrapolate):
    """
    Numba helper function for interpolation of 1d vertical column.
    Does constant extrapolation which is used for the climatology biass.
    extrapolate:
        - off: no extrapolation
        - linear: linear extrapolation
        - constant: constant extrapolation
        - nan: set to nan
    """
    targ_y = np.zeros(len(targ_x))
    for ti in range(len(targ_x)):
        i1 = -1
        i2 = -1
        require_extrap = False
        for si in range(len(src_x)):
            ty = np.nan
            # extrapolate lower end
            if (si == 0) and src_x[si] > targ_x[ti]:
                if extrapolate == 'linear':
                    i1 = si
                    i2 = si + 1
                elif extrapolate == 'constant':
                    i1 = si
                    i2 = si
                require_extrap = True
                break
            # exact match
            elif src_x[si] == targ_x[ti]:
                i1 = si
                i2 = si
                break
            # upper src_x found (interpolation)
            elif src_x[si] > targ_x[ti]:
                i1 = si - 1
                i2 = si
                break
            # we are still smaller than targ_x[ti] 
            else:
                pass

        # extrapolate upper end
        if i1 == -1:
            if extrapolate == 'linear':
                i1 = len(src_x) - 2
                i2 = len(src_x) - 1
            elif extrapolate == 'constant':
                i1 = len(src_x) - 1 
                i2 = len(src_x) - 1
            require_extrap = True

        # raise value if extrapolation is required but not enabled.
        if require_extrap and extrapolate == 'off':
            raise ValueError('Extrapolation deactivated but data '+
                             'out of bounds.')

        # interpolate/extrapolate values
        if require_extrap and extrapolate == 'nan':
            targ_y[ti] = np.nan
        else:
            if i1 == i2:
                targ_y[ti] = src_y[i1]
            else:
                targ_y[ti] = (
                    src_y[i1] + (targ_x[ti] - src_x[i1]) * 
                    (src_y[i2] - src_y[i1]) / (src_x[i2] - src_x[i1])
                )

    return(targ_y)


def determine_p_ref(p_min_gcm, p_min_corrected_gcm, p_ref_opts, p_ref_last=None):
    """
    Find lowest DELTA pressure level among p_ref_opts that lies 
    above the minimum pressure (e.g. currently set to 90% surface pressure)
    in both GCM and bias-corrected GCM climate.
    Also ensure that during the iterations, no reference pressure level 
    at lower altitude than during last iterations is used. This is to
    prevent the iteration algorithm to oscillate between two reference
    pressure levels and not converge.
    """
    for p in p_ref_opts:
        if (p_min_gcm > p) & (p_min_corrected_gcm > p):
            if p_ref_last is None:
                return(p)
            else:
                return(min(p, p_ref_last))




##############################################################################
##### SMOOTHING OF ANNUAL CYCLE FOR DAILY CLIMATE DELTAS
##############################################################################
def filter_data(annualcyclgcmw, variablename_to_smooth, outputpath):

	"""
	This function performs a temporal smoothing of an annual timeseries 
    (typically daily resolution) using a spectral filter 
    (Bosshard et al. 2011).

	Input:
		Input 1: Path to a netcdf file of the annual cycle to be smoothed. 
        Normally this is the change in a specific variable between 
        two simulations (e.g. warming). 
        Can be 4 or 3 dimensional, where the time is one dimension 
        and the others are space dimensions.
		Input 2: The name of the variable within the given netcdf file
		Input 3: Path to the output file
		
	Output:
		A netcdf file containing the smoothed annual cycle. 
	"""	

	Diff = xr.open_dataset(annualcyclgcmw
                )[variablename_to_smooth].squeeze()
	coords = Diff.coords

	print('Dimension that is assumed to be time dimension is called: ', 
            Diff.dims[0])
	print('shape of data: ', Diff.shape)

	Diff = Diff.data

	#create an array to store the smoothed timeseries
	#Diff_smooth=np.zeros_like(Diff, dtype=np.float32) 

	if len(Diff.shape) == 4:
		times = Diff.shape[0] 
		levels = Diff.shape[1]
		ygrids = Diff.shape[2]
		xgrids = Diff.shape[3]
	elif len(Diff.shape) == 3:
		times = Diff.shape[0]
		ygrids = Diff.shape[1]
		xgrids = Diff.shape[2]
		levels = 0
	else:
		sys.exit('Wrog dimensions of input file should be 3 or 4-D')


	if len(Diff.shape) == 4:
        #loop over levels to smooth the timeseries on every level
		for i in range(levels):
			for yy in range(ygrids):
				for xx in range(xgrids):	
                    #reconstruct the smoothed timeseries using function below
					Diff[:,i,yy,xx] = harmonic_ac_analysis(Diff[:,i,yy,xx]) 



	if len(Diff.shape) == 3:		
		for yy in range(ygrids):
			for xx in range(xgrids):	
            #dump the smoothed timeseries in the array on the original level
				Diff[:,yy,xx] = harmonic_ac_analysis(Diff[:,yy,xx]) 
			

	print('Done with smoothing')

	#del Diff

	Diff = xr.DataArray(Diff, coords=coords, name=variablename_to_smooth)
	Diff.to_netcdf(outputpath, mode='w')


def harmonic_ac_analysis(ts):
    """
    Estimation of the harmonics according to formula 12.19 -
    12.23 on p. 264 in Storch & Zwiers

    Is incomplete since it is only for use in surrogate smoothing 
    --> only the part of the formulas that is needed there

    Arguments:
        ts: a 1-d numpy array of a timeseries

    Returns:
        hcts: a reconstructed smoothed timeseries 
                (the more modes are summed the less smoothing)
        mean: the mean of the timeseries (needed for reconstruction)
    """

    if np.any(np.isnan(ts) == True): #if there are nans, return nans
        smooths = np.full_like(ts, np.nan) #sys.exit('There are nan values')
        return smooths
    else:
        #calculate the mean of the timeseries (used for reconstruction)
        mean = ts.mean() 

        lt = len(ts) #how long is the timeseries?
        P = lt

        #initialize the output array. 
        #we will use at max 4 modes for reconstruction 
        #(for peformance reasons, it can be increased)
        hcts = np.zeros((4,lt))

        timevector=np.arange(1,lt+1,1)	#timesteps used in calculation	

        #a measure that is to check that the performed calculation 
        # is justified.
        q = math.floor(P/2.) 

        #create the reconstruction timeseries, mode by mode 
        #(starting at 1 until 5, if one wants more smoothing 
        #this number can be increased.)
        for i in range(1,4): 
            if i < q: #only if this is true the calculation is valid
            
                #these are the formulas from Storch & Zwiers
                bracket = 2.*math.pi*i/P*timevector
                a = 2./lt*(ts.dot(np.cos(bracket))) 
                #dot product (Skalarprodukt) for scalar number output!
                b = 2./lt*(ts.dot(np.sin(bracket))) 
                
                #calculate the reconstruction time series
                hcts[i-1,:] = a * np.cos(bracket) + b * np.sin(bracket) 
            
            else: #abort if the above condition is not fulfilled. In this case more programming is needed.
                sys.exit('Whooops that should not be the case for a yearly '+
                'timeseries! i (reconstruction grade) is larger than '+
                'the number of timeseries elements / 2.')

        smooths = sum(hcts[0:3,:]) + mean
        return smooths




##############################################################################
##### BILINEAR REGRIDDING
##############################################################################
def regrid_lat_lon(ds_delta, ds_gcm, var_name,
                    method='bilinear', i_use_xesmf=0):
    """
    Method to do lat/lon bilinear interpolation for periodic or non-periodic
    grid either with xesmf (i_use_xesmf), or with an xarray-only 
    implementation if the xesmf package is not installed.
    """
    if method != 'bilinear':
        NotImplementedError()

    targ_lon = ds_gcm[LON_GCM]
    targ_lat = ds_gcm[LAT_GCM]

    ## determine if DELTA data set is periodic
    dlon_delta = np.median(np.diff(ds_delta[LON_DELTA].values))
    dlat_delta = np.median(np.diff(ds_delta[LAT_DELTA].values))
    if (dlon_delta + np.max(ds_delta[LON_DELTA].values) - 
                  np.min(ds_delta[LON_DELTA].values)) >= 359.9:
        periodic_lon = True
        if i_debug >= 1:
            print('Regridding: Use periodic boundary conditions for DELTA ' +
                    'input data as it ' +
                    'appears to be periodic in longitudinal ' +
                    'direction.')
    else:
        periodic_lon = False


    #### IMPLEMENTATION WITH XESMF
    ##########################################################################
    ## XESMF sometimes alters the exact values of the latitude coordinate
    ## a little bit which was found to be problematic. Therefore, there is an
    ## xarray-only implmenetation below.
    if i_use_xesmf:
        ds_in = ds_delta
        regridder = xe.Regridder(ds_in, ds_gcm, "bilinear", 
                                 periodic=periodic_lon)
        print(regridder)
        ds_out = regridder(ds_in[var_name])
        ds_delta = ds_out.to_dataset(name=var_name)
        # keep attributes of variables and coordinates
        for field in [var_name, TIME_DELTA, PLEV_DELTA, LAT_DELTA, 
                      LON_DELTA, 'height']:
            if field in ds_in:
                ds_delta[field].attrs = ds_in[field].attrs
        # keep global attributes
        ds_delta.attrs = ds_in.attrs

    #### IMPLEMENTATION WITH XARRAY
    ##########################################################################
    ## The results should be identical to XESMF
    ## except for tiny differences that appear to originate from
    ## numerical precision.
    else:
        #### LATITUDE INTERPOLATION
        ######################################################################
        ## make sure latitude is increasing with index
        if (ds_delta[LAT_DELTA].isel({LAT_DELTA:0}).values > 
            ds_delta[LAT_DELTA].isel({LAT_DELTA:-1}).values):
            if i_debug >= 1:
                print('Regridding: DELTA data has opposite ' +
                      'order of latitude. Apply reindexing.')
            # flip latitude dimension
            ds_delta = ds_delta.reindex(
                    {LAT_DELTA:list(reversed(ds_delta[LAT_DELTA]))})

        ## If DELTA dataset reaches poles (almost), add a pole grid point
        ## with the zonal average of the values closest to the pole
        if np.max(targ_lat.values) + dlat_delta > 89.9:
            north = ds_delta.isel({LAT_DELTA:-1})
            north[LAT_DELTA].values = 90
            north[var_name] = north[var_name].mean(dim=[LON_DELTA])
            ds_delta = xr.concat([ds_delta,north], dim=LAT_DELTA)
        if np.min(targ_lat.values) - dlat_delta < -89.9:
            south = ds_delta.isel({LAT_DELTA:0})
            south[LAT_DELTA].values = -90
            south[var_name] = south[var_name].mean(dim=[LON_DELTA])
            ds_delta = xr.concat([south,ds_delta], dim=LAT_DELTA)

        ## make sure there is no extrapolation to the North and South
        if ( (np.max(targ_lat.values) > np.max(ds_delta[LAT_DELTA].values)) |
             (np.min(targ_lat.values) < np.min(ds_delta[LAT_DELTA].values))):
            print('DELTA lat: min {} max {}'.format(
                            np.min(ds_delta[LAT_DELTA].values),
                            np.max(ds_delta[LAT_DELTA].values))) 
            print('GCM lat: min {} max {}'.format(
                            np.min(targ_lat.values),
                            np.max(targ_lat.values))) 
            raise ValueError('GCM dataset extends further North or South ' +
                              'than DELTA dataset!. Perhaps consider using ' +
                              'GCM on a subdomain only if global coverage ' +
                              'is not required?') 

        ## run interpolation
        ds_delta = ds_delta.interp({LAT_DELTA:targ_lat})

        #### LONGITUDE INTERPOLATION
        ######################################################################
        ### Implement periodic boundary conditions
        ### This is also a check and fix for a different longitude format 
        ### e.g. DELTA -180:180 while GCM 0:360 (or vice versa)
        if periodic_lon:
            if np.max(targ_lon.values) > np.max(ds_delta[LON_DELTA].values):
                lon_above = ds_delta.assign_coords(
                            {LON_DELTA:ds_delta[LON_DELTA] + 360})
                ds_delta = xr.concat([ds_delta, lon_above], dim=LON_DELTA)
            if np.min(targ_lon.values) < np.min(ds_delta[LON_DELTA].values):
                lon_below = ds_delta.assign_coords(
                            {LON_DELTA:ds_delta[LON_DELTA] - 360})
                ds_delta = xr.concat([lon_below, ds_delta], dim=LON_DELTA)

        ## make sure there is no extrapolation to the East and West
        if ( (np.max(targ_lon.values) > np.max(ds_delta[LON_DELTA].values)) |
             (np.min(targ_lon.values) < np.min(ds_delta[LON_DELTA].values))):
            print('DELTA lon: min {} max {}'.format(
                            np.min(ds_delta[LON_DELTA].values),
                            np.max(ds_delta[LON_DELTA].values))) 
            print('GCM lon: min {} max {}'.format(
                            np.min(targ_lon.values),
                            np.max(targ_lon.values))) 
            raise ValueError('GCM dataset extends further East or West ' +
                              'than DELTA dataset!. Perhaps consider using ' +
                              'GCM on a subdomain only if global coverage ' +
                              'is not required?') 

        ## run interpolation
        ds_delta = ds_delta.interp({LON_DELTA:targ_lon})

    ## test for NaN
    if np.sum(np.isnan(ds_delta[var_name])).values > 0:
        raise ValueError('NaN in DELTA dataset after interpolation.')

    return(ds_delta)
