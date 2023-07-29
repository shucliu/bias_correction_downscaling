#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
description     Settings namelist for all routines in PGW for ERA5
authors		Before 2022: original developments by Roman Brogli
                Since 2022:  upgrade to PGW for ERA5 by Christoph Heim 
"""
##############################################################################
##############################################################################

### GENERAL SETTINGS
##############################################################################
# debug output level
i_debug = 2 # [0-2]

# Input and output file naming convention for the CTRL and climate delta
# (SCEN-CTRL) files from the GCM.
# ({} is placeholder for variable name).
file_name_bases = {
    'SCEN-CTRL':    '{}_delta.nc',
    'CTRL':         '{}_historical.nc',
}

# File naming convention for ERA5 files to be read in and written out.
gcm_file_name_base = 'cas{:%Y%m%d%H}0000.nc'

# dimension names in GCM file
TIME_GCM        = 'time'
LON_GCM         = 'lon'
LAT_GCM         = 'lat'
LEV_GCM         = 'level'
HLEV_GCM        = 'level1'
SOIL_HLEV_GCM   = 'soil1'

# dimension names in delta file
TIME_DELTA        = 'time'
LON_DELTA         = 'lon'
LAT_DELTA         = 'lat'
PLEV_DELTA        = 'plev'
LEV_DELTA         = 'lev'

# map from CMOR variable names
# to variable names used in ERA5 files being process.
var_name_map = {
    # surface geopotential
    'zgs'  :'FIS',
    # geopotential
    'zg'   :'PHI',
    # air temperature
    'ta'   :'T',   
    # near-surface temperature
    'tas'  :'T_SKIN',
    # soil temperature
#    'st'   :'T_SO',
    # air relative humidity
    'hur'  :'RELHUM',
    # air specific humidity
    'hus'  :'QV',
    # lon-wind speed
    'ua'   :'U',
    # lat-wind speed 
    'va'   :'V',
    # surface pressure
    'ps'   :'PS',
}


### 02 PREPROCESS DELTAS 
##############################################################################

### SMOOTHING
####################################

### REGRIDDING
####################################
# depending on whether the xesmf pacakge is installed, it can be used
# for interpolation. Else, an xarray-based method is used.
# the latter should be identical to XESMF
# except for tiny differences that appear to originate from
# numerical precision
i_use_xesmf_regridding = 0


### SURFACE PRESSURE ADJUSTMENT SETTINGS 
##########################################################################
# reference pressure level
# if set to None, the reference pressure level is chosen locally.
# if the climate deltas have low vertical resolution (e.g. Amon data
# with only 6 vertical levels between 1000-500 hPa), settting
# p_ref_inp = None may be better. See publication
# for more information.
p_ref_inp = 30000 #50000 # Pa
#p_ref_inp = None
# surface pressure adjustment factor in the iterative routine
adj_factor = 0.95
# convergence threshold (maximum geopotential error)
# if procedure does not converge, raise this value a little bit.
thresh_phi_ref_max_error = 0.15
# maximum number of iterations before error is raised.
max_n_iter = 20
# re-interpolation turned on/off
i_reinterp = 0
##########################################################################

