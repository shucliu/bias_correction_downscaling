#GCM!/usr/bin/python
# -*- coding: utf-8 -*-
"""
description     Bias-corrected downscaling main routine to update GCM files with climatology 
                bias 
authors		Before 2022: original developments by Roman Brogli
                Since 2022:  upgrade to PGW for ERA by Christoph Heim 
                Since 2023:  modify the code for bias-corrected downscaling bu Shuchang Liu
"""
##############################################################################
import argparse, os
import xarray as xr
import numpy as np
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
from datetime import datetime, timedelta
from functions import (
    specific_to_relative_humidity,
    relative_to_specific_humidity,
    load_delta,
    load_delta_interp,
    integ_geopot,
    interp_logp_4d,
    determine_p_ref,
    )
from constants import CON_G, CON_RD
from parallel import IterMP
from settings import (
    i_debug,
    gcm_file_name_base,
    var_name_map,
    TIME_GCM, LEV_GCM, HLEV_GCM, LON_GCM, LAT_GCM, SOIL_HLEV_GCM,
    TIME_DELTA, PLEV_DELTA,
    i_reinterp,
    p_ref_inp,
    thresh_phi_ref_max_error,
    max_n_iter,
    adj_factor,
    file_name_bases,
    )
##############################################################################

def bias_correction(inp_gcm_file_path, out_gcm_file_path,
                delta_input_dir, step_dt,
                ignore_top_pressure_error,
                debug_mode=None):
    if i_debug >= 0:
        print('Start working on input file {}'.format(inp_gcm_file_path))

    #########################################################################
    ### PREPARATION STEPS
    #########################################################################
    # containers for variable computation
    vars_gcm = {}
    vars_corrected_gcm = {}
    deltas = {}

    # open data set
    gcm_file = xr.open_dataset(inp_gcm_file_path, decode_cf=False)

    ## compute pressure on GCM full levels and half levels
    # pressure on half levels
    pa_hl_gcm = (gcm_file.ak + 
                gcm_file[var_name_map['ps']] * gcm_file.bk).transpose(
                TIME_GCM, HLEV_GCM, LAT_GCM, LON_GCM)
    # if akm and akb coefficients (for full levels) exist, use them
    if 'akm' in gcm_file:
        akm = gcm_file.akm
        bkm = gcm_file.bkm
    # if akm and abk  coefficients do not exist, computed them
    # with the average of the half-level coefficients above and below
    else:
        akm = (
            0.5 * gcm_file.ak.diff(
            dim=HLEV_GCM, 
            label='lower').rename({HLEV_GCM:LEV_GCM}) + 
            gcm_file.ak.isel({HLEV_GCM:range(len(gcm_file.level1)-1)}).values
        )
        bkm = (
            0.5 * gcm_file.bk.diff(
            dim=HLEV_GCM, 
            label='lower').rename({HLEV_GCM:LEV_GCM}) + 
            gcm_file.bk.isel({HLEV_GCM:range(len(gcm_file.level1)-1)}).values
        )
    # pressure on full levels
    pa_gcm = (akm + gcm_file[var_name_map['ps']] * bkm).transpose(
                TIME_GCM, LEV_GCM, LAT_GCM, LON_GCM)

    # compute relative humidity in GCM
    gcm_file[var_name_map['hur']] = specific_to_relative_humidity(
                        gcm_file[var_name_map['hus']], 
                        pa_gcm, gcm_file[var_name_map['ta']]).transpose(
                        TIME_GCM, LEV_GCM, LAT_GCM, LON_GCM)

    #########################################################################
    ### UPDATE SURFACE AND SOIL TEMPERATURE
    #########################################################################
    # update surface temperature using near-surface temperature delta
    var_name = 'tas'
    if i_debug >= 2:
        print('update {}'.format(var_name))
    delta_tas = load_delta(delta_input_dir, var_name,
                           gcm_file[TIME_GCM], step_dt)
    gcm_file[var_name_map[var_name]].values += delta_tas.values
    # store delta for output in case of --debug_mode = interpolate_full
    deltas[var_name] = delta_tas

    #########################################################################
    ### START UPDATING 3D FIELDS
    #########################################################################
    # If no re-interpolation is done, the final bias-corrected GCM
    # variables can be computed already now, before updating the 
    # surface pressure. This means that the climatology bias or
    # interpolated on the GCM model levels of the GCM climate state.
    if not i_reinterp:

        ### interpolate climatology bias onto GCM grid
        for var_name in ['ta','hur','ua','va']:
            if i_debug >= 2:
                print('update {}'.format(var_name))

            ## interpolate climatology bias to GCM model levels
            ## use GCM climate state
            delta_var = load_delta_interp(delta_input_dir,
                    var_name, pa_gcm, gcm_file[TIME_GCM], step_dt,
                    ignore_top_pressure_error)
            deltas[var_name] = delta_var

            ## compute bias-corrected GCM variables
            vars_corrected_gcm[var_name] = (
                    gcm_file[var_name_map[var_name]] + 
                    deltas[var_name]
            )


    #########################################################################
    ### UPDATE SURFACE PRESSURE USING ITGCMTIVE PROCEDURE
    #########################################################################
    if i_debug >= 2:
        print('###### Start with iterative surface pressure adjustment.')
    # change in surface pressure between GCM and bias-corrected GCM
    delta_ps = xr.zeros_like(gcm_file[var_name_map['ps']])
    # increment to adjust delta_ps with each iteration
    adj_ps = xr.zeros_like(gcm_file[var_name_map['ps']])
    # maximum error in geopotential (used in iteration)
    phi_ref_max_error = np.inf

    it = 1
    while phi_ref_max_error > thresh_phi_ref_max_error:

        # update surface pressure
        delta_ps += adj_ps
        ps_corrected_gcm = gcm_file[var_name_map['ps']] + delta_ps

        # recompute pressure on full and half levels
        pa_corrected_gcm = (akm + ps_corrected_gcm * bkm).transpose(
                    TIME_GCM, LEV_GCM, LAT_GCM, LON_GCM)
        pa_hl_corrected_gcm = (gcm_file.ak + ps_corrected_gcm * gcm_file.bk).transpose(
                    TIME_GCM, HLEV_GCM, LAT_GCM, LON_GCM)


        if i_reinterp:
            # interpolate GCM climate state variables as well as
            # climatology bias onto updated model levels, and
            # compute bias-corrected climate state variables
            for var_name in ['ta', 'hur']:
                vars_gcm[var_name] = interp_logp_4d(
                                gcm_file[var_name_map[var_name]], 
                                pa_gcm, pa_corrected_gcm, extrapolate='constant')
                deltas[var_name] = load_delta_interp(delta_input_dir,
                                                var_name, pa_corrected_gcm,
                                                gcm_file[TIME_GCM], step_dt,
                                                ignore_top_pressure_error)
                vars_corrected_gcm[var_name] = vars_gcm[var_name] + deltas[var_name]

        # Determine current reference pressure (p_ref)
        if p_ref_inp is None:
            # get delta pressure levels as candidates for reference pressure
            p_ref_opts = load_delta(delta_input_dir, 'zg',
                                gcm_file[TIME_DELTA], step_dt)[PLEV_DELTA]
            # maximum reference pressure in GCM and bias-corrected GCM climate states
            # (take 95% of surface pressure to ensure that a few model
            # levels are located in between which makes the solution
            # smoother).
            p_min_gcm = pa_hl_gcm.isel(
                        {HLEV_GCM:len(pa_hl_gcm[HLEV_GCM])-1}) * 0.95
            p_min_corrected_gcm = pa_hl_corrected_gcm.isel(
                        {HLEV_GCM:len(pa_hl_gcm[HLEV_GCM])-1}) * 0.95
            # reference pressure from a former iteration already set?
            try:
                p_ref_last = p_ref
            except UnboundLocalError:
                p_ref_last = None
            # determine local reference pressure
            p_ref = xr.apply_ufunc(determine_p_ref, p_min_gcm, p_min_corrected_gcm, 
                    p_ref_opts, p_ref_last,
                    input_core_dims=[[],[],[PLEV_DELTA],[]],
                    vectorize=True)
            if HLEV_GCM in p_ref.coords:
                del p_ref[HLEV_GCM]
            # make sure a reference pressure above the required model
            # level could be found everywhere
            if np.any(np.isnan(p_ref)):
                raise ValueError('No reference pressure level above the ' +
                        'required local minimum pressure level could not ' +
                        'be found everywhere. ' +
                        'This is likely the case because your geopotential ' +
                        'data set does not reach up high enough (e.g. only to ' +
                        '500 hPa instead of e.g. 300 hPa?)')
        else:
            p_ref = p_ref_inp

        #p_sfc_era.to_netcdf('psfc_era.nc')
        #p_ref.to_netcdf('pref.nc')
        #quit()

        # convert relative humidity to speicifc humidity in corrected gcm
        vars_corrected_gcm['hus'] = relative_to_specific_humidity(
                        vars_corrected_gcm['hur'], pa_corrected_gcm, vars_corrected_gcm['ta'])

        # compute updated geopotential at reference pressure
        phi_ref_corrected_gcm = integ_geopot(pa_hl_corrected_gcm, gcm_file.FIS, vars_corrected_gcm['ta'], 
                                    vars_corrected_gcm['hus'], gcm_file[HLEV_GCM], p_ref)

        # recompute original geopotential at currently used 
        # reference pressure level
        phi_ref_gcm = integ_geopot(pa_hl_gcm, gcm_file.FIS,
                                    gcm_file[var_name_map['ta']], 
                                    gcm_file[var_name_map['hus']], 
                                    gcm_file[HLEV_GCM], p_ref)

        delta_phi_ref = phi_ref_corrected_gcm - phi_ref_gcm

        ## load climate delta at currently used reference pressure level
        climate_delta_phi_ref = load_delta(delta_input_dir, 'zg',
                            gcm_file[TIME_GCM], step_dt) * CON_G
        climate_delta_phi_ref = climate_delta_phi_ref.sel({PLEV_DELTA:p_ref})
        del climate_delta_phi_ref[PLEV_DELTA]

        # error in future geopotential
        phi_ref_error = delta_phi_ref.values - climate_delta_phi_ref.values

        adj_ps = - adj_factor * ps_corrected_gcm / (
                CON_RD * 
                vars_corrected_gcm['ta'].sel({LEV_GCM:np.max(gcm_file[LEV_GCM])})
            ) * phi_ref_error
        if LEV_GCM in adj_ps.coords:
            del adj_ps[LEV_GCM]

        phi_ref_max_error = np.abs(phi_ref_error).max()
        if i_debug >= 2:
            print('### iteration {:03d}, phi max error: {}'.
                            format(it, phi_ref_max_error))

        it += 1

        if it > max_n_iter:
            raise ValueError('ERROR! Pressure adjustment did not converge '+
                  'for file {}. '.format(inp_gcm_file_path) +
                  'Consider increasing the value for "max_n_iter" in ' +
                  'settings.py')

    #########################################################################
    ### FINISH UPDATING 3D FIELDS
    #########################################################################
    # store computed delta ps for output in case of 
    # --debug_mode = interpolate_full
    deltas['ps'] = ps_corrected_gcm - gcm_file.PS

    ## If re-interpolation is enabled, interpolate climatology bias for
    ## ua and va onto final bias-corrected climate state GCM model levels.
    if i_reinterp:
        for var_name in ['ua', 'va']:
            if i_debug >= 2:
                print('add {}'.format(var_name))
            ver_gcm = interp_logp_4d(gcm_file[var_name_map[var_name]], 
                            pa_gcm, pa_corrected_gcm, extrapolate='constant')
            delta_var = load_delta_interp(delta_input_dir,
                    var_name, pa_corrected_gcm,
                    gcm_file[TIME_GCM], step_dt,
                    ignore_top_pressure_error)
            vars_corrected_gcm[var_name] = var_gcm + delta_var
            # store delta for output in case of 
            # --debug_mode = interpolate_full
            deltas[var_name] = delta_var

    #########################################################################
    ### DEBUG MODE
    #########################################################################
    ## for debug_mode == interpolate_full, write final climatology bias
    ## to output directory
    if debug_mode == 'interpolate_full':
        var_names = ['tas','ps','ta','hur','ua','va','st']
        for var_name in var_names:
            print(var_name)
            # creat output file name
            out_file_path = os.path.join(Path(out_gcm_file_path).parents[0],
                                '{}_delta_{}'.format(var_name_map[var_name], 
                                            Path(out_gcm_file_path).name))
            # convert to dataset
            delta = deltas[var_name].to_dataset(name=var_name_map[var_name])
            # save climate delta
            delta.to_netcdf(out_file_path, mode='w')

    #########################################################################
    ### SAVE bias-corrected GCM FILE
    #########################################################################
    ## for production mode, modify GCM file and save
    else:
        ## update fields in GCM file
        gcm_file[var_name_map['ps']] = ps_corrected_gcm
        for var_name in ['ta', 'hus', 'ua', 'va']:
            gcm_file[var_name_map[var_name]] = vars_corrected_gcm[var_name]
        del gcm_file[var_name_map['hur']]


        ## save updated GCM file
        gcm_file.to_netcdf(out_gcm_file_path, mode='w')
        gcm_file.close()
        if i_debug >= 1:
            print('Done. Saved to file {}.'.format(out_gcm_file_path))



##############################################################################

def debug_interpolate_time(
                inp_gcm_file_path, out_gcm_file_path,
                delta_input_dir, step_dt,
                ignore_top_pressure_error,
                debug_mode=None):
    """
    Debugging function to test time interpolation. Is called if input
    inputg argument --debug_mode is set to "interpolate_time".
    """
    # load input GCM file
    # in this debugging function, the only purpose of this is to obtain 
    # the time format of the GCM file
    gcm_file = xr.open_dataset(inp_gcm_file_path, decode_cf=False)

    var_names = ['tas','hurs','ps','ta','hur','ua','va','zg']
    for var_name in var_names:
        print(var_name)
        ## for ps take era climatology file while for all other variables
        ## take climate delta file
        #if var_name == 'ps':
        #    name_base = era_climate_file_name_base
        #else:
        #    name_base = climate_delta_file_name_base
        name_base = climate_delta_file_name_base
        # create gcm input file name (excluding ".nc")
        gcm_file_name = name_base.format(var_name).split('.nc')[0]
        # creat output file name
        out_file_path = os.path.join(Path(out_gcm_file_path).parents[0],
                                    '{}_{}'.format(gcm_file_name, 
                                        Path(out_gcm_file_path).name))
        # load climate delta interpolated in time only
        delta = load_delta(delta_input_dir, var_name, gcm_file[TIME_GCM], 
                       target_date_time=step_dt,
                       name_base=name_base)
        # convert to dataset
        delta = delta.to_dataset(name=var_name)

        delta.to_netcdf(out_file_path, mode='w')
    gcm_file.close()






##############################################################################
if __name__ == "__main__":
    ## input arguments
    parser = argparse.ArgumentParser(description =
    """
    Perturb GCM with climatology bias (ERA5-GCM). Settings can be made in
    "settings.py".
    ##########################################################################

    Main function to update GCM files with the climatology bias.
    The terminology used is CTRL referring to the historical GCM
    climatology, SCEN referring to the bias-corrected
    climatology, and SCEN-CTRL (a.k.a. -bias) referring to the
    bias signal which should be applied to the GCM files.
    The script adds (and requires) SCEN-CTRL for:
        - ua
        - va
        - ta (using tas)
        - hus (computed using a hur and hurs climate delta)
        - surface and soil temperature
    and consequently iteratively updates ps to maintain hydrostatic
    balance. During this, the climatology bias for zg is additionally required.
    Finally, the CTRL ps is also needed.

    ##########################################################################

    If the variable names in the GCM files to be processed deviate from
    the CMOR convention, the dict 'var_name_map' in the file 
    settings.py allows to map between the CMOR names and the names in the GCM
    file. Also the coordinate names in the GCM or the GCM climate
    delta files can be changed in settings.py, if required.

    ##########################################################################

    The code can be run in parallel on multiple GCM files at the same time.
    See input arguments.

    ##########################################################################

    Some more information about the iterative surface pressure
    adjustment:

    - The procedure requires a reference pressure level (e.g. 500 hPa) for
    which the geopotential is computed. Based on the deviation between the
    computed and the GCM reference pressure geopotential, the surface pressure
    is adjusted. Since the climatology bias may not always be available at 
    native vertical GCM resolution, but the climatology bias for the geopotential
    on one specific pressure level itself is computed by the GCM using data
    from all GCM model levels, this introduces an error in the surface
    pressure adjustment used here. See publication for more details.
    The higher (in terms of altitdue) the reference pressure is chosen, 
    the larger this error may get. 
    To alleviate this problem, the default option is that the reference
    pressure is determined locally as the lowest possible pressure above
    the surface for which a climate delta for the geopotential is available.
    In general -- even more so if climatology bias have coarse vertical 
    resolution -- it seems to be a good choice to use this default.

    - If the iteration does not converge, 'thresh_phi_ref_max_error' in
    the file settings.py may have to be raised a little bit. Setting
    i_debug = 2 may help to diagnose if this helps.


    - As a default option, the climatology bias are interpolated to
    the GCM model levels of the GCM climate state before the surface
    pressure is adjusted (i_reinterp = 0).
    There is an option implemented (i_reinterp = 1) in which the
    deltas are re-interpolated on the updated GCM model levels
    with each iteration of surface pressure adjustment. This was
    found to lead more balanced PGW climate states if the climate
    deltas have coarse vertical resolution. However, it also
    implies that the GCM fields are extrapolated at the surface
    (if the surface pressure increases) the effect of which was not
    tested in detail. The extrapolation is done assuming that the
    boundary values are constant, which is not ideal for height-dependent
    variables like e.g. temperature. As a default, it is recommended to set
    i_reinterp = 0.

    ##########################################################################

    """, formatter_class=RawDescriptionHelpFormatter)

    # input gcm directory
    parser.add_argument('-i', '--input_dir', type=str, default=None,
            help='Directory with GCM input files to process. ' +
                 'These files are not overwritten but copies will ' +
                 'be save in --output_dir .')

    # output gcm directory
    parser.add_argument('-o', '--output_dir', type=str, default=None,
            help='Directory to store processed GCM files.')

    # first bc step to compute 
    parser.add_argument('-f', '--first_step', type=str,
            default='2006080200',
            help='Date of first GCM time step to process. Format should ' +
            'be YYYYMMDDHH.')

    # last bc step to compute 
    parser.add_argument('-l', '--last_step', type=str,
            default='2006080300',
            help='Date of last GCM time step to process. Format should ' +
            'be YYYYMMDDHH.')

    # delta hour increments
    parser.add_argument('-H', '--hour_inc_step', type=int, default=3,
            help='Hourly increment of the GCM time steps to process '+
            'between --first_step and --last_step. Default value ' +
            'is 3-hourly, i.e. (00, 03, 06, 09, 12, 15, 18, 21 UTC).')

    # climate delta directory (already remapped to GCM grid)
    parser.add_argument('-d', '--delta_input_dir', type=str, default=None,
            help='Directory with GCM climatology bias (SCEN-CTRL) to be used. ' +
            'This directory should have a climate delta for ta,hur,' +
            'ua,va,zg,tas,hurs (e.g. ta_delta.nc), as well as the ' +
            'CTRL climatology value for ps (e.g. ps_historical.nc). ' +
            'All files have to be horizontally remapped to the grid of ' +
            'the GCM files used.')

    # number of parallel jobs
    parser.add_argument('-p', '--n_par', type=int, default=1,
            help='Number of parallel tasks. Parallelization is done ' +
            'on the level of individual GCM files being processed at ' +
            'the same time.')

    # flag to ignore the error from to pressure extrapolation at the model top
    parser.add_argument('-t', '--ignore_top_pressure_error',
            action='store_true',
            help='Flag to ignore an error due to pressure ' +
            'extrapolation at the model top if GCM climatology bias reach ' +
            'up less far than GCM. This can only be done if GCM data ' +
            'is not used by the limited-area model '+
            'beyond the upper-most level of the GCM climate ' +
            'deltas!!')

    # input gcm directory
    parser.add_argument('-D', '--debug_mode', type=str, default=None,
            help='If this flag is set, the GCM files will not be ' +
                 'modified but instead the processed climatology bias '
                 'are written to the output directory. There are two ' +
                 'options: for "-D interpolate_time", the climatology bias ' +
                 'are only interpolated to the time of the GCM files ' +
                 'and then stored. for "-D interpolate_full", the ' +
                 'full routine is run but instead of the processed GCM ' +
                 'files, only the difference between the processed and ' +
                 'the unprocessed GCM files is store (i.e. the climatology ' +
                 'bias after full interpolation to the GCM grid).')


    args = parser.parse_args()
    ##########################################################################

    # make sure required input arguments are set.
    if args.input_dir is None:
        raise ValueError('Input directory (-i) is required.')
    if args.output_dir is None:
        raise ValueError('Output directory (-o) is required.')
    if args.delta_input_dir is None:
        raise ValueError('Delta input directory (-d) is required.')

    # check for debug mode
    if args.debug_mode is not None:
        if args.debug_mode not in ['interpolate_time', 'interpolate_full']:
            raise ValueError('Invalid input for argument --debug_mode! ' +
                            'Valid arguments are: ' +
                             '"interpolate_time" or "interpolate_full"')

    # first date and last date to datetime object
    first_step = datetime.strptime(args.first_step, '%Y%m%d%H')
    last_step = datetime.strptime(args.last_step, '%Y%m%d%H')

    # time steps to process
    step_dts = np.arange(first_step,
                        last_step+timedelta(hours=args.hour_inc_step),
                        timedelta(hours=args.hour_inc_step)).tolist()

    # if output directory doesn't exist create it
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    IMP = IterMP(njobs=args.n_par, run_async=True)
    fargs = dict(
        delta_input_dir = args.delta_input_dir,
        ignore_top_pressure_error = args.ignore_top_pressure_error,
        debug_mode = args.debug_mode,
    )
    step_args = []

    ##########################################################################
    # iterate over time step and prepare function arguments
    for step_dt in step_dts:
        print(step_dt)

        # set output and input GCM file
        inp_gcm_file_path = os.path.join(args.input_dir, 
                gcm_file_name_base.format(step_dt))
        out_gcm_file_path = os.path.join(args.output_dir, 
                gcm_file_name_base.format(step_dt))

        step_args.append(dict(
            inp_gcm_file_path = inp_gcm_file_path,
            out_gcm_file_path = out_gcm_file_path,
            step_dt = step_dt
            )
        )

    # choose either main function (bias_correction) for production mode and
    # debug mode "interpolate_full", or function time_interpolation 
    # for debug mode "interpolate_time"
    if (args.debug_mode is None) or (args.debug_mode == 'interpolate_full'):
        run_function = bias_correction
    elif args.debug_mode == 'interpolate_time':
        run_function = debug_interpolate_time
    else:
        raise NotImplementedError()

    # run in parallel if args.n_par > 1
    IMP.run(run_function, fargs, step_args)

