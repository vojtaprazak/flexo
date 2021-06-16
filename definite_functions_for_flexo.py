#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:23:35 2019

@author: vojta
"""
#VP EDIT: 17/11/2020 moved this to processchunks_flexo_wrapper
#import matplotlib
##this is required.  Even writing plots to file
#matplotlib.use('Agg')

import sys
import os
from os.path import join, abspath, realpath, split, isfile, isdir
import mrcfile
import numpy as np
from subprocess import check_output, Popen, PIPE
#EDIT VP 11/11/2020 changed MapParser_f32_new star import, also imported Map
import MapParser_f32_new
from EMMap_noheadoverwrite import Map
from PEETModelParser import PEETmodel
from PEETMotiveList import PEETMotiveList
from PEETPRMParser import PEETPRMFile 
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks, butter, freqs
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage.morphology import grey_dilation, grey_erosion
from scipy.optimize import minimize
#from scipy import misc
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy import fftpack as fft
from scipy import stats
from numpy.fft import fftfreq#, fftshift #already defined from scipy
#from scipy import signal #clash with signal
from scipy.ndimage.interpolation import zoom, shift, rotate
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
#from joblib import Parallel, delayed
#import multiprocessing
from copy import deepcopy
#from shutil import copyfile
from skimage.filters import threshold_yen
from skimage.feature import peak_local_max
import glob
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import signal
import time
import warnings

#import timeit
if sys.version[0] == '3':
    from itertools import zip_longest
elif sys.version[0] == '2':
    from itertools import izip_longest as zip_longest

def deprecated_cc_but_better(pcl_n, ref, query, limit, interp, tilt_angles,
                  zero_tilt, centre_bias = 0.1, thickness_scaling = 1.5,
                  step = 5, out_dir = False, debug = 0,
                  log = False, #new
                  allow_large_jumps = False):

    """
    Intended to take in data for 1 particle.
    Zero_tlt from 1
    """
    if not log and debug > 0:
        print('cc_but_better: No log file specified, logging disabled.')
        debug = 0
        
    debug_out = []
    map_size = limit*interp*2
    max_iters = 8
    iter_n = 0
    imp_mat = []
    imp_bounds = 1., 1.4
    #imp values > ~1.4 are unreasonably high and suggest something failed
    starting_centre_bias = centre_bias
    
    #define middle tilts, zero_tlt is numbered from 1
    if len(ref) < step:
        bot, top = 0, len(ref)
    else:
        bot = int(zero_tilt - 1 - (step - 1)//2)
        top = int(zero_tilt - 1 + (step - 1)//2 + 1)
    
    #make stack of ccmaps
    ccmaps = np.array([(get_cc_map(ref[x], query[x], limit, interp))
                        for x in range(len(ref))])
    #CC values need to be shifted to positive values!
    ccmaps = ccmaps - np.min(ccmaps, axis = (1,2))[:, None, None]

    #######################################################################
    #initial CC of middle section: distance matrix biased towards 0,0 shift
    out_name = "%02d_mid_compare_ccs.png" % pcl_n
    write_to_log(log, (('#'*5 + 'Particle %s' + '#'*5 +
                       '\nAligning tilts %s-%s (numbered from 1)')
                       % (pcl_n, bot + 1, top + 1)), debug)
    
    shifts, imp = cc_middle(pcl_n, ref, query, tilt_angles, ccmaps, interp,
                            limit, map_size, top, bot, centre_bias,
                            thickness_scaling, out_dir, debug,
                            'init', out_name)
    #increase centre_bias if imp < 1
    while (imp < imp_bounds[0] or imp > imp_bounds[1]) and iter_n < max_iters:
        iter_n += 1
        #break if nothing changes after 2 iterations
        if np.any(np.unique(imp_mat, return_counts = True)[1] > 2):
            write_to_log(log, ("compare_ccs: peak value has not changed in"
                    " the last three iterations (%.4f).  Continuing with"
                    " current shifts.") % imp, debug)
            #reset centre bias...debatable
            centre_bias = starting_centre_bias
            break     
        ocb = centre_bias
        #TBD!!!!
        centre_bias = failed_peak_scaling(centre_bias)        
        if imp > imp_bounds[1]:
            debug_out.extend(
            ("compare_ccs: averaged peak improvement\t %.4f.\nThis is an"
             "unreasonable amount and something probably went wrong." % imp))

        else:
            debug_out.extend('compare_ccs: negative peak change %.4f.' % imp)
        debug_out.extend(
            ("Increasing centre_bias from %.2f to %.2f. (iteration %s of %s)" %
                                    (ocb, centre_bias, iter_n, max_iters - 1)))

        med = (np.median(shifts[bot : top, 0]),
               np.median(shifts[bot : top, 1]))
        compare_out_name = "%02d_mid_compare_ccs_iter%s.png" % (pcl_n, iter_n)
        biased_out_name = out_name = '%02d_mid_%s-%s_iter%s.png' % (
                                            pcl_n, bot + 1,top + 1, iter_n)
        shifts, imp = cc_middle(pcl_n, ref, query, tilt_angles, ccmaps, interp,
                                limit, map_size, top, bot, centre_bias,
                                thickness_scaling, out_dir, debug, med,
                                compare_out_name, biased_out_name, iter_n)     
        imp_mat.append(imp)
    else:
        debug_out.extend("compare_ccs: averaged peak improvement %s" % imp)
        #print wmessage   
         
    #walk up   
    if len(ref) > step:
        shifts, wmsg = cc_walk(pcl_n, ccmaps, shifts, interp, limit,
                       len(ref), top, 1, step, tilt_angles, centre_bias,
                       thickness_scaling, out_dir, debug,
                       allow_large_jumps)
        out_name = ("%02d_%s-%s_compare_ccs.png"
                    % (pcl_n, top + 1, len(ref) + 1))
        imp, avg_ccmaps = compare_ccs(pcl_n, ref, query, len(ref), top, shifts,
                                      interp, limit, out_dir, debug)
        if debug > 0:
            debug_out.extend("\nStarting cc_walk up.")
            debug_out.extend(wmsg)
            wmessage = ["Particle %s tilts %s-%s averaged peak change %.4f" %
                        (pcl_n, top + 1, len(ref) + 1, imp)]
            debug_out.extend(wmessage)
            #print wmessage
        #walk down
        shifts, wmsg = cc_walk(pcl_n, ccmaps, shifts, interp, limit,
                       bot, 0, -1, step, tilt_angles, centre_bias,
                       thickness_scaling, out_dir, debug,
                       allow_large_jumps)
        out_name = "%02d_%s-%s_compare_ccs.png" % (pcl_n, 1, bot + 1)
        imp, avg_ccmaps = compare_ccs(pcl_n, ref, query, bot, 0, shifts,
                                      interp, limit, out_dir, debug)
        if debug > 0:
            debug_out.extend("\nStarting cc_walk down.")
            debug_out.extend(wmsg)
            debug_out.extend(
                    "Particle %s tilts %s-%s averaged peak change %.4f" %
                        (pcl_n, top + 1, len(ref) + 1, imp))
            #print wmessage
    shifts = np.divide(shifts, float(interp))-float(limit)
    #xy is flipped
    shifts = np.flip(shifts, axis = 1)
    return shifts, debug_out, ccmaps

def deprecated_get_cc_map(target, probe, max_dist, interp = 1,
                          outfile = False, zoom_cc_map = True):
    """Outputs cross-correlation map of two images.
    Input arguments:
        target/reference image [numpy array]
        probe/query image [numpy array]
        maximum distance to search [even integer]
        interpolation [integer]
        outfile: if intered, writes an MRC image of the map     
    Returns CC map [numpy array]
    """
    if interp > 1 and not zoom_cc_map:
        target, probe = zoom(target, interp), zoom(probe, interp)
    #norm
    target = (target - np.mean(target))/(np.std(target))
    probe = (probe - np.mean(probe))/(np.std(probe))
        
    cc_fft = fftn(target) * np.conj(fftn(probe))
    cc_fft = ifftn(cc_fft).real
    cc_fft = ifftshift(cc_fft)
    if interp > 1 and zoom_cc_map:
        cc_fft = zoom(cc_fft, interp)
    edge = int((cc_fft.shape[0] - (max_dist * 2 * interp))//2) #only square iamges!!!!!!
    cc_fft = cc_fft[edge:-edge, edge:-edge]
    if outfile:
        with mrcfile.new(outfile, overwrite=True) as mrc:
            cc_fft = cc_fft.copy(order='C')
            mrc.set_data(np.float16(cc_fft))
    return cc_fft

def deprecated_buggy_peaks(cc_map):
    """Finds peaks in 2D array.
    Input arguments:
        2D array, e.g. CC map
    Returns:
        peak coordinates sorted by maximum value [2D array, XY]
        peak values
    """
    pkxy = peak_local_max(cc_map, 1)
    warn = 0
    if pkxy.ndim > 1: #multipel peaks --> sort by peak hight
        peakval = cc_map[pkxy[:, 0], pkxy[:, 1]] 
        tt = np.argsort(peakval)
        peakval = sorted(np.divide(peakval, np.std(cc_map)), reverse =True)
        pkxy = pkxy[tt[::-1]] #sort using tt in reverse order
    if len(pkxy) == 0: #take map maximum if peak_local_max fails to find a peak
        pkxy = np.array(np.where(cc_map == cc_map.max())).squeeze()
        warn = 1
        peakval = np.divide(cc_map[pkxy[0], pkxy[1]], float(np.std(cc_map)))
    return pkxy, peakval, warn


##########################################
    #####################################



def write_mrc(out_name, data):
    with mrcfile.new(out_name, overwrite=True) as mrc:
        #cc_fft = np.swapaxes(cc_fft,0,2).copy(order='C')
        #out_data = data.copy(order='C')
        mrc.set_data(np.float32(data))
        #float16 doesn't get written out properly

def find_nearest(array, value):
    """modified from stackoverflow 2566412"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_apix_and_size(path):
    p = float(check_output('header -p %s' % path, shell = True).split()[0])
    s = np.array([int(x) for x in (check_output('header -s %s'\
                  % path, shell = True).split())])
    return p, s  

def get_origin(path):
    s = np.array([float(x) for x in (check_output('header -o %s'\
                  % path, shell = True).split())])    
    return s    

def read_comfiles(rec_dir):
    if not isfile(abspath(join(rec_dir, 'newst.com'))):
        raise Exception("File not found. %s" % abspath(join(rec_dir, 'newst.com')))
    else:
        with open(abspath(join(rec_dir, 'newst.com')),'r') as f1:
            for line in f1:
                try:
                    if line.split()[0] == ('InputFile'):
                        st = join(rec_dir, line.split()[1])
                    elif line.split()[0] == ('TransformFile'):
                        xf = join(rec_dir, line.split()[1])
#                    elif line.split()[0] == ('SizeToOutputInXandY'):
#                        sides = [int(s) for s in line.split()[1].split(',')]
#                        orientation = sides[0] - sides[1]
#        f.write('\nSizeToOutputInXandY\t%s,%s' %
#                (tomo_size[0], tomo_size[1]))

                except:
                    #meant to catch white spaces
                    pass
    #check for excludelist, xaxistilt and thickness
    excludelist = []
    if not isfile(abspath(join(rec_dir, 'tilt.com'))):
        raise Exception("File not found. %s" % abspath(join(rec_dir, 'tilt.com')))
    else:
        with open(abspath(join(rec_dir, 'tilt.com')),'r') as f1:
            for line in f1:
                try:              
                    if line.split()[0] == ('InputProjections'):
                        ali = join(rec_dir, line.split()[1])
                    elif line.split()[0] == ('TILTFILE'):
                        tlt = join(rec_dir, line.split()[1])
                    elif line.split()[0] == ('THICKNESS'):
                        thickness = int(line.split()[1])
                    elif line.split()[0] == ('XAXISTILT'):
                        global_xtilt = (line.split()[1])
                    elif line.split()[0]==('LOCALFILE'):
                        localxf = join(rec_dir, line.split()[1])
                    elif line.split()[0]==('nZFACTORFILE'):
                        zfac = join(rec_dir, line.split()[1])
                    elif line.split()[0]==('nXAXISTILT'):
                        xfile = join(rec_dir, line.split()[1])

                    elif line.split()[0]==('EXCLUDELIST'):
                        # DV CHANGE (extensive changes to excludelist parsing)
                        test = line.split()
                        if len(test) == 2:
                            # Comma separated/only one value
                            a = test[1].split(',')
                        else:
                            # Space separated
                            a = test[1:]
                        for b in a:
                            c = b.split('-')
                            if len(c) == 1:
                                excludelist.append(int(b))
                            else:
                                excludelist.extend(list(range(int(c[0]),
                                                         int(c[1]) + 1)))
                        excludelist.sort()  
                        #VP change 13/06/2020: excludelist should be numbered
                        #from 1.

                    elif line.split()[0]==('OFFSET'):
                        OFFSET = float(line.split()[1])    
                    elif line.split()[0]==('SHIFT'):
                        SHIFT = np.array([float(line.split()[1]),
                                          float(line.split()[2])])  
                except:
                    #meant to catch white spaces
                    pass
    try:
        str(zfac)
    except:
        zfac = False
    try:
        str(xfile)
    except:
        xfile = False
    try:
        OFFSET + 1
    except:
        OFFSET = 0.
    try:
        SHIFT + 1
    except:
        SHIFT = np.array([0.,0.])
    try:
        isinstance(global_xtilt, str)
    except:
        global_xtilt = False
    try:
        isinstance(localxf, str)
    except:
        localxf = False
    if not excludelist:
        excludelist = False
    try:
        test = thickness
    except:
        raise Exception("Tilt.com does not contain entry for tomogram " +
                        "thickness 'THICKNESS'.") 
        
    if not isfile(abspath(join(rec_dir, 'align.com'))):
        raise Exception("File not found. %s" % abspath(
                                        join(rec_dir, 'align.com')))
    else:
        separate_group = ''
        with open(abspath(join(rec_dir, 'align.com')),'r') as f1:
            for line in f1:
                try:
                    if line.split()[0] == ('SeparateGroup'):
                        separate_group = (line.split()[1])  
                    elif line.split()[0] == ('AxisZShift'):
                        axiszshift = (line.split()[1])  
                except:
                    #meant to catch white spaces
                    pass
    return (st, xf, ali, tlt, thickness, global_xtilt, localxf, excludelist,
            OFFSET, SHIFT, separate_group, axiszshift, zfac, xfile)

def verify_inputs(rec_dir, base_name, out_dir, defocus_file, 
                  input_binning = False, zero_tlt = False):
    """
    
    """  
    #check reconstruction dir
    if not isdir(rec_dir):
        raise Exception('Specified reconstruction folder does not exist.  %s '\
                        % rec_dir)     
    if realpath(rec_dir) == realpath(out_dir):
        raise Exception("Trust me, you don't want the output directory to be your original reconstruction directory...")
    if not isdir(out_dir):
        os.makedirs(out_dir) 
    tomo = join(rec_dir, base_name + '.rec')
    full_tomo = join(rec_dir, base_name + '_full.rec')
    if not isfile(tomo):
        raise Exception("Original tomogram not found. %s" % tomo)
    if not isfile(full_tomo):
        raise Exception("Original unrotated tomogram not found. %s" 
                        % full_tomo)
    
    (st, xf, ali, tlt, thickness, global_xtilt, localxf, excludelist,
     OFFSET, SHIFT, separate_group, axiszshift, zfac, xfile
     ) = read_comfiles(rec_dir)
    #check for x tilt
    if not isfile(xf):
        raise Exception("File not found. %s" % xf) 
    if not isfile(join(out_dir, base_name + '.xf')):
        os.symlink(xf, join(out_dir, base_name + '.xf'))        
    #check_for unaligned stack file
    if not isfile(st):
        raise Exception("File not found. %s" % st)   
    if not isfile(ali):
        raise Exception("File not found. %s" % ali)
    ali_apix, ali_size = get_apix_and_size(ali)
    #check .tlt file
    if not isfile(tlt):
        raise Exception("File not found. %s" % tlt)
    else:
        tilt_angles = [float(x.strip()) for x in open(tlt, 'r')]
        if len(tilt_angles) != ali_size[2]:     
            raise ValueError(('The number of tilt angles in tilt file (%s)'+
                              'does not match the number of images in' +
                              ' aligned stack (%s).')
                             % (len(tilt_angles), ali_size[2]))   

    #os.symlink(tlt, join(out_dir, split(tlt)[1]))    
    #check defocus file
    if not defocus_file:
        defocus_file = realpath(join(rec_dir, str(base_name) + '.defocus'))
    if not isfile(defocus_file):
        raise Exception("File not found. %s" % defocus_file)
#
    #VP change: using get_origin function instead of MapParser
    #unused atm?????
    #origin = get_origin(tomo)


    #getting sizes and psizes             
    stack_apix, stack_size = get_apix_and_size(st)
    tomo_apix, tomo_size = get_apix_and_size(tomo)
    full_tomo_apix, full_tomo_size = get_apix_and_size(full_tomo)
    if tomo_apix != full_tomo_apix:
        raise ValueError('The original unrotated tomogram (_full.rec)' +
                         'pixel size does not match the rotated tomogram.')
    apix = tomo_apix

    #If tomo_binning != ali_binning --> create a binned aligned stack
    if not input_binning:
        tomo_binning = np.round(tomo_apix/stack_apix, decimals = 1)   
        ali_binning = np.round(ali_apix/stack_apix, decimals = 1)   
        print('Tomogram binning = %s' % tomo_binning)
        print('Aligned stack binning = %s' % ali_binning)      
    else:
        tomo_binning = input_binning
    
    if isinstance(zero_tlt, bool):
        zero_tlt = find_nearest(tilt_angles, 0)
        #zero_tlt needs to be adjusted for excludelist. It's not used
        #in flexo_model_from..., only in just_flexo where it needs to
        #correspond match alis, etc where excludelist tilts were removed 
        if not isinstance(excludelist, bool): 
            #just count the number of excludelist entirest that are smaller
            #than zero_tlt, then subtract from zero_tlt
            #excludelist is numbered from 1
            relevant_excludelist = [x for x in excludelist if x <= zero_tlt + 1]
            zero_tlt -= len(relevant_excludelist) 
            zero_tlt += 1 #numbered from 1!
            
    return (st, tomo, full_tomo, tomo_size, ali, tlt, apix, tomo_binning,
            defocus_file, 
            thickness, global_xtilt, excludelist, axiszshift, separate_group,
            zero_tlt, xf, SHIFT, OFFSET, localxf, zfac, xfile)


#########################################################################################

def rotate_model(tomo, full_tomo, ali, base_name,
                 model_file, csv, rec_dir, out_dir, tlt,
                 tomo_size, model_file_binning, skelly_model = False,
                 add_tilt_params = 'skip'):
    """
    model_file_binning is the binning of model file relative to tomogram.
    I.e. a model that's twice the size of specified tomogram is bin 0.5.
    """
    
    print('modfile %s' % model_file)
    print(csv)
    
    #model for particle extraction
    #modpath, modname = split(model_file)
    reprojected_mod = join(out_dir, 'reprojected_fid.mod')
#    binmod = join(out_dir, 'bin_' + modname)
    binmod = join(out_dir, 'bin_%s.mod' % base_name)

    #read model file and apply offsets from csv
    m = np.array(PEETmodel(model_file).get_all_points())
    
    try:
        open(csv).readlines()
        #mod is binary so this should fail
    except:
        raise Exception('Invalid PEET csv file. %s' % csv)
    motl = PEETMotiveList(csv)

    offsets = motl.get_all_offsets()
    m += offsets
    tmp_mod = join(out_dir, 'tmp.mod')
    if float(model_file_binning) != 1.:
        print('Model file binning %s' % model_file_binning)
        m = m * model_file_binning

    mout = PEETmodel()
    for p in range(len(m)):
        mout.add_point(0, 0, m[p])
    mout.write_model(binmod)
    model_file = binmod
    
# =============================================================================
#     if m.max() < max(tomo_size)/2:
#         print 'WARNING! Max coordinate of rotate_model output tmp.mod if less than half the size of input tomogram!'
#         print 'WARNING! rotate_model output tmp.mod is arbitrarily scaled by 2 to make it work for model_file_binning 0.25 and  tomo_binning 0.5'
#         m = m * 2
#         mout = PEETmodel()
#         for p in range(len(m)):
#             mout.add_point(0, 0, m[p])
#         mout.write_model(binmod)
#         model_file = binmod
# =============================================================================
      
    if skelly_model:
        full_tomo = skelly_model

    check_output('imodtrans -I %s %s %s' % (
            tomo, binmod, tmp_mod), shell = True)
    check_output('imodtrans -i %s %s %s' % (
            full_tomo, tmp_mod, tmp_mod), shell = True)

    tilt_str = [('tilt -inp %s -ou %s -TILTFILE %s -THICKNESS %s' +
                ' -ProjectModel %s') % (ali, reprojected_mod,
                                       tlt, tomo_size[2], tmp_mod)
                ]
    #add_tilt_params = 'skip'
    #warnings.warn('add tilt params disabled')
    
    if np.any(add_tilt_params != 'skip'):
        for y in range(len(add_tilt_params)):
            tilt_str.append(add_tilt_params[y])     
    check_output(('').join(tilt_str), shell = True)
    print(('').join(tilt_str))
    rawmod = PEETmodel(reprojected_mod).get_all_points()
    max_tilt = int(np.round(max(rawmod[:,2])+1, decimals = 0))
    rsorted_pcls = np.array([np.array(rawmod[x::max_tilt])
                                for x in range(max_tilt)])


    return rsorted_pcls, reprojected_mod

###############################################################################

def replace_pcles(average_map, tomo_size, csv_file, mod_file, outfile, apix,
                  group_mask = False, extra_bin = False, 
                  average_volume_binning = 1, rotx = False):
    """
    If rotx == True, will output volume in "full.rec" orientation, i.e. XZ
    Due to the lack of header information, imod models that ware saved on
    existing volumes (e.g. original tomo) will not match the unrotated 
    (rotx = False) unless imodtrans -I [plotback] or model2point > point2model
    is used.
    
    Either way the model file supplied should be in XY orientation
    """
    
    
    if isinstance(average_map, str):
        ave = MapParser_f32_new.MapParser.readMRC(average_map)
    else:
        ave = average_map #intended for parallelisation

        
    xsize = ave.x_size()
    ysize = ave.y_size()
    zsize = ave.z_size()  
    if average_volume_binning != 1:
        xsize = xsize * average_volume_binning
        ysize = ysize * average_volume_binning
        zsize = zsize * average_volume_binning
    
    if type(csv_file) == str:
        motl = PEETMotiveList(csv_file)
    else:
        motl = csv_file
    mat_list = motl.angles_to_rot_matrix()
    offsets = motl.get_all_offsets()
    #mod = read_mod_file(mod_file)
    if type(mod_file) == str:
        mod = PEETmodel(mod_file).get_all_points()
    else:
        mod = mod_file
    
#    print offsets, mod
    if np.any(group_mask):
        mat_list = np.array(mat_list)[group_mask]
        offsets = np.array(offsets)[group_mask]
        mod = (np.array(mod)[group_mask]).squeeze()
    if extra_bin:
        offsets = np.array(offsets) * extra_bin
        mod = (np.array(mod) * extra_bin).squeeze()
        

    tomo = Map(np.zeros(np.flip(tomo_size, 0), dtype='float32'),[0,0,0],\
               apix,'replace_pcles') #replaced ave.apix
#    tomo_size.reverse()
#    tomo = Map(np.zeros(tomo_size, dtype='float32'),[0,0,0],\
#               apix,'replace_pcles') #replaced ave.apix
    
    
    #tomo_size.reverse()
    if mod.max() > np.array(tomo_size).max():
        print('Maximum model coordinates exceed volume size. %s %s'\
        % (mod.max(),  np.array(tomo_size).max()))
    if mod.ndim == 1:
        mod = mod[None]
#    print offsets, mod
    for p in range(len(mod)):
        x_pos = int(round(offsets[p][0] + mod[p][0]))
        y_pos = int(round(offsets[p][1] + mod[p][1]))
        z_pos = int(round(offsets[p][2] + mod[p][2]))
#        print x_pos, y_pos, z_pos
        x_offset = offsets[p][0] + mod[p][0] - x_pos
        y_offset = offsets[p][1] + mod[p][1] - y_pos
        z_offset = offsets[p][2] + mod[p][2] - z_pos      
#        print x_offset,y_offset, z_offset
        new_ave = ave.copy()
        shifted_ave = new_ave.rotate_by_matrix(mat_list[p],
                                               ave.centre(), cval = 0)
          
        if average_volume_binning == 1:  
            shifted_ave.fullMap = shift(shifted_ave.fullMap, 
                                        (z_offset, y_offset, x_offset))
        else:     
            z_offset /= average_volume_binning
            y_offset /= average_volume_binning
            x_offset /= average_volume_binning
            shifted_ave.fullMap = shift(shifted_ave.fullMap, 
                                        (z_offset, y_offset, x_offset))  
            shifted_ave.fullMap = zoom(shifted_ave.fullMap,
                                       average_volume_binning)  



        x_d = xsize % 2
        y_d = ysize % 2
        z_d = zsize % 2
        
        x_p_min = np.math.floor(max(0, x_pos - xsize / 2))
        x_p_max = np.math.ceil(min(tomo_size[0], x_d + x_pos + xsize / 2))
        y_p_min = np.math.floor(max(0, y_pos - ysize / 2))
        y_p_max = np.math.ceil(min(tomo_size[1], y_d + y_pos + ysize / 2))
        z_p_min = np.math.floor(max(0, z_pos - zsize / 2))
        z_p_max = np.math.ceil(min(tomo_size[2], z_d + z_pos + zsize / 2))

        x_n_min, y_n_min, z_n_min = 0, 0, 0
        x_n_max, y_n_max, z_n_max = (xsize, ysize, zsize)
        
        if x_p_min == 0:
            x_n_min = np.math.floor(xsize / 2 - x_pos)
        if y_p_min == 0:
            y_n_min = np.math.floor(ysize / 2 - y_pos)
        if z_p_min == 0:
            z_n_min = np.math.floor(zsize / 2 - z_pos)

        if x_p_max == tomo_size[0]:
            x_n_max = np.math.ceil(tomo_size[0] - (x_pos - xsize / 2))
        if y_p_max == tomo_size[1]:
            y_n_max = np.math.ceil(tomo_size[1] - (y_pos - ysize / 2))
        if z_p_max == tomo_size[2]:
            z_n_max = np.math.ceil(tomo_size[2] - (z_pos - zsize / 2))

        x_p_min = int(x_p_min)
        x_p_max = int(x_p_max)
        y_p_min = int(y_p_min)
        y_p_max = int(y_p_max)
        z_p_min = int(z_p_min)
        z_p_max = int(z_p_max)
        
        x_n_min = int(x_n_min)
        x_n_max = int(x_n_max)
        y_n_min = int(y_n_min)
        y_n_max = int(y_n_max)
        z_n_min = int(z_n_min)
        z_n_max = int(z_n_max)
        
        try:
            tomo.fullMap[z_p_min:z_p_max, y_p_min:y_p_max, x_p_min:x_p_max]\
        += shifted_ave.fullMap[z_n_min:z_n_max, y_n_min:y_n_max, x_n_min:x_n_max]
        except:
            raise ValueError('Particle model coordinates are outside specified region bounds.  Please make sure input 3D model fits the input tomogram.')
    if rotx:
        tomo.fullMap = np.rot90(-tomo.fullMap, k = 3)
    else:
        tomo.fullMap = -tomo.fullMap
    if outfile:
        print('Writing MRC file %s' % (outfile))
    
        write_mrc(outfile, tomo.fullMap)

#########################################################################################

def make_and_reproject_plotback(base_name, out_dir, average_volume, tomo_size,
                                csv, model_file, apix, model_file_binning,
                                tomo_binning, tlt, ali, 
                                average_volume_binning,
                                existing_plotback = False,
                                non_default_ali_path = False,
                                add_tilt_params = False,
                                lamella_mask = False):

    """
    Backplots average volume into an empty tomogra, then reprojects into tilt series.
    Uses existing_plotback instead of making a new one if specified.  
    existing_plotback also requires non_default_ali_path to be specified.

    """
    if not existing_plotback:
        plotback_path = join(out_dir, 'plotback.mrc')
        plotback_ali_path = join(out_dir, 'plotback.ali')
        replace_pcles(average_volume,
                      tomo_size,
                      csv,
                      model_file,
                      plotback_path,
                      apix = apix,
                      group_mask = False,
                      extra_bin = model_file_binning,
                      average_volume_binning = average_volume_binning,
                      rotx = True)
    else:
        plotback_path = existing_plotback
        plotback_ali_path = non_default_ali_path
    
    if lamella_mask:
        mask_and_reproject(out_dir, base_name, lamella_mask, plotback_path,
                           plotback_path, plotback_ali_path, ali, tlt, 
                           tomo_size[2], add_tilt_params)
    else:
        reproject_volume(plotback_path, ali, tlt, tomo_size[2],
                         plotback_ali_path, add_tilt_params)
    check_output('alterheader -d %s,%s,%s %s' %
        (apix, apix, apix, plotback_path), shell = True)
    check_output('alterheader -d %s,%s,%s %s' %
        (apix, apix, apix/float(tomo_binning), plotback_ali_path), shell = True)
    return plotback_path, plotback_ali_path
    

def lame_mask(surface_model, tomo_size, out_mask = False, plot_planes = False,
              rotx = False):
    """
    Makes a mask that chops of, e.g. crap on the surface of a lamella.
    surface_model [str] path to MODIFIED tomopitch.mod that fits tomo_full.rec
                    This can be any model with points (regardless of contour
                    clustering), but the volume in the tomogram does need to
                    sit quite flat (which it should anyway).  This is because
                    the 2 surfaces are identified by the Kmeans of the point Z
                    coordinates.
    tomo_size [list of 3 ints] - XYZ (of tomo.rec).
    rtox [bool] has tomo been rotated around x?
    Returns a masked tomogram with the same orientation as tomo_full.rec
    """
    def fit_pts_to_plane(voxels):  
        #https://math.stackexchange.com/questions/99299
        #/best-fitting-plane-given-a-set-of-points
        xy1 = (np.concatenate([voxels[:, :-1],
               np.ones((voxels.shape[0], 1))], axis=1))
        z = voxels[:, -1].reshape(-1, 1)
        fit = np.matmul(np.matmul(
                np.linalg.inv(np.matmul(xy1.T, xy1)), xy1.T), z)
        errors = z - np.matmul(xy1, fit)
        residual = np.linalg.norm(errors)
        return fit, residual
    
    def def_plane(fit, tomo_size):
        X, Y = np.meshgrid(
                np.arange(0, tomo_size[0]), np.arange(0, tomo_size[1]))
        Z = fit[0]*X + fit[1]*Y + fit[2]
        return X, Y, Z
    
    def rmodel(pts):
        r = np.zeros(pts.shape)
        r[:, 0], r[:, 1], r[:, 2] = pts[:, 0], pts[:, 2], pts[:, 1]
        return r
    
#    def get_plane(voxels, tomo_size):
#        """
#        voxels: x y z  (m x 3)
#        tomo_size: x y z
#        """
#        return X, Y, Z

    allpts = PEETmodel(surface_model).get_all_points()
    if not rotx:
        allpts = rmodel(allpts)
    
    #sort Z points into 2 surfaces
    km = KMeans(2).fit(allpts[:,2:3])
    tree = KDTree(km.cluster_centers_)
    dst, pos = tree.query(allpts[:,2:3])
    upper = allpts[pos == 0]
    lower = allpts[pos == 1]
    if np.median(upper[:,2:3]) < np.median(lower[:,2:3]):
        upper, lower = lower, upper
        
    fit, _ = fit_pts_to_plane(upper)
    X1, Y1, Z1 = def_plane(fit, tomo_size)
    fit, _ = fit_pts_to_plane(lower)
    X2, Y2, Z2 = def_plane(fit, tomo_size)
        
    if plot_planes:
        #from mpl_toolkits.mplot3d import Axes3D #unused?
        plt.subplot(111, projection = '3d')
        ax = plt.subplot(111, projection = '3d')
        ax.scatter(allpts[:,0], allpts[:,1], allpts[:,2])
        ax.plot_wireframe(X1, Y1, Z1)
        ax.plot_wireframe(X2, Y2, Z2, color = 'k')
    mask = (np.ones((tomo_size[1], tomo_size[0], tomo_size[2]))[:,:]
            *list(range(tomo_size[2])))
    mask = np.logical_and(mask < Z1[:,:,None], mask > Z2[:,:,None])
    mask = np.swapaxes(mask, 1, 2)*1
    if rotx:
        mask = np.rot90(mask, k = -1, axes = (0,1))
#    masked = np.where(np.swapaxes(mask, 1, 2), v, 0)
    if out_mask:
        write_mrc(out_mask, mask)
    else:
        return mask


def NEW_mask_from_plotback(volume, out_mask, grey_dilation_level = 5,
                           lamella_model = False,
                           lamella_mask_path = False,
                           out_smooth_mask = False):
    """Creates a mask out of a backplotted volume.  Curently used
    only by make_nonoverlapping_alis function.
    
    {grey_dilation_level} - integer, negative values switch to erosion
    
    Input can be path [string] or np.array """
    gdl = grey_dilation_level  
    if type(volume) == str:
        m = deepcopy(mrcfile.open(volume).data)
    else:
        m = volume
    
    #threshold in a sensible? way.  
    #First flatten volume into an image to get thr
    shortest_axis = np.where(np.array(m.shape) == min(m.shape))[0][0]
    thr = threshold_yen(np.mean(m, axis = shortest_axis))
    m = m < thr

    
    
    #m = m**2 > 1E-3
#        if gopen:
#            from scipy.ndimage.morphology import grey_opening
#            m = grey_opening(m, gdl)
    #negative values ==> erosion
    if gdl < 0:
        gdl = -gdl
        m = grey_erosion(m, gdl)
    else:
        m = grey_dilation(m, gdl)
    write_mrc(out_mask, m)
    
    #apply smooth edge, hardcoded for now
    if not out_smooth_mask:
        bb, out_smooth_mask = split(out_mask)
        out_smooth_mask = join(bb, 'smooth_' + out_smooth_mask)
    print('applied mtffilter -3 -l 0.001,0.03 to smooth mask')
    check_output('mtffilter -3 -l 0.001,0.03 %s %s' %
                 (out_mask, out_smooth_mask), shell = True)   
    
    if lamella_model:
        #this will break if the volume is not in "full.rec" orientation
        tomo_size = [m.shape[2], m.shape[0], m.shape[1]]
        bb, aa = split(out_mask)
        lamella_only_mask = join(bb, 'lamella_' + aa)
        lame_mask(lamella_model, tomo_size, lamella_only_mask)
        check_output('clip multiply %s %s %s' % (out_smooth_mask,
                    lamella_only_mask, lamella_mask_path), shell = True)    
        return lamella_mask_path
    else:
        return out_smooth_mask

    
##############################################################################

def make_non_overlapping_pcl_models(sorted_pcls, box_size, out_dir,
                                    threshold = 0.01):
    """ Generates non-overlapping particle models i.e. particle model points 
    that come into proximity (defined by box size) at any view of the tilt
    series are split into independent models.
    Inputs:
    sorted_pcls [numpy array] 
        [number of tilts:model points per tilt:xcoords, ycoords, tilt number]
    box_size [int] or [tuple or list of two ints]
        Determines maximum allowed point-point distance.
        Will be forced into a square.
    out_dir [str] path to write model files
    threshold [float] 
        can be used to get rid of models that have very few points in them.
        E.g. 0.01 removes models that have less than 1% of total particles.
    Outputs:
        outmods [list] list of paths for each non-overlapping model
        novlp_sorted_pcls [list of numpy arrays]
            sorted_pcls style array of points belonging to each group
        pcl_ids [list of lists] 
            (len = number of groups) of indices belonging to each group
        groups [list of masks] 
        remainder [list] leftover particle indices 
        ssorted_pcls [numpy array]
            [number of tilts:model points per tilt:
            xcoords, ycoords, tilt number, particle index, group id (from 0))]
            group id == len(groups) indicates unallocated pcls 
    """

    #box size can be 1 int, a tuple or a list
    if isinstance(box_size, int):
        box_size = [box_size, box_size]
    #force square box
    if box_size[0] != box_size[1]:
        box_size = [max(box_size), max(box_size)]    

    # model point ordinalsf
    indices = np.arange(sorted_pcls.shape[1])

    ssorted_pcls = np.dstack((sorted_pcls,
                              np.zeros((sorted_pcls.shape[0],
                                        sorted_pcls.shape[1], 2))))  
    ssorted_pcls[:,:,3] =  list(range(sorted_pcls.shape[1]))    
    #[number of tilts:model points per tilt:
    # xcoords, ycoords, tilt number, particle index, group id (from 0))]


    #for each tilt, for each particle, generate a list of particles 
    #(and distances) that are within {box_size}
    pvp = np.zeros((sorted_pcls.shape[0],
                    sorted_pcls.shape[1],
                    sorted_pcls.shape[1], 2))
    for x in range(sorted_pcls.shape[0]): 
        tree = KDTree(ssorted_pcls[x,:,:2])
        #return only neighbours within box_size of each point
        dst, pos = tree.query(ssorted_pcls[x,:,:2],
                              ssorted_pcls.shape[1],
                              distance_upper_bound = box_size[0])
        pvp[x,:,:,0] = dst
        pvp[x,:,:,1] = pos

    #reduce to a boolean mask
    #for each particle, particles (index = particle number) that are far enough
    #are set to 1
    mm = np.zeros((sorted_pcls.shape[1], sorted_pcls.shape[1]))    
    for x in range(pvp.shape[1]):
        tmp = np.zeros((pvp.shape[0], pvp.shape[1]))
        #this has to be done for every tilt
        for y in range(pvp.shape[0]):        
            tmp[y] = np.isin(indices, pvp[y,x,:,1], invert = True)
        mm[x] = np.all(tmp, axis = 0)
    for x in range(mm.shape[0]):
        #ignore particle distance to itself
        mm[x,x] = 1


    groups = []
    #start allocating particle indices into groups
    x = 0
    while x < len(indices):
        #len(indices) + 1 is essentially an impossibly large index that 
        #is used to filter out particle indices that have been allocated
        #to a group
        #on the first pass, tmp_indices includes all particle indices
        tmp_indices = indices[indices != len(indices) + 1]
        y = 0
        
        #Extract boolian mask of "sufficiently distant particles" (a)
        #of the first available particle.
        #Do logical AND with the mask of the next available aprticle (--> tmpa).
        #If tmpa[1] == True: a = tmpa (i.e. the masks are combined).
        #This continues untill all indices are exhausted
        #--> first group is complete.
        a = np.array(mm[tmp_indices][0], dtype = 'bool')#mm[x,x] is always True      
        while tmp_indices[y] < tmp_indices[-1]: #set to -1 from -2 16/4/20
            y += 1
            tmpa = np.logical_and(a, mm[tmp_indices[y]])
            if  tmpa[tmp_indices[y]]:
                a = tmpa
        
        if x != 0:
            #Check for duplicates on all subsequent passes after the first
            #and remove them
            test = np.array(groups).sum(axis = 0) + a  > 1
            if np.any(test):
                mmm = np.where(test == True)[0]
                a[mmm] = False  
        groups.append(a)
        
        #indices that were allocated to a group are replaced with a value
        #larger than the largest existing index --> they will not be included
        #in the list of available indices for the next pass
        indices[a] = len(indices) + 1
        
        #sum the remaining particle indices whose value is not 
        #{len(indices) + 1}
        #if there is just one remaining index, make x = len(indices)
        #this will exit the main loop
        if (indices < len(indices) + 1).sum() < 2:
            #add the last group (with a single particle)
            #it will be removed if it's below threshold
            groups.append(np.logical_not(np.any(groups, axis = 0)))
            x = len(indices)
        else:
            #otherwise, set x to the smallest unallocated index
            x = indices[indices != len(indices) + 1][0]            
    groups = np.array(groups)

    #filter out small groups    
    toosmall = groups.sum(axis = 1)/float(mm.shape[0]) < threshold
    if np.any(toosmall):
        print(('%s group(s) had less than %s %% the total number of '
               +'particles and were removed.') % (sum(toosmall), threshold*100))
        indices = np.arange(sorted_pcls.shape[1])
        remainder = indices[np.sum(groups[toosmall], axis = 0, dtype = bool)]
        #set unallocated particle group id to len(groups) + 1
        ssorted_pcls[0, remainder, 4] = len(groups) + 1
        groups = groups[np.logical_not(toosmall)]
        #combined filtered masks and apply to list of indices
        #--> unused indices
    else:
        remainder = False
    print('Number of non-overlapping models ' + str(len(groups)))
    
    pcl_ids = []
    outmods = []
    for x in range(len(groups)):
        #list (len = number of groups) of indices belonging to each group
        pcl_ids.append(np.array(ssorted_pcls[0, groups[x], 3], dtype = 'int32'))
        #add group identifier to ssorted_pcls, numbered from zero
        ssorted_pcls[0, groups[x], 4] = x  
        #write peet models for each group
        out_name = join(out_dir, 'model_group_%02d.mod' % (x))
        outmods.append(out_name)
        outmod = PEETmodel()
        g = np.swapaxes(sorted_pcls[:, groups[x]], 1, 0)
        for p in range(len(g)):
            if p != 0:
                    outmod.add_contour(0)
            for r in range(g.shape[1]):
                outmod.add_point(0, p, g[p,r])
        outmod.write_model(out_name)   
        
    #remaining particles set to len(groups) + 1
    ssorted_pcls[0, np.logical_not(np.any(groups, axis = 0)), 4] = len(groups) + 1
    
    #generate "sorted_pcls style" array for each group
    novlp_sorted_pcls = []
    for g in range(len(groups)):
        rawmod = PEETmodel(outmods[g]).get_all_points()
        max_tilt = int(np.round(max(rawmod[:,2])+1, decimals = 0))
        novlp_sorted_pcls.append(np.array([np.array(rawmod[h::max_tilt])\
                                    for h in range(max_tilt)]))
        
    #outmods probably don't need to be written out - they're not used for
    #anything, maybe nice as sanity check?
    
    return outmods, novlp_sorted_pcls, pcl_ids, groups, remainder, ssorted_pcls


###############################################################################


def mult_mask(lamella_mask_path, out_mask):
    """Used only within make_nonoverlapping_alis, but need to sit outside
    function for joblib to run
    """
    check_output('clip multiply %s %s %s' % (
        lamella_mask_path, out_mask, out_mask), shell = True)

def reproject_volume(tomo, ali, tlt, thickness, out_tomo,
                     add_tilt_params = False):
    str_tilt_angles = [str(x.strip('\n\r').strip(' ')) for x in open(tlt)]
    tilt_str = [
                ('tilt -REPROJECT %s -recfile %s -inp %s -TILTFILE %s' + 
                ' -THICKNESS %s -output %s') % ((',').join(str_tilt_angles),
                                        tomo, ali, tlt, thickness, out_tomo)
                ]
    print(tilt_str)                                            
    if not isinstance(add_tilt_params, bool):
        for y in range(len(add_tilt_params)):
            tilt_str.append(add_tilt_params[y])
#    print ('').join(tilt_str)
    check_output(('').join(tilt_str), shell = True)

def mask_and_reproject(out_dir, base_name, mask, tomo, masked_tomo, out_ali,
                       ali, tlt, thickness, var_str, skip_mask = False):
    """
    """
    if not skip_mask:
        check_output('clip multiply %s %s %s' % 
                (mask, tomo, masked_tomo), shell = True)        
    reproject_volume(masked_tomo, ali, tlt, thickness,
                     out_ali, var_str) 



def format_nonoverlapping_alis(out_dir, base_name, average_map, tomo, ali, tlt,
                             pmodel, csv_file, apix, rsorted_pcls, tomo_size,
                             var_str, box_size,
                             machines, pmodel_bin = False,
                             grey_dilation_level = 5, 
                             average_volume_binning = 1,
                             lamella_mask_path = False,
                             threshold = 0.01,
                             use_init_ali = False):


    """
    
    Inputs:
        out_dir [str]
        
        average_map [str] - average volume used for plotback, white on black
                            (generally chimera segmented volume)
        
        apix [float]
        
        pmodel [str] path to imod 3d model - has to be identical to model used
        to generate rsorted_pcls!!!!
        
        rsorted_pcls [np.ndarray] particle coordinates in a very specific format....
    Output:
        plotback - oriented edgeon (i.e. same orientation as IMOD _full.rec)
        
        
        """
        
    average_apix, average_size = get_apix_and_size(average_map)
    if average_volume_binning != 1:
        average_apix /= average_volume_binning
        average_size = np.array(np.array(average_size, dtype = 'float16')
        * average_volume_binning, dtype = 'int16')
    if np.round(average_apix, decimals = 1) != np.round(apix, decimals = 1):
        print(('WARNING:\n Average map pixel size does not match input ' + 
                ' tomogram pixel size: %s %s!' % (average_apix, apix)))
    print('Using average map size to generate non overlapping models.')
    

    #generate non-overlapping models    
    (outmods, novlp_sorted_pcls, pcl_ids, groups, remainder, ssorted_pcls
     ) = make_non_overlapping_pcl_models(
                                        rsorted_pcls,
                                        box_size,
                                        out_dir,
                                        threshold = threshold
                                        ) 
    #remainder (i.e. unallocated particle ordinals) currently doesn't do anything
    #names of input and output files
    plotback_list = []
    sub_ali_list = []
    masked_tomo_list = []
    out_mask_list = []
    smooth_mask_list = []
    plotback_ali_list = []
    for x in range(len(groups)):
        plotback_list.append(join(out_dir, 'plotback_%02d.mrc' % x))
        out_mask_list.append(join(out_dir, 'mask_%02d.mrc' % x))
        smooth_mask_list.append(join(out_dir, 'smooth_mask_%02d.mrc' % x))
        if use_init_ali:
            sub_ali_list.append(ali)
        else:
            sub_ali_list.append(join(out_dir, 'subtracted_%02d.mrc' % x))
        masked_tomo_list.append(join(out_dir, 'masked_%02d.mrc' % x))
        plotback_ali_list.append(join(out_dir, 'plotback_%02d.ali' % x))
    ssorted_path = join(out_dir, base_name + '_ssorted.npy')
    group_path = join(out_dir, base_name + '_groups.npy')
    nvlp_path = join(out_dir, base_name + '_nvlp.npy')

    group_ids = [str(x) for x in range(len(groups))] 
    #combine paths into a single array of length = len(groups)
    #(group_ids[0], outmods[1], plotback_list[2], sub_ali_list[3],
    #masked_tomo_list[4], out_mask_list[5], smooth_mask_list[6],
    #plotback_ali_list[7])
    nvlp_files = np.vstack((group_ids, outmods, plotback_list, sub_ali_list,
        masked_tomo_list, out_mask_list, smooth_mask_list, plotback_ali_list))

    np.save(ssorted_path, ssorted_pcls)
    np.save(group_path, groups)
    np.save(nvlp_path, nvlp_files)

    #divide groups among available machines
    #probably best to do one tomo/chunk
    #per_core = max(1, int(np.round(len(groups)/float(len(machines)))))
    per_core = 1
    #make a list of files for each core
    c_tasks = []
    for x in range(0, len(groups), per_core):
        tmp = nvlp_files[:, x:x + per_core]
        if per_core == 1:
        #output is written out as comma separated strings.  Have to add an
        #empty tuple in case there is only one entry per core
            tmp1 = []
            for y in range(len(tmp)):
                tmp1.append((tmp[y], ()))
            tmp1 = np.array(tmp1 ,dtype = object)
        c_tasks.append(tmp)
        
    #these variables have to be dealt with separately in case they are boolian
    ###
    if lamella_mask_path:
        lmp = '>lamella_mask_path = "%s"' % lamella_mask_path
    else:
        lmp = '>lamella_mask_path = False'  
    if not isinstance(var_str, bool):
        vs = '>var_str = [%s]' % (',').join(
                ['"' + str(s) + '"' for s in var_str])
    else:
        vs = '>var_str = False'
    ###
    
    def c_task_str(c_tasks, corei, index):
        """convert array elements into printable strings """
        return (',').join(['"' + str(y) + '"' for y in c_tasks[corei][index]])

    #this is a standin for adding "flexo installation path" to PATH    
    path = (',').join(['"' + x + '"' for x in sys.path if x != ''])
    for x in range(len(c_tasks)):
        out_s = (
        '>sys.path.extend([%s])' % path,
        '>from definite_functions_for_flexo import chunk_non_overlapping_alis',
        '>average_map = "%s"' % average_map,
        '>csv_file = "%s"' % csv_file,
        '>pmodel = "%s"' % pmodel,
        '>group_path = "%s"' % group_path,
        lmp,
        '>out_dir = "%s"' % out_dir,
        '>base_name = "%s"' % base_name,
        '>ali = "%s"' % ali,
        '>tlt = "%s"' % tlt,
        '>tomo = "%s"' % tomo,
        '>group_ids = [%s]' % (',').join([str(y) for y in c_tasks[x][0]]),
        '>plotback_list = [%s]' %  c_task_str(c_tasks, x, 2),
        '>sub_ali_list = [%s]' %  c_task_str(c_tasks, x, 3),
        '>masked_tomo_list = [%s]' %  c_task_str(c_tasks, x, 4),
        '>out_mask_list = [%s]' %  c_task_str(c_tasks, x, 5),
        '>smooth_mask_list = [%s]' %  c_task_str(c_tasks, x, 6),
        '>plotback_ali_list = [%s]' % c_task_str(c_tasks, x, 7),
        vs,
        '>tomo_size = [%s]' %  (',').join([str(y) for y in tomo_size]),
        '>apix = %s' %  apix,
        '>pmodel_bin = %s' %  pmodel_bin,
        '>average_volume_binning = %s' %  average_volume_binning,
        '>grey_dilation_level = %s' %  grey_dilation_level,
        '>use_init_ali = %s' % use_init_ali,
        '>chunk_non_overlapping_alis(average_map,',
        '>                           csv_file,',
        '>                           pmodel,',
        '>                           group_path,',
        '>                           lamella_mask_path,',
        '>                           out_dir,',
        '>                           base_name,',
        '>                           ali,',
        '>                           tlt,',
        '>                           tomo,',
        '>                           group_ids,',
        '>                           plotback_list, ',
        '>                           sub_ali_list,',
        '>                           masked_tomo_list,',
        '>                           out_mask_list,',
        '>                           smooth_mask_list,',
        '>                           plotback_ali_list,',
        '>                           var_str,',
        '>                           tomo_size,',
        '>                           apix,',
        '>                           pmodel_bin,',
        '>                           average_volume_binning,',
        '>                           grey_dilation_level,',
        '>                           use_init_ali)',
        )
        comfile = join(out_dir, base_name + '-%03d.com' % x)
        with open(comfile, 'w') as f:
            for x in range(len(out_s)):
                f.write(out_s[x] + '\n')
                
    return (group_path, sub_ali_list, plotback_ali_list, ssorted_path,
            plotback_list, out_mask_list, smooth_mask_list)


def chunk_non_overlapping_alis(average_map, csv_file, pmodel, group_path, 
                               lamella_mask_path, out_dir, base_name, ali,
                               tlt, tomo, group_ids, plotback_list, 
                               sub_ali_list, masked_tomo_list, out_mask_list,
                               smooth_mask_list, plotback_ali_list, var_str,
                               tomo_size, apix, pmodel_bin, 
                               average_volume_binning, grey_dilation_level,
                               use_init_ali = False):
    """
    This can handle a single tomo as input (still needs to be a list) or
    multiple inputs.
    """
    groups = np.load(group_path)[group_ids]
    average_map = MapParser_f32_new.MapParser.readMRC(average_map) 
    csv_file = PEETMotiveList(csv_file)

    print('Backplotting.')
    pmodel = PEETmodel(pmodel).get_all_points()    
    #make plotbacks
    #LOOPING IS INTENTIONAL in case there are multiple tomos as input.
    #the shape of groups with a single input is (1, nnn) meaning that the loops
    #are over [0]
    for a in range(len(groups)):
        replace_pcles(average_map, tomo_size, csv_file, pmodel,
                      plotback_list[a], apix, groups[a], pmodel_bin, 
                      average_volume_binning, True)
        
    print('Reprojecting backplots.')
    if lamella_mask_path:
        for a in range(len(groups)):
            mask_and_reproject(out_dir, base_name, lamella_mask_path,
                               plotback_list[a], plotback_list[a],
                               plotback_ali_list[a], ali, tlt, tomo_size[2],
                               var_str)
    else:
        for a in range(len(groups)):
            reproject_volume(plotback_list[a], ali, tlt, tomo_size[2], 
                             plotback_ali_list[a], var_str)    
    if not use_init_ali:
        print('Generating tomogram masks.')
        for a in range(len(groups)):
            NEW_mask_from_plotback(plotback_list[a], out_mask_list[a],
                grey_dilation_level, False, False, smooth_mask_list[a])
    
        if lamella_mask_path:
            #combined masks with an existing lamella mask
            for a in range(len(groups)):
                mult_mask(lamella_mask_path, smooth_mask_list[a])
            
        print('Reprojecting masked tomos.')
        for a in range(len(groups)):
            mask_and_reproject(out_dir, base_name, smooth_mask_list[a], tomo, 
                               masked_tomo_list[a], sub_ali_list[a],
                               ali, tlt, tomo_size[2], var_str)
        
##############################################################################

def pad_pcl(pcl, box_size, sorted_pcl, stack_size):
    """Pads 2D particle coordinates that are outside the 
    stack coordinates with mean (of the particle)."""
    if pcl.shape[1] != (box_size[0]):
        pad = np.abs(pcl.shape[1] - box_size[0])
        if 0 > sorted_pcl[0] - box_size[0]//2:
            pcl = np.pad(pcl, ((0, 0), (pad, 0)), 'constant',\
                         constant_values=(np.mean(pcl), np.mean(pcl)))
        elif stack_size[0] < sorted_pcl[0] + box_size[0]//2:
            pcl = np.pad(pcl,((0, 0), (0, pad)), 'constant',\
                         constant_values=(np.mean(pcl), np.mean(pcl)))
    if pcl.shape[0] != (box_size[1]):
        pad = np.abs(pcl.shape[0] - box_size[1])
        if 0 > sorted_pcl[1] - box_size[1]//2:
            pcl = np.pad(pcl, ((pad, 0), (0, 0)), 'constant',\
                         constant_values=(np.mean(pcl), np.mean(pcl)))
        elif stack_size[1] <  sorted_pcl[1] + box_size[1]//2:
            pcl = np.pad(pcl, ((0, pad), (0, 0)), 'constant',\
                         constant_values=(np.mean(pcl), np.mean(pcl)))
    return pcl

##############################################################################

def extract_2d_simplified(stack, sorted_pcls, box_size, excludelist = False,
                          offsets = False):
    """Simplified version of 2D particle extraction from a 2D stack.
    Input:
        stack [str or numpy array] path of image stack or mrcfile array
        sorted_pcls [2D numpy array] [tilt number: xcoord, ycoord, z]
                    - coordinates of particles to be extracted
        box_size [list of even natural numbers] e.g. [40,38]
        excludelist [list of ints] tilt numbers to be excluded (numbered from 1)

    Returns extracted particles [numpy array [particle, tilt number, [X,Y]]]
        """
    def get_2d_offset(pcl):
        return pcl - round(pcl)
    
    if isinstance(stack, str):
        mrc = deepcopy(mrcfile.open(stack).data)
    else:
        mrc = stack

    if not isinstance(excludelist, bool):
        #etomo comfiles have excludelist numbered from 1. 
        excludelist = np.array(excludelist) - 1
        #remove tilts based on excludelist
        exc_mask = np.isin(list(range(len(mrc))), excludelist, invert = True)
        mrc = mrc[exc_mask]
        sorted_pcls = sorted_pcls[exc_mask]

    stack_size = np.flip(mrc.shape)
    
    if np.squeeze(sorted_pcls).ndim == 2:
        sorted_pcls = np.squeeze(sorted_pcls)
        #i.e. in case there is only one particle
        all_pcls = np.zeros((1, stack_size[2], box_size[1], box_size[0]))  
        all_offsets = np.zeros((1, stack_size[2], 2))

        #expand sorted_pcls to (n, 1, n) array
        s = np.zeros((sorted_pcls.shape[0], 1, sorted_pcls.shape[1]))
        s[:, 0, :] = sorted_pcls
        sorted_pcls = s
        it0 = 1
    else:   
        all_pcls = np.zeros((
                sorted_pcls.shape[1], stack_size[2], box_size[0], box_size[1]))
        it0 = sorted_pcls.shape[1]
        all_offsets = np.zeros((sorted_pcls.shape[1], stack_size[2], 2))
    for x in range(it0):
        for y in range(sorted_pcls.shape[0]):
            x1 = int(round(max(0, sorted_pcls[y, x, 0] - box_size[0]//2)))
            x2 = int(round(min(
                        stack_size[0], sorted_pcls[y, x, 0] + box_size[0]//2)))        
            y1 = int(round(max(0, sorted_pcls[y, x, 1] - box_size[1]//2)))
            y2 = int(round(min(
                        stack_size[1], sorted_pcls[y, x, 1] + box_size[1]//2)))           
            if (np.array([x1,y1,x2,y2]) < 0).any():
                raise ValueError('View %s of particle %s is completely\
                                 outside the image stack. You should rethink\
                                 your life choices...' % (y, x))
            pcl = mrc[y, y1:y2, x1:x2]
            pcl = pad_pcl(pcl, box_size, sorted_pcls[y,x], stack_size)
            all_pcls[x, y] = pcl  

            if offsets:
                dx = get_2d_offset(sorted_pcls[y, x, 0])
                dy = get_2d_offset(sorted_pcls[y, x, 1])  
                # offsets are YX to match format of .xf
                all_offsets[x, y] = np.array([dy, dx])
    if it0 == 1:
        all_pcls = all_pcls.squeeze()
    if offsets:
        # offsets are YX to match format of .xf
        return all_pcls, all_offsets
    else:
        return all_pcls#, all_resid

##############################################################################

def ctf(wl, ampC, Cs, defocus, ps, f):
    """ returns ctf curve """
    a = (-np.sqrt(1 - ampC**2)*np.sin(2*np.pi/wl*(defocus*(f*wl)**2/
        2-Cs*(f*wl)**4/4) + ps) - ampC*np.cos(2*np.pi/wl*
        (defocus*(f*wl)**2/2 - Cs*(f*wl)**4/4) + ps))
    return a

def CTF_convolution(wl, ampC, Cs, defocus, ps, inp, apix):
    """ 300kV wl = 0.0196869700756 """
    (Nx, Ny) = inp.shape
    Dx = 1./(Nx*apix)
    Dy = 1./(Ny*apix)
    FFTimg = fft.fftshift(fft.fft2(inp))
    x = np.arange(-Dx*Nx/2, Dx*Nx/2, Dx) 
    y = np.arange(-Dy*Ny/2, Dy*Ny/2, Dy)
    xx, yy = np.meshgrid(x, y, sparse = True, indexing = 'ij')
    CTF = ctf(wl, ampC, Cs, defocus, ps, np.sqrt(xx**2 + yy**2))
    BPimg = fft.ifft2(fft.ifftshift(FFTimg * CTF)).real* -1.
    return BPimg

def dose_filter(y):
    """Returns optimal 1/spatial frequency for a specified electron dose"""
    y = max(y, 7.03) #truncate negative values
    a = ((0.4*y - 2.81)/0.245)**(1./ - 1.665) #from grant and grigorieff
    a = 1./a
    return a

def butterworth_filter(cutoff, box_size, apix, order = 6,
                       t = 'low', analog=True):
    """Daven
    Generates butter curve"""
    if type(cutoff) ==list:
        cutoff = [1./x for x in cutoff]
    else:
        cutoff = 1./cutoff  
    b, a = butter(order, cutoff, t, analog = analog)
    #VP 20201030 importing butter specifically
    #b, a = signal.butter(order, cutoff, t, analog = analog)
    
    d = dist_from_centre_map(box_size, apix)
    freq = np.unique(d)
#    w, h = signal.freqs(b, a, freq)
    w, h = freqs(b, a, freq)
    #VP 20201030 importing feqs specifically

    f = interp1d(w, abs(h))
    return f(d)

def dist_from_centre_map(box_size, apix): 
    #author: Daven Vasishtan
    #while it may seem possible to do this with sqrt of meshgrid**2, the
    #latter produces an asymmetric array 
    #importantly, the the difference in speed is negligible
    #for a 1000,1000 image they are almost identical, the latter is faster
    #with smaller arrays
    init_dims = []
    for d in range(2):
        if box_size[d]%2 == 1:
            init_dims.append(fftshift(fftfreq(box_size[d], apix))**2)
        else:
            f = fftshift(fftfreq(box_size[d], apix))[:box_size[d]//2]
            f = np.append(f, f[:: -1])* -1
            init_dims.append(f**2)
    return np.sqrt(init_dims[1][:,None] + init_dims[0])

def filter_tilts_butter_p(ali, dose, apix):
    box = ali.shape[1], ali.shape[0]
    lowpass = dose_filter(dose)
    window = butterworth_filter(lowpass, box, apix, order = 4) #lowpass box apix
    FFTimg = fft.fftshift(fft.fft2(ali))
    BPimg = fft.ifft2(fft.ifftshift(FFTimg * window)).real
    return BPimg

def butter_filter(img, lowpass, apix):
    box = img.shape[1], img.shape[0]
    window = butterworth_filter(lowpass, box, apix, order = 4) #lowpass box apix
    FFTimg = fft.fftshift(fft.fft2(img))
    BPimg = fft.ifft2(fft.ifftshift(FFTimg * window)).real
    return BPimg

def dose_list(ali, zerotlt, dose, orderL, dosesym = False,
              pre_exposure = False, return_freq = True):
    """
    Generate list of accummulated electron doses/tilt or the optimal
    filtering frequencies.
    zerotlt [int] numbered from 1
    """
    if not dosesym:
        #from zerotlt to zero
        if orderL:
            doselist = (np.array(list(range(zerotlt - 1, -1, -1))
                            + list(range(zerotlt, len(ali)))) * dose)
        ##from zerotlt to max
        else:
            doselist = np.array(
                    list(range(len(ali) - 1, len(ali) - zerotlt - 2, -1))
                    + list(range(len(ali) - zerotlt - 1)))
    else:
        order = np.zeros((len(ali)), dtype = 'int16')
        order[0] = int(zerotlt - 1)
        for x in range(1, len(ali)):
            n = np.ceil(x/2.)
            if x % 2 != 0:
                order[x] = int(zerotlt - 1 + n * (-1)**n)
            else:
                order[x] = int(zerotlt - 1 - n * (-1)**n)
        doselist = np.arange(0, len(ali) * dose, dose)
    if pre_exposure:
        doselist += pre_exposure 
    if return_freq:
        lowfreqs = [dose_filter(x) for x in doselist]
        return lowfreqs
    else:
        return doselist

def ctf_convolve_andor_dosefilter_wrapper(ali, zerotlt, dose, apix, V = 300000,
  Cs = 27000000, wl = 0.01968697007561453, ampC = 0.07, ps = 0, defocus = 0,
  butter_order = 4, dosesym = False, orderL = True, pre_exposure = 0):
    """  
    Can convolute with CTF and dosefilter (butter) at the same step
    OR
        just convolute with CTF
    OR
        just dosefilter    
    Zero_tilt is the index of the first tilt starting from 1 !!!!
    (i.e. index of tilt with the lowest accummulated dose)
    Goes from zerotlt to tilt 0, then zerotlt to last tilt.
    OR
        Dosesymmetric  
    Constant doserate    
    if orderL == True, tilts are filtered from zerotlt to 0, then zerotlt to max
    
    {zerotlt} - numbered from 1!!
    
    """
    ali = deepcopy(ali)
    
    #make a list of zeros
    if not np.any(defocus):
        defocus = [0 for x in range(len(ali))]
    if dose:
    #i.e. skips dosefiltering if dose == 0/False
        if isinstance(dose, str):
            try:
                dose = float(dose)
            except:
                raise TypeError('Invalid dose value.')
        #generate list of lowpass frequencies       
        lowfreqs = dose_list(ali, zerotlt, dose, orderL, dosesym = dosesym,
                  pre_exposure = pre_exposure, return_freq = True)
        for x in range(len(ali)):   
            ali[x] = ctf_convolve_andor_butter(ali[x], apix, V, Cs, wl,
                       ampC, ps, lowfreqs[x], defocus[x], butter_order)
    else:
    #skip dosefilter
        lowfreqs = 0
        for x in range(len(ali)):      
            ali[x] = ctf_convolve_andor_butter(ali[x], apix, V, Cs, wl,
                           ampC, ps, lowfreqs, defocus[x], butter_order)   
    return ali
    

def ctf_convolve_andor_butter(inp, apix, V = 300000.0, Cs = 27000000.0,
                              wl = 0.01968697007561453, ampC = 0.07, ps = 0,
                              lowpass = 0, defocus = 0, butter_order = 6):
    """Convolute with CTF, phase flip and bandpass at the same time
    (saves ffting back and forth)
    Intended for a single image
    Input:
        inp [np array [xy]]
        apix [float] [angstrom]
        V [int] [volt]
        Cs [float] [angstrom]
        wl [float] [angstrom]
        defocus [int/float] [angstrom]
        ps - lets not bother with that... 0
        lowpass [float] [spatial freq]
    """
    (Nx, Ny) = inp.shape
    CTF = 1
    if defocus:
        Dx = np.float128(1)/np.float128(Nx*apix)
        Dy = np.float128(1)/np.float128(Ny*apix)
        x = np.arange(-Dx * Nx/2, Dx * Nx/2, Dx, dtype = 'float32') 
        y = np.arange(-Dy * Ny/2, Dy * Ny/2, Dy, dtype = 'float32')
        xx, yy = np.meshgrid(x, y, sparse = True, indexing = 'ij')
        CTF = ctf(wl, ampC, Cs, defocus, ps, np.sqrt(xx**2 + yy**2))    
    window = 1
    if lowpass:
        window = butterworth_filter(\
                    lowpass, (Nx, Ny), apix, order = 4)
    FFTimg = fft.fftshift(fft.fft2(inp))
    filtered = fft.ifft2(fft.ifftshift(\
                            ((FFTimg * CTF)) * window)).real
    if defocus:
        filtered = filtered * -1.     
    return filtered

##############################################################################

def convert_ctffind_to_defocus(defocus_file, base_name, rec_dir, out_dir):
    """I followed the instructons for ctfphaseflip where tilts are listed in 
    reverse order."""
    tlt = os.path.join(os.path.realpath(rec_dir), str(base_name) + '.tlt')
    tilt_angles = [float(x.strip()) for x in open(tlt, 'r')]
    tilt_angles.reverse()
    defocus=[]
    defocus2=[]
    angles=[]
    with open(defocus_file,'r')as f:
        for line in f:
            if not line.split(' ')[0]=='#':
                a=float('{0:.2f}'.format(float(line.split(' ')[1])/10))
                defocus.append(a)
                b=float('{0:.2f}'.format(float(line.split(' ')[2])/10))
                defocus2.append(b)
                c=float('{0:.1f}'.format(float(line.split(' ')[3])))
                angles.append(c)  
    defocus.reverse()
    defocus2.reverse()
    angles.reverse()
    order=list(range(1,(len(tilt_angles)+1)))
    order.reverse()
    out_defocus_file = os.path.abspath(os.path.join(out_dir, str(base_name) +
                                                    '.defocus'))
    with open(out_defocus_file,'w') as output_file:
        output_file.write('1  0 0. 0. 0  3\n')
        for i, j, k, l, m in zip(order, tilt_angles, defocus2, 
                                 defocus, angles): #lower defocus value first
            output_file.write(str(i)+'\t'+str(i)+'\t'+str(j)+'\t'+str(j)+
                              '\t'+str(k)+'\t'+str(l)+'\t'+str(m)+'\n')
    return out_defocus_file

###############################################################################
def read_defocus_file(defocus_file, base_name = False, rec_dir = False,
                      out_dir = False):
    """
    reads deofocus file, converts to IMOD defocus 3 format.
    Optional arguments required if input is CTFFind format.
    """
    #First check what type of defocus file it is.  
    #CTFFind is converted to .defocus 3.  
    with open(defocus_file, 'r') as ff:
        first_line = ff.readlines()[0]
    if any([a == 'CTFFind' for a in first_line.split()]):
        if not all((base_name, rec_dir, out_dir)):
            raise Exception('read_defocus_file: optional arguments required'
                            + ' to read CTFFind format.')
        defocus_file = convert_ctffind_to_defocus(defocus_file, base_name,
                                                  rec_dir, out_dir)

    with open(defocus_file, 'r') as f:
        df=[]        
        for g in f.readlines():
            df.append([float(y) for y in g.split()])
    if str(first_line.split()[-1]) == '2':
        df[0] = df[0][0: -1]     
        df = np.array(df)
        df = df[np.argsort(df[:,0])]
        #standardise to astigmatism format by doubling up defocus value and
        #adding zero for ast. angle
        df = np.column_stack((df, df[:,-1], np.zeros(np.shape(df)[0])))
    else:
        df = np.array(df[1:])
        df = df[np.argsort(df[:,0])]
    return df
    
def get_pcl_defoci(base_name, tlt, rec_dir, out_dir, sorted_pcls, ali,
                   apix, defocus_file, excludelist = False, verbose = True,
                   ):
    """
    defocus in angstroms
    
    using rsorted_pcls instead of sorted_pcls - that is more appropriate
    considering im using apix of the binned tomo
    """
    def remove_exc_entries(df, excludelist):
        #leaving excludelist numbered from one, same as defocus file
        excludelist = np.array(excludelist)
        df = np.array(df)
        exc_mask = np.isin(df[:,0], excludelist, invert = True)
        return df[exc_mask]

   
    tilt_angles = [float(x.strip('\n\r').strip(' ')) for x in open(tlt)]
    stack_size = np.array(MapParser_f32_new.MapParser.readMRCHeader(ali)[:3])
    df = read_defocus_file(defocus_file, base_name, rec_dir, out_dir)    
    if not isinstance(excludelist, bool): 
        df = remove_exc_entries(df, excludelist)
    meandf = np.mean((df[:, 4], df[:, 5]), axis = 0) * 10 # mean defocus in angstroms

    if not isinstance(excludelist, bool): 
        #etomo comfiles have excludelist numbered from 1. 
        excludelist = np.array(excludelist) - 1
        #remove tilts based on excludelist
        exc_mask = np.isin(list(range(stack_size[2])), excludelist, invert = True)
        sorted_pcls = sorted_pcls[exc_mask]
        tilt_angles = np.array(tilt_angles)[exc_mask]
        
    all_defoci = np.zeros((sorted_pcls.shape[:2]))
    sin_tlt = np.sin(np.radians(np.array(tilt_angles))) #sine of tilt angles
    xpos = np.negative(sorted_pcls[:, :, 0] - stack_size[0]//2) * apix
    
#    print 'testing', sorted_pcls.shape, all_defoci.shape, meandf.shape, xpos.shape
    
    
    for x in range(sorted_pcls.shape[1]):
        #distance from tilt axis at 0 degrees in angstroms 
        dst = (sin_tlt * xpos[:, x])
        all_defoci[:, x] = meandf - dst

    if verbose:
        print('REPLACE BY A PLOT??')
# =============================================================================
#         print 'len(tilt angles) = %s' % len(tilt_angles)
#         print 'model.shape = %s' % str(model.shape)
#         print 'all_defoci.shape = %s' % str(all_defoci.shape)
#         if excludelist:
#             print 'len(orig_defoci) = %s' % len(orig_defoci)
#             print 'len(defoci0) = %s' % len(defoci0)
# =============================================================================
    if out_dir:
        #save pcled array for debugging
        np.save(out_dir + '/all_defoci.npy', np.flip(all_defoci, 0))
    return np.flip(all_defoci, 0)

##############################################################################

def norm(x):
    """normalise to 0 mean and 1 standard deviation"""
    return (x - np.mean(x))/np.std(x)
    
def circle(x, y, radius = 1, col = 'red'):
    """
    annotate plot with a circle
    usage:
    fig, axs = plt.subplots(3, 2)
    axs[0,x].add_artist(circle(pk[0,1], pk[0,0], 4))
    """
    circle = Circle((x, y), radius, clip_on = False, linewidth = 1,
                    edgecolor = col, facecolor = (0, 0, 0, 0))
    return circle


def g2d(centre_x, centre_y, size, centre_bias):
    """generates a aquare 2d gaussian array
        exp(-(x-b)**2/(2c)**2) 
        The peak is centered on the bottom left corner of a pixel.
        
        Inputs:
            centre_x [int/float]
            centre_y [int/float]
            size [int] output array side length
            centre_bias [float] value of peak maximum
        Output:
            [numpy array] square 2d gaussian
    """
    c = centre_bias
    centre_values = (centre_y, centre_x) #Y, X
    shift = centre_values[0] - (size//2), centre_values[1] - (size//2)
    shift = np.array(shift)/(size//2.)
    x = np.linspace(1 + shift[0], -1 + shift[0], size)
    y = np.linspace(1 + shift[1], -1 + shift[1], size)
    xx, yy = np.meshgrid(x, y)
    xy = np.sqrt(xx**2 + yy**2)
    #a =  np.exp(-4*(np.log(2)*xy**2))/fwhm**2
    return  np.exp(-(xy**2)/2*c**2)

def c_mask(size, centre_y, centre_x, distance): 
    """
    Circular mask centered on centre_y, centre_x
    """
    #EDIT VP 11/11/2020 centre_values was unused
    #centre_values = (centre_y, centre_x) #Y, X
    shift = (size//2) - centre_y, (size//2) - centre_x
    x = np.arange(-size//2 + shift[0], size//2 + shift[0], 1)
    y = np.arange(-size//2 + shift[1], size//2 + shift[1], 1)
    xx, yy = np.meshgrid(x,y)
    xy = np.sqrt(xx**2 + yy**2)
    mxy = xy <= distance
    return mxy

def compare_ccs(pcl_n, ref, query, top, bot, shifts, interp, limit,
                output_dir = False, debug = 0, out_name = False):
    """
    The projection of query corrected by shifts should match the 
    average better than the original query.  Returned values > 1 indicate
    improvement.
    Inputs:
        {query} - query stack
        {ref} - ref stack
        {bot, top} - use tilts [bot:top] 
        {shifts} - coordinates from buggy_peaks 
            
    """
    size = interp*limit*2
    peak_tol = interp*2
    med_shift = np.median(shifts[bot:top], axis = 0) 
    #make a mask that's {maximum shift} smaller than the box
    max_shift = np.absolute(np.max(shifts[bot:top]) - size//2)
    new_mask = create_circular_mask(ref.shape[1:], ref.shape[1]//2 - max_shift)
    shifted_query = []
    for x in range(bot, top):
        s = np.divide(shifts[x], float(interp)) - float(limit)
        shifted_query.append(shift(query[x], s))
    #corrected query   
    av_sq = np.median(np.array(shifted_query), axis = 0)
    #initial query
    av_q = np.median(query[bot:top], axis = 0)
    #ref:
    av_r = np.median(ref[bot:top], axis = 0)

##testing#############################################
#    shifted_ref = []
#    for x in range(bot, top):
#        s = np.divide(shifts[x], float(interp)) - float(limit)
#        shifted_ref.append(shift(ref[x], s))
#    av_sr = np.median(np.array(shifted_ref), axis = 0)
#
#    backshifted_query = []
#    for x in range(len(shifted_query)):
#        s = np.divide(shifts[bot + x], float(interp)) - float(limit)
#        backshifted_query.append(shift(shifted_query[x], np.negative(s)))
#    av_bsq = np.median(np.array(backshifted_query), axis = 0)
#
#    testing = '/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/nec_test3/example_particles'
#    
#    write_mrc(join(testing, 'av_sq%02d.mrc' % pcl_n), av_sq)
#    write_mrc(join(testing, 'av_q%02d.mrc' % pcl_n), av_q)
#    write_mrc(join(testing, 'av_r%02d.mrc' % pcl_n), av_r)
#    write_mrc(join(testing, 'av_sr%02d.mrc' % pcl_n), av_sr)
#    #new_mask = 1
#    unshifted_cc = get_cc_map(av_r*new_mask, av_q*new_mask, limit, interp)
#    sunshifted_cc = get_cc_map(av_sr*new_mask, av_sq*new_mask, limit, interp)
#    shifted_cc = get_cc_map(av_r*new_mask, av_sq*new_mask, limit, interp)
#    bsq_cc = get_cc_map(av_q*new_mask, av_bsq*new_mask, limit, interp)
#    bsq_cc2 = get_cc_map(av_r*new_mask, av_bsq*new_mask, limit, interp)
#
#    f,a = plt.subplots(3,3)
#    print np.max(unshifted_cc), np.max(sunshifted_cc), np.max(shifted_cc), np.max(bsq_cc2)
#    print np.max(unshifted_cc), np.max(shifted_cc)
#    print np.max(shifted_cc)/np.max(unshifted_cc), np.min(shifted_cc)/np.min(unshifted_cc)
#    print np.max(shifted_cc)/np.max(sunshifted_cc), np.min(shifted_cc)/np.min(sunshifted_cc)
#    print np.max(shifted_cc)/np.max(bsq_cc2), np.min(shifted_cc)/np.min(bsq_cc2)
#    
#    f.suptitle(('median projections'+
#               ('\nshifted_cc/unshifted_cc %.5f, %.5f (max, min values)' %
#                (np.max(shifted_cc)/np.max(unshifted_cc), np.min(shifted_cc)/np.min(unshifted_cc)))
#                +('\nshifted_cc/shifted_ref_and_query_cc %.5f, %.5f' % 
#                 (np.max(shifted_cc)/np.max(sunshifted_cc), np.min(shifted_cc)/np.min(sunshifted_cc)))
#                 +('\nshifted_cc/shifted_then_backshifted_query_cc %.5f, %.5f ' %
#                  (np.max(shifted_cc)/np.max(bsq_cc2), np.min(shifted_cc)/np.min(bsq_cc2)))
#               ))
#    a[0,0].imshow(unshifted_cc)
#    a[0, 0].title.set_text('unshifted_cc')
#    a[0,1].imshow(unshifted_cc-sunshifted_cc)
#    a[0, 1].title.set_text('unshifted_cc - shifted_ref_and_query_cc')    
#    a[1,0].imshow(av_q)
#    a[1, 0].title.set_text('averaged query')
#    a[1,1].imshow(av_sq)
#    a[1, 1].title.set_text('averaged shifted query')
#    a[1,2].imshow(av_r)
#    a[1, 2].title.set_text('averaged ref')
#    a[2,0].imshow(unshifted_cc- bsq_cc2)
#    a[2, 0].title.set_text('unshifted_cc - shifted_then_backshifted_query_cc')
#
#
#    plt.show()
##    write_mrc(join(testing, 'unshiftedcc%02d.mrc' % pcl_n), unshifted_cc)
##    write_mrc(join(testing, 'sunshiftedcc%02d.mrc' % pcl_n), sunshifted_cc)
##    write_mrc(join(testing, 'diff%02d.mrc' % pcl_n), unshifted_cc-sunshifted_cc)
#    
#testing#############################################

    #cc between ref and query
    unshifted_cc = get_cc_map(av_r*new_mask, av_q*new_mask, limit, interp)
    #cc between ref and query that was "corrected" by calculated shifts
    shifted_cc = get_cc_map(av_r*new_mask, av_sq*new_mask, limit, interp)

    #shift everything to positive range...debatable
    mi = np.min((unshifted_cc, shifted_cc))  
    unshifted_cc -= mi
    shifted_cc -= mi 

    #check unshifted cc map values within 4 unbinned pixels of median shift, 
    #but make sure cc map indices are not out of bounds
    box_y = (max(0, int(med_shift[0]) - peak_tol),
              min(int(med_shift[0]) + peak_tol, size))
    box_x = (max(0, int(med_shift[1]) - peak_tol),
              min(int(med_shift[1]) + peak_tol, size))  
    umask = c_mask(size, med_shift[1], med_shift[0], peak_tol)
    #umask = c_mask(size, size/2, size/2, peak_tol)
    m_unshifted_cc = np.where(umask, unshifted_cc, 0)
    ubox = m_unshifted_cc[box_y[0]:box_y[1], box_x[0]:box_x[1]]

    #check shifted cc map within 4 unbinned pixels of centre 
    #(the peak should now be centered)
    smask = c_mask(size, size//2, size//2, peak_tol)
    
    
    #m_shifted_cc = shifted_cc*smask
    m_shifted_cc = np.where(smask, shifted_cc, 0)

    sbox_xy = (max(0, size//2 - peak_tol), min(size//2 + peak_tol, size))
    sbox = m_shifted_cc[sbox_xy[0]:sbox_xy[1], sbox_xy[0]:sbox_xy[1]]
  
    #is there cc improvement within the 4 pixels of the epxected peak?
    smax, umax = np.max(shifted_cc[smask]), np.max(unshifted_cc[umask])
    r = smax/umax
    
#    a[2].plot(np.max(m_unshifted_cc, axis = 0))
#    a[2].plot(np.max(m_shifted_cc, axis = 0)) 
#    a[3].plot(np.max(m_unshifted_cc, axis = 1))
#    a[3].plot(np.max(m_shifted_cc, axis = 1)) 
#    f.show()
#    plt.show()
    
    if debug > 1:
        fig, axs = plt.subplots(3, 2, figsize = (7, 11))
        fig.suptitle(('CCmaps of unshifted vs. shifted query #%s.' + 
                     '\nPeak change = %.2f%%') % (pcl_n, r*100))
        axs[0, 0].imshow(unshifted_cc, cmap = 'Greys')
        axs[0, 0].add_artist(circle(med_shift[1], med_shift[0], peak_tol))
        axs[0, 0].title.set_text('Unshifted query CC')
        axs[0, 0].scatter(med_shift[1], med_shift[0],
                           marker = '+', c = 'red')
        axs[0, 0].annotate('median shift', (med_shift[1], med_shift[0]))
        axs[0, 1].imshow(shifted_cc, cmap = 'Greys')
        axs[0, 1].add_artist(circle(size//2, size//2, peak_tol))
        axs[0, 1].title.set_text('Shifted query CC')        
        axs[1, 0].imshow(ubox, cmap = 'Greys')
        axs[1, 0].title.set_text('Unshifted zoom')
        axs[1, 1].imshow(sbox, cmap = 'Greys')
        axs[1, 1].title.set_text('Shifted zoom')    
        axs[2, 0].plot(np.max(m_unshifted_cc, axis = 0), label = 'unshifted')
        axs[2, 0].plot(np.max(m_shifted_cc, axis = 0), label = 'shifted')
        axs[2, 0].title.set_text('X maximum projection')
        axs[2, 0].legend(loc = 'lower right')
        axs[2, 1].plot(np.max(m_unshifted_cc, axis = 1), label = 'unshifted')
        axs[2, 1].plot(np.max(m_shifted_cc, axis = 1), label = 'shifted')
        axs[2, 1].title.set_text('Y maximum projection')
        axs[2, 1].legend(loc = 'lower right')
        #fig.tight_layout()
        if not out_name:
            out_name = 'postCC_improvement_check_%s.png' % pcl_n 
        fig.savefig(join(output_dir, out_name), bbox_inches = 'tight')  
        plt.close()

    avg_ccmaps = (unshifted_cc, shifted_cc)
    return r, avg_ccmaps

def failed_peak_scaling(x):
    """
    Used to scale centre_bias
    sigmoidal curve: centre_bias grows exponentially up to ~3, then 
    plateaus at ~17.  Starting centre_bias values get multiplied by 1.
    """
    x = x*max(1, (1./x**0.18 + 0.4))
    return x

def bias_increase(tilt_angle, centre_bias, thickness_scaling):
    """
    Scales centre_bias to compensate for loss of signal at 
    higher tilts.
    used within biased_cc and cc_walk 
    
    Inputs:
        tilt_angle [float, list/array of floats] 
        centre_bias [positive float] 
        thickness_scaling [positive float]
    Output:
        cos_centre_bias [float, list/array of floats] type matches tilt_angle
    """
    cos_centre_bias = (centre_bias
                   *1/np.cos(np.radians(tilt_angle*thickness_scaling)))
    return cos_centre_bias


def cc_walk(pcl_n, ccmaps, shifts, interp, limit, top, bot, direction, step,
            tilt_angles, centre_bias, thickness_scaling,
            out_dir = False, debug = 0, allow_large_jumps = False):
    """
    bot = top
    top = len(ref)
    {direction} - 1 => walk up, -1 => walk down
    
    
    {failed_peak} - Each time peak is out of bounds, centre_bias and power 
                    gets scaled by these values


    bias 0.1, power 1.5 working well for capsids
    
    """

    #testing##########
    
    def loop(pcl_n, n, x, shifts, med, step, interp, limit, centre_bias,
             debug = 0):
        """
        x - tilt number
        """
        warnings = []
        #walking average, previous {step} values
        db = g2d(med[1], med[0], interp*limit*2, centre_bias)
        #std = np.std(shifts[x-step:x-1,0]), np.std(shifts[x-step:x-1:,1])    
        ccmap = ccmaps[x]*db    
        if debug < 2:
            pk, val, warn = buggy_peaks(ccmap)
            if warn:
                #no peak found, buggy_peaks returned ccmap max value
                # ==> use med
                pk = np.array([med, [0,0]])
                w = ["Warning: buggy_peaks failed to find a peak." +
                     " Using median shift"]
                warnings.extend(w)
        if debug > 0:
            #check ratio of db max and quarter box from the centre
            #THIS NEEDS TO BE REDONE!!!!!!!!!!!!!!!!!!!!!
            #
            #
            #
            #
            #
            #
            dbmax = np.max(db)
            mm = c_mask(interp*limit*2, med[1], med[0], interp*limit/2)
            try:
                quarter = np.min(db[mm])
                dbratio = dbmax/quarter
                wmsg = ["Weighting function ratio at maximum and quarter" +
                        " box distance from the centre %.2f" % dbratio]
                warnings.extend(wmsg)   
            except:
                wmsg = ["Failed to find weighting function ratio." +
                        " Particle %s tilt %s" % (pcl_n, x)]
                print(wmsg)
                warnings.extend(wmsg)   
        if debug > 1:
            pk, val, warn = buggy_peaks(ccmaps[x])
            if warn:
                pk = np.array([pk, [0,0]])
            axs[0, n].imshow(ccmaps[x], cmap = 'Greys')
            axs[0, n].add_artist(circle(pk[0,1], pk[0,0], interp))
            axs[0, n].title.set_text('Raw cc #%s:\n peak[%s,%s]'
                                           % (x, pk[0,1], pk[0,0]))
            axs[2, n].plot(norm(np.max(ccmaps[x], axis = 0)), label = 'raw')           
            pk, val, warn = buggy_peaks(ccmap)
            if warn:
                #no peak found, buggy_peaks returned ccmap max value
                # ==> use med
                pk = np.array([med, [0,0]])
                w = ["buggy_peaks failed to find a peak.  Using median shift"]
                warnings.extend(w)
            axs[1, n].imshow(ccmap, cmap = 'Greys')
            axs[1, n].add_artist(circle(pk[0,1], pk[0,0], interp))
            axs[1, n].title.set_text('biased cc #%s:\n peak[%s,%s]'
                                           % (x, pk[0,1], pk[0,0]))     
            axs[2, n].plot(norm(np.max(ccmap, axis = 0)), label = 'biased')
            axs[2, n].title.set_text('X max proj.')
            axs[2, n].legend(loc = 'lower right')
            #display db:
            #a[n].plot(np.diagonal(db))
        if pk.ndim > 1:
            return pk[0], warnings
        if len(pk) == 0:
            raise ValueError(
                ('get_cc_map, particle %s tilt %s returned a flat array.' +
                "This can happen if the input model and volumes don't match")
                % (pcl_n, x))  
    n = -1
    if direction < 0:
        #swap things around to walk down...
        bot, top = top, bot - 1
    if debug == 2:
        fig, axs = plt.subplots(3, np.absolute(top-bot),
                                figsize = (np.absolute(top-bot)*3.5, 11))
        fig.suptitle('Ccs of tilts %s-%s' % (
                min(top, bot) + 1, max(top, bot) + 1))
        #display db:
        #f, a = plt.subplots(1, np.absolute(top-bot))

    debug_out = []
    for x in range(bot, top, direction): 
        n += 1
        f0, f1 = (min(x - step*direction,  x - 1*direction),
                   max(x - step*direction,  x - 1*direction))
        med = np.median(shifts[f0:f1], axis = 0)    
        #centre bias  and power are increased with tilt angle. 
        #The 1/cos powers were determined empirically. 1 turns the tilt angle increase off
        cos_centre_bias = bias_increase(tilt_angles[x], centre_bias,
                                        thickness_scaling)
        pk, w = loop(pcl_n, n, x, shifts, med, step, interp, limit, 
                  cos_centre_bias, debug)
        if debug > 0:
            debug_out.extend(['\nTilt\t%s' % (x + 1)])
            debug_out.extend(w)

        #Now check for values outside 5x shift STD.  Need a lower bound
        #that != 0 (when STD = 0).  When med shift == 0 and STD == 0, half an
        #uninterpolated pixel
  
        xystd = np.std(shifts[f0:f1], axis = 0)
        std = np.sqrt(xystd[0]**2 + xystd[1]**2)
        dxy_shift = np.absolute(shifts[x - 1*direction] - pk)
        d_shift = np.sqrt(dxy_shift[0]**2 + dxy_shift[1]**2)
        med_xy_shift = np.array([interp*limit, interp*limit]) - med
        md_shift = np.sqrt(med_xy_shift[0]**2 +  med_xy_shift[1]**2)
        if debug > 0:
            wm = (
                ("Centre bias, tilt-adjusted centre bias\t%.2f\t%.2f" %
                                 (centre_bias, cos_centre_bias)),
                ("Peak, current shift, median shift\t%s\t%.2f\t%.2f" %
                                 (pk, d_shift, md_shift)),
                ("STD of shifts %s-%s\t\t\t%.2f" % (f0 + 1, f1 + 1, std))
                )
            debug_out.extend(wm)
        lbound = max(5*std, md_shift, interp/2)        
        iters = 0
        max_iters = 6
        n_centre_bias = centre_bias
        while d_shift > lbound:
            iters += 1
            if iters < max_iters:
                #tilt angle weighting:
                #changing variable names to n_ so that the bias increase
                #is not preserved outside this loop
                n_centre_bias = failed_peak_scaling(n_centre_bias)
                n_cos_centre_bias = bias_increase(
                        tilt_angles[x], n_centre_bias, thickness_scaling)
                pk, w = loop(pcl_n, n, x, shifts, med, step, interp, limit, 
                      n_cos_centre_bias, debug)
                dxy_shift = np.absolute(shifts[x - 1*direction] - pk)
                d_shift = np.sqrt(dxy_shift[0]**2 + dxy_shift[1]**2)
                if debug > 0:
                    if iters == 1:
                        debug_out.extend(
                            ["Current shift is larger than" +
                             " 5xSTD of previous tilts (%.2f)" % (5*std)]
                                                )
                    wm = (    
                        ("Iteration %s of %s" % (iters, max_iters - 1)),
                        ("Centre bias\t%.2f"  % n_cos_centre_bias),
                        ("Current shift\t\t%.2f" % d_shift)
                        )
                    debug_out.extend(wm)
                    debug_out.extend(w)

            else:
                #Force shift to be 3*STD in the direction of the last peak.
                #get direction of change
                plus_or_minus = dxy_shift - med_xy_shift
                for hm in range(len(plus_or_minus)):
                        if plus_or_minus[hm] == 0:
                                plus_or_minus[hm] = 1
                plus_or_minus = plus_or_minus/np.absolute(plus_or_minus)

                if allow_large_jumps:     
                    tpk = pk
                else:
                    tpk = med + (3*xystd*plus_or_minus)
                if debug > 0:
                    wm = (
                            (('Peak out of bounds after %s iterations.'
                              % (max_iters - 1))),
                            ('Forcing peak %s' % tpk)
                         )
                    debug_out.extend(wm)
                    if debug > 1:                           
                        axs[1, n].add_artist(
                                circle(pk[1], pk[0], interp, 'blue'))
                pk = tpk
                break
        shifts[x] = pk
    if debug == 2:
        fig.savefig(
                    join(out_dir, '%02d_ccwalk_%s-%s.png' % 
                         (pcl_n, min(top, bot) + 1, max(top, bot) + 1)),
                    bbox_inches = 'tight'
                    )
        plt.close()

    return shifts, debug_out


def create_circular_mask(box, radius = 0):
    """stack overflow 44865023"""
    c = [int(box[0]/2), int(box[1]/2)]
    if not radius:
        radius = min(c[0], c[1], box[0] - c[0], box[1] - c[1])
    Y,X = np.ogrid[:box[0], :box[1]]
    dist_from_center = np.sqrt((X - c[0])**2 + (Y - c[1])**2)
    mask = dist_from_center <= radius
    return mask
    


##############################################################################
def csv_sync(out_dir, chunk_base, return_sorted = False):
    """
    combine csv files with name {chunk_base-*_tmp.csv} into a single
    chunk_base.csv file.  pcl ids are sorted in ascending order.
    """
    #read in all csv files and append them to a list
    csv_list = glob.glob(join(out_dir, chunk_base + '-*_tmp.csv'))
    if csv_list == []:
        raise Exception('csv_sync: No files matching %s-*_tmp.csv in %s' % 
                        (chunk_base, out_dir))
    csva, sorted_shifts = read_csv(csv_list)
    
    out_csv = join(out_dir, chunk_base + '.csv')
    if os.path.isfile(out_csv):
        os.rename(out_csv, out_csv + '~')
    #write completed csv
    with open(out_csv, 'w') as f:
        f.write('pcl id, tilt number, x shift, y shift\n')
        for x in range(len(csva)):
            f.write('%s,%s,%.2f,%.2f\n' % (int(csva[x][0]), int(csva[x][1]),
                                           csva[x][2], csva[x][3]))
    if return_sorted:
        return out_csv, sorted_shifts
    else:
        return out_csv

def plot_shifts(shifts, out_dir):
    shifts =np.array(shifts)
    std_shiftsx = [np.std(shifts[:,x,0]) for x in range(shifts.shape[1])]
    std_shiftsy = [np.std(shifts[:,x,1]) for x in range(shifts.shape[1])]
    mean_shiftsx = [np.mean(shifts[:,x,0]) for x in range(shifts.shape[1])]
    mean_shiftsy = [np.mean(shifts[:,x,1]) for x in range(shifts.shape[1])]
    f, axes = plt.subplots(1,2, sharey = False)
    axes[0].errorbar(list(range(shifts.shape[1])), mean_shiftsx, yerr = std_shiftsx)
    axes[0].title.set_text('X axis')    
    axes[0].set(xlabel = 'Tilt number', ylabel = 'shift [pixels]')
    axes[1].errorbar(list(range(shifts.shape[1])), mean_shiftsy, yerr = std_shiftsy)
    axes[1].set(xlabel = 'Tilt number')
    axes[1].title.set_text('Y axis')    
    
    plt.suptitle('Mean shift and STD')
    plt.savefig(join(out_dir, 'shifts.png'))
    plt.close()
    std_shiftsx = [np.std(shifts[0,x,0]) for x in range(shifts.shape[1])]
    std_shiftsy = [np.std(shifts[0,x,1]) for x in range(shifts.shape[1])]
    mean_shiftsx = [np.mean(shifts[0,x,0]) for x in range(shifts.shape[1])]
    mean_shiftsy = [np.mean(shifts[0,x,1]) for x in range(shifts.shape[1])]
    f, axes = plt.subplots(1,2, sharey = False)
    axes[0].errorbar(list(range(shifts.shape[1])), mean_shiftsx, yerr = std_shiftsx)
    axes[0].title.set_text('X axis')    
    axes[0].set(xlabel = 'Tilt number', ylabel = 'shift [pixels]')
    axes[1].errorbar(list(range(shifts.shape[1])), mean_shiftsy, yerr = std_shiftsy)
    axes[1].set(xlabel = 'Tilt number')
    axes[1].title.set_text('Y axis')    

    plt.suptitle('Mean shift and STD')
    plt.savefig(join(out_dir, 'single_shifts.png'))
    plt.close()


##############################################################################

def rotate_shifts(shifts, angles):  
    """Rotates shifts around 0.  Positive counter-clockwise """
    rshifts = []
    for x in range(angles.shape[0]):
        rx =  (shifts[:,x,0] * np.cos(np.radians(angles[x]))
                - shifts[:,x,1]*np.sin(np.radians(angles[x])))
        ry =  (shifts[:,x,0]*np.sin(np.radians(angles[x])) 
                + shifts[:,x,1]*np.cos(np.radians(angles[x])))
        tshifts = np.dstack((rx,ry))
        rshifts.append(tshifts)
    rshifts = np.swapaxes(np.array(rshifts).squeeze(), 0,1)
    return rshifts

def read_csv(csv_paths):
    """
    Read a single csv, or combine multiple csv files into a list.
    First line of each file is removed if it contains a string.
    """
    #convert to list if there is only one path specified
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    csva = []    
    for x in range(len(csv_paths)):
        with open(csv_paths[x]) as f:
            rc = csv.reader(f, delimiter = ',')
            cl = [y for y in rc]
            try:
                #remove first line if it's a str
                #read as str, need to try int()
                [int(y) for y in cl[0]]
            except:
                del(cl[0])
            csva.extend(cl)
    csva = np.array(csva, dtype = float)
    scsva = []
    sorted_shifts = []
    for x in np.array(np.unique(csva[:,0]), dtype = int):
        scsva.extend(csva[csva[:, 0] == x])
        sorted_shifts.append(csva[csva[:, 0] == x])
        
    sorted_shifts = np.swapaxes(np.array(sorted_shifts, dtype = float), 0, 1)
    #rearrange to match the format of ssorted_pcls:
    #[ntilts:npcls:xcoord, ycoord, zcoord(tilt), pcl index]
    sorted_shifts = sorted_shifts[:,:,[2, 3, 1, 0]]
    return np.array((scsva, sorted_shifts))


def p_shifts_to_model_points(ssorted_pcls, sorted_shifts, out_dir,                          
                             excludelist = False,
                             groups = False,
                             ali = False,
                             debug = 0):

    """
    ssorted_pcls [numpy array]
        [number of tilts:model points per tilt:
        xcoords, ycoords, tilt number, particle index, group id (from 0))]
        group id == len(groups) indicates unallocated pcls 
    sorted_shifts [str or numpy array]
        [ntilts:npcls:xcoord, ycoord, zcoord(tilt), pcl index]
        if str, expecting combined csv path
    out_dir [str]
    
    """
#TBD
#    if debug > 1:
#        plot_shifts(shifts, out_dir)
    #remove particles not included in any groups

    #allow sorted_shifts input to be path to csv file(s)
    if isinstance(sorted_shifts, str):
        csva, sorted_shifts = read_csv(sorted_shifts)
    #remove pcls not included in any groups
    if not isinstance(groups, bool):
        if groups.ndim == 1:
            #in case there is only one group
            gmask = groups
        else:
            gmask = np.sum(groups, axis = 0, dtype = bool)
        ssorted_pcls = ssorted_pcls[:, gmask]
    #trim rsorted_pcls using excludelist (shifts should already be cleaned)
    if not isinstance(excludelist, bool):
        excludelist = np.array(excludelist) - 1
        exc_mask = np.isin(list(range(len(ssorted_pcls))), excludelist,
                           invert = True)
        ssorted_pcls = ssorted_pcls[exc_mask]

    #check that ssorted_pcls pcl indices and tilt numbers match sorted_shifts
    if (np.array(ssorted_pcls[:, :, 2:4], dtype = int) ==
        np.array(sorted_shifts[:, :, 2:4], dtype = int)).all():
        ssorted_pcls = ssorted_pcls[:, :, :3]
        ssorted_pcls[:, :, :2] = (ssorted_pcls[:, :, :2]
                                    - sorted_shifts[:, :, :2])
    else:
        raise ValueError(('Unmatched ssorted_pcls and sorted_shifts numbering.'
                + '\nssorted_pcls[0, :, 3]:\n%s\nsorted_shifts[0, :, 3]:\n%s'
                + '\nssorted_pcls[:, 0, 2]:\n%s\nsorted_shifts[:, 0, 2]:\n%s')
                         % (ssorted_pcls[0, :, 3], sorted_shifts[0, :, 3],
                         ssorted_pcls[:, 0, 2], sorted_shifts[:, 0, 2]))

    outmod = PEETmodel() 
    for p in range(ssorted_pcls.shape[1]):
        #I suspect an contour is already created with PEETmodel() instance,
        #no need to add it for the first pcl
        if p != 0:
                outmod.add_contour(0)        
        for r in range(ssorted_pcls.shape[0]):
            outmod.add_point(0, p, ssorted_pcls[r,p])
            
    outmod_name = abspath(join(out_dir, 'flexo_ali.fid'))
    outmod.write_model(outmod_name)

    if ali:
        #set image coordinate information from the given image file
        check_output('imodtrans -I %s %s %s' % 
                     (ali, outmod_name, outmod_name), shell = True)
    return outmod_name


##############################################################################

def get_binned_size(img_shape, binning):
   binned_shape = []
   offsets = []
   for dim in img_shape:
       dim_bin = int(float(dim)/float(binning))
       rem = dim-(binning*dim_bin)
       if dim % 2 == dim_bin % 2:
           if rem > 1:
               dim_bin = dim_bin+2
       else:
           dim_bin=dim_bin+1
           rem=rem+binning
       if rem > 1:
           offset = -(binning-(rem/2.))
       else:
           offset = 0
       binned_shape.append(dim_bin)
       offsets.append(offset)
   return tuple(binned_shape), tuple(offsets)  

###############################################################################

def readable_excludelist(excludelist):
    """
    Converts to a comma separated string, abbreviates strictly ascending
    lists as ranges.
    e.g. [2,3,4,6,7,8] --> '2-4,6-8'
        [2,3] --> '2,3'
        [2] --> '2'
    """

    if len(excludelist) == 1:
        return str(excludelist[0])

    excludelist.sort()
    startstop = []
    tmp = [excludelist[0]]  
    for x in range(1, len(excludelist)):
        if excludelist[x] - 1 == excludelist[x - 1]:
            tmp.append(excludelist[x])
        else:
            startstop.append(tmp)
            tmp = [excludelist[x]]
        if x == len(excludelist) - 1:
            startstop.append(tmp)
    r = []
    for x in range(len(startstop)):
        if len(startstop[x]) < 3:
            r.extend([str(y) for y in startstop[x]])
        else:
            r.append('%s-%s' % (startstop[x][0], startstop[x][-1]))
    return (',').join(r)

def format_align(out_dir, base_name, ali, tlt, binning, fid_model,
                 axiszshift, xf, separate_group,
                 fidn, n_patches, global_only, globalXYZ, OFFSET,
                 excludelist = False,
                 com_ext = ''):
    overlap = 0.5
    base_output = abspath(join(out_dir, base_name))
    output_model = str(base_output) + '.3dmod'
    output_resid = str(base_output) + '.resid'
    output_xyz = str(base_output) + '.xyz'
    output_tilt = str(base_output) + '.tlt'
    xfile = str(base_output) + '.xtlt'
    output_tltxf = str(base_output) + '.tltxf'
    output_nogapsfid = str(base_output) + '_nogaps.fid'
    output_xf = str(base_output) + '.xf'
    output_resmod = str(base_output) + '.resmod'
    
    xstretch = 0
    if xstretch:
        output_zfac = str(base_output) + '.zfac'
    else:
        output_zfac = False
    if not global_only:
        output_localxf = str(base_output) + '_local.xf'
    else:
        output_localxf = False
        

    fid_n1, fid_n2 = fidn

    if tlt == output_tilt:
        fidtlt = str(base_output) + '.fidtlt'
        if os.path.isfile(fidtlt):
            os.rename(fidtlt, fidtlt + '~')
        os.rename(tlt, fidtlt)
#        if os.path.isfile(fidtlt):
#            tlt = fidtlt
    tilt_angles = [float(x.strip('\n\r').strip(' ')) for x in open(tlt)]
    zero_tlt = find_nearest(tilt_angles, 0)

    #rotation is zero if aligned stack is used
#    rotation = check_output('xf2rotmagstr %s' % xf, shell = True)
#    rotation = np.median(np.array(
#            [float(x.split()[2][0:-1]) for x in rotation.split('\n')[2:-1]]))


    with open(join(out_dir, 'align%s.com' % com_ext), 'w') as f:
        f.write('$tiltalign\t-StandardInput')
        f.write('\nModelFile\t%s' % fid_model)
        f.write('\nImageFile\t%s' % ali)
        f.write('\nImagesAreBinned\t%s' % int(binning))
        f.write('\nOutputModelFile\t%s' % output_model) 
        f.write('\nOutputResidualFile\t%s' % output_resid) 
        f.write('\nOutputFidXYZFile\t%s' % output_xyz) 
        f.write('\nOutputTiltFile\t%s' % output_tilt) 
        f.write('\nOutputXAxisTiltFile\t%s' % xfile) 
        f.write('\nOutputTransformFile\t%s' % output_tltxf) 
        f.write('\nOutputFilledInModel\t%s' % output_nogapsfid) 
        if xstretch:
            f.write('\OutputZFactorFile\t%s' % output_zfac) 
        #rotation angle is adjusted by offset by checking final ali .xf
        #it should be 0 since it's the aligned stack that is being used!

#        f.write('\nRotationAngle\t%s' % -tilt_axis_angle) 
        f.write('\nRotationAngle\t0') 



        if separate_group != '':
            f.write('\nSeparateGroup\t%s' % separate_group) 
        f.write('\nTiltFile\t%s' % tlt) 
        if OFFSET:
            try:
                OFFSET[0]
                f.write('\nAngleOffset %s' % OFFSET[0]) 
            except:
                f.write('\nAngleOffset %s' % OFFSET)         
        f.write('\nRotOption\t1') 
        f.write('\nRotDefaultGrouping\t5') 
        f.write('\nTiltOption\t2')
        f.write('\nTiltOption\t0')
        
        
        f.write('\nTiltDefaultGrouping\t5')
        f.write('\nMagReferenceView\t%s' % zero_tlt)
        f.write('\nMagOption\t1')
        f.write('\nMagDefaultGrouping\t4')
        f.write('\nXStretchOption\t%s' % xstretch)
        f.write('\nSkewOption\t0')
        f.write('\nXStretchDefaultGrouping\t7')
        f.write('\nSkewDefaultGrouping\t11')
        f.write('\nBeamTiltOption\t0')
        f.write('\nXTiltOption\t4')
        f.write('\nXTiltDefaultGrouping\t2000')
        f.write('\nResidualReportCriterion\t3.0')
        f.write('\nSurfacesToAnalyze\t2')
        f.write('\nMetroFactor\t0.25')
        f.write('\nMaximumCycles\t1000')
        f.write('\nKFactorScaling\t1')
        f.write('\nNoSeparateTiltGroups\t1')
        f.write('\nAxisZShift\t%s' % axiszshift)
        f.write('\nShiftZFromOriginal\t1')
        f.write('\nShiftZFromOriginal\t1')
        if not isinstance(excludelist, bool):
            f.write('\nExcludeList %s' % readable_excludelist(excludelist))
        if not global_only:
            f.write('\nLocalAlignments\t1')
            f.write('\nOutputLocalFile\t%s' % output_localxf)
            f.write('\nMinSizeOrOverlapXandY\t%s,%s' % (overlap, overlap))
            f.write('\nMinFidsTotalAndEachSurface\t%s,%s' % (fid_n1, fid_n2))
            if globalXYZ:
                f.write('\nFixXYZCoordinates\t1')
            else:
                print('global xyz coordinates can be relaxed in the last iteration')
                f.write('\nFixXYZCoordinates\t0')
            f.write('\nLocalOutputOptions\t1,0,1')
            f.write('\nLocalRotOption\t3')
            f.write('\nLocalRotDefaultGrouping\t6')
            f.write('\nLocalTiltOption\t5')
            f.write('\nLocalTiltDefaultGrouping\t6')
            f.write('\nLocalMagReferenceView\t%s' % zero_tlt)
            f.write('\nLocalMagOption\t3')
            f.write('\nLocalMagDefaultGrouping\t7')
            f.write('\nLocalXStretchOption\t0')
            f.write('\nLocalXStretchDefaultGrouping\t7')
            f.write('\nLocalSkewOption\t0')
            f.write('\nLocalSkewDefaultGrouping\t11')
            f.write('\nNumberOfLocalPatchesXandY\t%s,%s' % (n_patches[0], n_patches[1]))
        else:
            output_localxf = False
        f.write('\nRobustFitting\t')
        f.write('\n$xfproduct -StandardInput')
        f.write('\nInputFile1\t%s' % xf)
        f.write('\nInputFile2\t%s' % output_tltxf)
        f.write('\nOutputFile\t%s' % output_xf)
        #f.write('\n$b3dcopy -p %s %s' % (output_tltxf, output_xf))
        #f.write('\n$b3dcopy -p %s %s' % (output_tilt, output_fidtlt))
        f.write('\n$if (-e %s) patch2imod -s 10 %s %s' % (output_resid, output_resid, output_resmod))
        #f.write('\n$if (-e ./savework) ./savework')
        return output_xf, output_localxf, output_tilt, output_zfac

def format_newst(base_name, out_dir, st, xf, binning,
                 tomo_size,
                 output_ali = False, com_ext = ''):
    if not output_ali:
        base_output = abspath(join(out_dir, base_name))
        output_ali = str(base_output) + '.ali'
    
    #here for simplicity, but takes ~150ms...
    st_apix, st_size = get_apix_and_size(st)
    binned_size = get_binned_size(st_size, binning)[0][:2]
    if tomo_size[0] > tomo_size[1] and binned_size[0] < binned_size[1]:
        binned_size = binned_size[::-1]
    elif tomo_size[0] < tomo_size[1] and binned_size[0] > binned_size[1]:
        binned_size = binned_size[::-1]
        
    with open (join(out_dir, 'newst%s.com' % com_ext) ,'w') as f:
        f.write('$newstack -StandardInput')
        f.write('\nInputFile\t%s' % st)
        f.write('\nOutputFile\t%s' % output_ali)
        f.write('\nTransformFile\t%s' % xf)
        f.write('\nTaperAtFill\t1,0')
        f.write('\nAdjustOrigin')
        f.write('\nSizeToOutputInXandY\t%s,%s' % binned_size)
        f.write('\nOffsetsInXandY\t0.0,0.0')
#        if excludelist:
#            f.write('\nExcludeSections\t%s' % (',').join(str(x) for x in excludelist))
        print('Newst binning %s' % binning)
        f.write('\nBinByFactor\t%s' % int(binning))
        f.write('\nAntialiasFilter\t-1')
        #f.write('\n$if (-e ./savework) ./savework')
    return output_ali

def format_tilt(base_name, out_dir, ali, tlt, binning, thickness, global_xtilt,
                OFFSET = False, SHIFT = (0,0), xfile = False, localxf = False,
                zfac = False, excludelist = False, output_rec = False,
                fakesirt = 0, com_ext = '',
                #reprojection: all three required
                reproject = False,
                repvol = False,
                repali = False): 
    base_output = abspath(join(out_dir, base_name))
    if not output_rec:
        output_rec = str(base_output) +  '_full.rec'
#    xfile = str(base_output) +  '.xtlt'
#    zfac = str(base_output) +  '.zfac'

    with open (join(out_dir, 'tilt%s.com' % com_ext) ,'w') as f:
        f.write('$tilt -StandardInput')
        if not isinstance(reproject, bool):
            f.write('\nREPROJECT\t%s' % (',').join(
                                                [str(x) for x in reproject]))
            f.write('\nRecFileToReproject %s' % repvol)
            output_rec = repali
        f.write('\nInputProjections\t%s' % ali)
        f.write('\nOutputFile\t%s' % output_rec)
        f.write('\nTILTFILE\t%s' % tlt)
        
        #this is needed specifically for orthogonal subtraction
        #but shouldn't affect normal rec
        f.write('\nWeightAngleFile\t%s' % tlt)
        
        f.write('\nIMAGEBINNED\t%s' % int(binning))
        f.write('\nTHICKNESS\t%s' % thickness)
        f.write('\nRADIAL\t0.5\t0.1') 
        if fakesirt:
            f.write('\nFakeSIRTiterations\t%s' % fakesirt)
        f.write('\nFalloffIsTrueSigma\t1')
        f.write('\nXAXISTILT\t%s' % global_xtilt)
        f.write('\nSCALE\t0.0\t0.2')
        f.write('\nPERPENDICULAR')
        if not isinstance(excludelist, bool):
            f.write('\nEXCLUDELIST %s' % readable_excludelist(excludelist))
        f.write('\nSUBSETSTART\t0 0')
        f.write('\nAdjustOrigin')
        f.write('\nActionIfGPUFails 1,2')
        if xfile:
            f.write('\nXTILTFILE\t%s' % xfile) 
        if OFFSET:
            try:
                OFFSET[0]
                f.write('\nOFFSET %s' % OFFSET[0]) 
            except:
                f.write('\nOFFSET %s' % OFFSET) 
        if np.any(SHIFT):
            f.write('\nSHIFT %s %s' % (SHIFT[0], SHIFT[1]))
        if localxf:
            f.write('\nLOCALFILE\t%s' % localxf)
        if zfac:
            f.write('\nZFACTORFILE\t%s' % zfac)
    return output_rec

def format_ctfcorr(ali, tlt, xf, defocus_file, out_dir, base_name,
                   V, Cs, ampC, apix,
                   deftol = 200,
                   interp_w = 4,
                   output_ali = False, com_ext = ''):

    if not output_ali:
        base_output = abspath(join(out_dir, base_name))
        output_ali = str(base_output) + '_ctfcorr.ali'
    
    with open(join(out_dir, 'ctfcorrection%s.com' % com_ext), 'w') as f:
        f.write('$ctfphaseflip\t-StandardInput')
        f.write('\nInputStack\t%s' % ali)
        f.write('\nAngleFile\t%s' % tlt)
        f.write('\nOutputFileName\t%s' % output_ali)
        f.write('\nTransformFile\t%s' % xf)
        f.write('\nDefocusFile\t%s' % defocus_file)
        f.write('\nVoltage\t%s' % int(V/1E3)) #kV
        f.write('\nSphericalAberration\t%s' % (Cs/1E7)) #mm
        f.write('\nDefocusTol\t%s' % deftol)
        f.write('\nPixelSize\t%s' % (apix/10.)) #nanometers
        f.write('\nAmplitudeContrast\t%s' % ampC)
        f.write('\nInterpolationWidth\t%s' % interp_w)

    return output_ali

#################################################################################
def get_tomo_transform(ref, query, angrange, transform = 'rotation', 
                           interp = 1, out_dir = False, nstrips = 20,
                           subset = 4, new = True,
                           edge = 3):
    """
    chops tomos into {nstrips} strips that are then flattened into 2d images.
    checks rotations for either rotation between these
    
    returns either Y, X rotation or translation (same convention as imod tilt)
    
    translation is unreliable if there is rotation between tomos
    
    ref, query {path to tomograms  or np arrays} 
        expecting flexo first, then initial tomo
        VOLUMES HAVE TO BE ROTATED BY 90 DEGREES!
        This is done automatically if paths are specified instead of arrays
    angrange {even real}
    subset {real} perform rotation with every nth strip
    
    """
    if isinstance(ref, str):
        tiny_flexo = deepcopy(mrcfile.open(ref).data)
        tiny_flexo = np.rot90(tiny_flexo, k = 1, axes = (0,1))
    else:
        tiny_flexo = ref
    if isinstance(query, str):
        tiny_orig = deepcopy(mrcfile.open(query).data)
        tiny_orig = np.rot90(tiny_orig, k = 1, axes = (0,1))
    else:
        tiny_orig = query

    #chop into three strips, rather than flattening the whole tomo
    volshape = np.array(tiny_flexo.shape)
    borders = np.array(
            np.round([volshape[1]*0.1, volshape[2]*0.1]), dtype = 'int')
    width = np.array(
            np.round((volshape[1:3] - borders*2)/nstrips), dtype = 'int')
    
    def make_strips(volume, border, width, axis, nstrips = 20):
        if axis == 2:
            volume = np.swapaxes(volume, 1, 2)
        s = []
        for x in range(nstrips):
            s.append(np.mean(
                    volume[:, border + width*x : width*(x + 1) + border],
                    axis = 1))
        return np.array(s)
    def filter_strips(stack, apix, lowpass = 10):
        f = []
        for x in range(len(stack)):
            f.append(butter_filter(stack[x], lowpass, apix))
        return np.array(f)
    
    ysh = []
    yrot = []
    xsh = []
    xrot = []
    if transform == 'rotation':
        #project along Y axis (XZ)
        oxz = make_strips(tiny_orig, borders[0], width[0], 1, nstrips) 
        #oxz = filter_strips(oxz, apix)
        fxz = make_strips(tiny_flexo, borders[0], width[0], 1, nstrips)   
        #fxz = filter_strips(fxz, apix)
        
        #project along X axis  (YZ)
        oyz = make_strips(tiny_orig, borders[1], width[1], 2, nstrips)  
        #oyz = filter_strips(oyz, apix)
        fyz = make_strips(tiny_flexo, borders[1], width[1], 2, nstrips)  
        #fyz = filter_strips(fyz, apix)

        #do rotation check for a subset of strips
        sub_oxz = oxz[::subset]
        sub_fxz = fxz[::subset]
        sub_oyz = oyz[::subset]
        sub_fyz = fyz[::subset]
        
        if out_dir:

            write_mrc(join(out_dir, 'matchref_ystrips.mrc'), oxz)
            write_mrc(join(out_dir, 'matchq_ystrips.mrc'), fxz)
            write_mrc(join(out_dir, 'matchref_xstrips.mrc'), oyz)
            write_mrc(join(out_dir, 'matchq_xstrips.mrc'), fyz)
        
        #first check for rotation, then remake flexo tomogram 
        for x in range(len(sub_oxz)):
            yrot.append(in_plane_rotation(sub_fxz[x], sub_oxz[x], angrange,
                                   out_dir, sub_oxz[x].shape[0]//2 - 1,
                                   interp = 1,
                      order = 3, refine = True, edge = edge, mode = 'nearest'))

        for x in range(len(sub_oyz)):
            xrot.append(in_plane_rotation(sub_fyz[x], sub_oyz[x], angrange,
                                   out_dir, sub_oyz[x].shape[0]//2 - 1,
                                   interp = 1,
                      order = 3, refine = True, edge = edge, mode = 'nearest'))        
        yrot = np.array(yrot[yrot != np.nan])
        xrot = np.array(xrot[xrot != np.nan])

        #remake flexo tomogram with corrected rotation
        if out_dir:
            f, axes = plt.subplots()
            f.suptitle('relative tomogram rotation')
            axes.plot(yrot, label = 'Y rotation')
            axes.plot(xrot, label = 'X rotation')
            axes.legend()
            axes.set_xlabel('strip number')
            axes.set_ylabel('rotation [degrees]')
            f.savefig(join(out_dir, 'tomogram_rotation.png'))  
            plt.close()

        return np.median(yrot), np.median(xrot)

    elif transform == 'translation':
        oxy = np.mean(tiny_orig, axis = 0)
        fxy = np.mean(tiny_flexo, axis = 0)
        #project along Y axis (XZ)
        oxz = make_strips(tiny_orig, borders[0], width[0], 1, nstrips) 
        fxz = make_strips(tiny_flexo, borders[0], width[0], 1, nstrips)   
        
        #get x shift from xy projection (y should not change)
        cc = ncc(oxy, fxy, (min(oxy.shape)//2)//2 - 1, interp)    
        xsh = (cc.shape[0]//2. - np.where((cc == cc.max()))[0],
                   cc.shape[1]//2. - np.where((cc == cc.max()))[1])    

        xsh = np.flip(np.squeeze(xsh), axis = 0)
        for x in range(len(oxz)):
            cc = ncc(
                    oxz[x], fxz[x], (min(oxz[x].shape)//2)//2 - 1, interp)
            sh = (cc.shape[0]/2. - np.where((cc == cc.max()))[0],
                       cc.shape[1]//2. - np.where((cc == cc.max()))[1])
       
            ysh.append(sh) 
        ysh = np.flip(np.squeeze(ysh), axis = 1)
        ysh[:,1] = -ysh[:,1] #X shift, Z shift
        mysh = np.median(ysh, axis = 0) 
        #xy X shift is much better than xz X shift
        mysh[0] = xsh[0]
        print('xy shift, xz shift %s %s' % (str(xsh), str(mysh)))
        if out_dir:

            write_mrc(join(out_dir, 'matchref_ystrips.mrc'), oxz)
            write_mrc(join(out_dir, 'matchq_ystrips.mrc'), fxz)

            f, axes = plt.subplots(2,1, figsize = (6,10))
            f.suptitle('relative tomogram translation')
            axes[0].title.set_text('Z shift from Y slices')
            axes[0].plot(ysh[:, 0])
            axes[1].title.set_text('X shift from Y slices')
            axes[1].plot(ysh[:, 1])
#            axes[2].title.set_text('Z shift from X slices')
#            axes[2].plot(-xsh[:, 0])

            axes[1].set_xlabel('strip number')
            axes[1].set_ylabel('shift [pixels]')
            f.savefig(join(out_dir, 'tomogram_translation.png'))  
            plt.close()
        return mysh
        

def match_tomos(tomo_binning, out_dir, base_name, rec_dir, how_tiny, tomo_size,
                copy_orig = False, fakesirt = 0):
    
    """
    returns translation and rotation between two tomograms:
        1) extracts parameters from rec_dir and out_dir comfiles
        2) makes (binned) versions
        3) checks rotation and translation by comparing strips of each tomo
    How tiny: additional binning relative to input tomo
    works fine at bin 16 (60 apix), accurate to ~3 unbinned pixels and 0.2 degrees
    copy_orig [str] path to correctly binned ref tomo for matching
        if not False, skips reference tomo generation
    """
    out_bin = tomo_binning*how_tiny

    angrange = 20 #+/- 10 degrees
    bin_orig_ali = join(out_dir,'matchref_' + base_name + '.ali')
    bin_flexo_ali = join(out_dir,'matchq_' + base_name + '.ali')
    out_full_tomo = abspath(join(out_dir, 'matchq_' + base_name + '_full.rec'))
#    out_tomo = abspath(join(out_dir, 'matchq_' + base_name + '.rec'))
    tiny_orig_full =  abspath(join(
                            out_dir, 'matchref_' + base_name + '_full.rec'))    
#    tiny_orig =  abspath(join(out_dir, 'matchref_' + base_name + '.rec'))

    #tiny_size = np.array(get_binned_size(tomo_size, how_tiny)[0])


    #first, get initial tomogram binned:
    #read_comfiles output: 0 st, 1 xf, 2 ali, 3 tlt, 4 thickness,
        #5 global_xtilt, 6 localxf, 7 excludelist, 8 OFFSET, 9 SHIFT,
        #10 separate_group, 11 axiszshift, 12 zfac, 13 xfile
    op = read_comfiles(rec_dir)
    format_newst(base_name, out_dir, op[0], op[1], out_bin, tomo_size,
                 bin_orig_ali, '_orig')
    format_tilt(base_name, out_dir, bin_orig_ali, op[3], out_bin, op[4],
                op[5], op[8], op[9], op[6], op[12], op[13], op[7],
                tiny_orig_full, fakesirt, '_orig')
    imodscript('newst_orig.com', realpath(out_dir))  
    imodscript('tilt_orig.com', realpath(out_dir))  
#        check_output("clip rotx %s %s" % (tiny_orig_full, tiny_orig), shell = True)

#deprecate copy_orig
#    else:
#        if not isfile(copy_orig):
#            os.symlink(copy_orig, tiny_orig_full)

    #make flexo tomo:
    fp = read_comfiles(out_dir)
    format_newst(base_name, out_dir, fp[0], fp[1], out_bin, tomo_size,
                 bin_flexo_ali)
    format_tilt(base_name, out_dir, bin_flexo_ali, fp[3], out_bin, op[4],
                fp[5], fp[8], fp[9], fp[6], fp[12], fp[13], fp[7],
                out_full_tomo, fakesirt)
    imodscript('newst.com', realpath(out_dir))  
    imodscript('tilt.com', realpath(out_dir))  
#    check_output("clip rotx %s %s" % (out_full_tomo, out_tomo), shell = True)
    
    
    
    #check that tomogram sizes match:
    apix, tiny_orig_size = get_apix_and_size(tiny_orig_full)
    apix, tiny_size = get_apix_and_size(out_full_tomo)
#    print 'match_tomos: tomogram size after binning %s' % tiny_size
    size_match = np.subtract(tiny_size, tiny_orig_size) != (0,  0, 0)
    if np.any(size_match):
        if np.any(size_match > 2):
            raise ValueError
        else:
            tiny_size = tiny_orig_size  
    #read into memory and rotate around x
    ref = deepcopy(mrcfile.open(out_full_tomo).data)
    ref = np.rot90(ref, k = 1, axes = (0,1))
    query = deepcopy(mrcfile.open(tiny_orig_full).data)
    query = np.rot90(query, k = 1, axes = (0,1))
    #check rotation
    add_yrot, add_xtilt = get_tomo_transform(ref, query, angrange, 'rotation',
                                             how_tiny, out_dir)
    OFFSET = np.round((fp[8] - add_yrot), decimals = 1)
    global_xtilt = np.round(( float(fp[5]) - add_xtilt), decimals = 1)
    
    
    #re-make flexo tomo with adjusted rotations:
    format_tilt(base_name, out_dir, bin_flexo_ali, fp[3], out_bin, op[4],
                global_xtilt, OFFSET, fp[9], fp[6], fp[12], fp[13], fp[7],
                out_full_tomo, fakesirt)
    imodscript('newst.com', realpath(out_dir))  
    imodscript('tilt.com', realpath(out_dir))  
#    check_output("clip rotx %s %s" % (out_full_tomo, out_tomo), shell = True)
    ref = deepcopy(mrcfile.open(out_full_tomo).data)
    ref = np.rot90(ref, k = 1, axes = (0,1))
    
    gshift = get_tomo_transform(ref, query, angrange, 'translation', how_tiny,
                                out_dir)
    SHIFT = fp[9] - gshift*tomo_binning

    print('first round offset, xtilt, shift %s %s %s' % (OFFSET, global_xtilt, SHIFT))

    #check rotation and shift again
    format_tilt(base_name, out_dir, bin_flexo_ali, fp[3], out_bin, op[4],
                global_xtilt, OFFSET, SHIFT, fp[6], fp[12], fp[13], fp[7],
                out_full_tomo, fakesirt)
    imodscript('newst.com', realpath(out_dir))  
    imodscript('tilt.com', realpath(out_dir))  
#    check_output("clip rotx %s %s" % (out_full_tomo, out_tomo), shell = True)
    ref = deepcopy(mrcfile.open(out_full_tomo).data)
    ref = np.rot90(ref, k = 1, axes = (0,1))

    add_yrot, add_xtilt = get_tomo_transform(ref, query, angrange, 'rotation',
                                             how_tiny, out_dir)
    OFFSET = np.round((OFFSET - add_yrot), decimals = 1)
    global_xtilt = np.round((global_xtilt - add_xtilt), decimals = 1)
    gshift = get_tomo_transform(ref, query, angrange, 'translation', how_tiny,
                                out_dir)
    SHIFT = SHIFT - gshift*tomo_binning 
    print('second round offset, xtilt, shift %s %s %s' % (OFFSET, global_xtilt, SHIFT))

    
    print('Calculated shift at bin %s: X %s, Z %s' % (out_bin, SHIFT[0]/out_bin,
                                           SHIFT[1]/out_bin))
    print('Calculated tilt angle and xtilt offset %s, %s' % (
                                            add_yrot, add_xtilt))
    
    return SHIFT, OFFSET, global_xtilt


#############################################################################

def imodscript(comfile, abspath):
    """Executes IMOD comfile but ignores "sys.exit" that's executed after successful completion.
    """
    comfile=comfile.split('.')[0]
    check_output('vmstopy '+str(abspath)+'/'+str(comfile)+'.com '+str(abspath)+'/'+str(comfile)+'.log '+str(abspath)+'/'+str(comfile)+'.py',shell=True)
    pwd=os.getcwd()
    with open(str(abspath)+'/'+str(comfile)+'.py','r')as f:
        lines=[]
        for x in f.readlines():
            if x=="    prnstr('SUCCESSFULLY COMPLETED', file=log)\n":
                lines.append(str('    pass ##')+x)
            elif x=='  sys.exit(exitCode)\n':
                lines.append(str('##')+x)
            elif x=='  log.close()\n':
                lines.append(str('##')+x)
            else:
                lines.append(x)
    with open(str(abspath)+'/'+str(comfile)+'.py','w')as f:
        for x in lines:
            f.writelines(x)
    
    sys.path.append(abspath)  #adds folder to path, can then execute python script as usual
    os.chdir(abspath)
    exec(compile(open(str(abspath)+'/'+str(comfile)+'.py').read(), str(abspath)+'/'+str(comfile)+'.py', 'exec'))
    os.chdir(pwd)
    sys.path.pop(-1) #removes last path entry

##############################################################################
def write_to_log(log, s, debug = 1):
    if debug > 0:
        if s != '' or s != '\n':
        #I would rather not print empty lines
            f = open(log, 'a+')
            f.write(s + '\n')
            f.close()

def progress_bar(total, prog):
    """
    Command line progress bar.
    
    """
    bl, status = 40, ""
    prog = float(prog)/float(total)
    if prog >= 1.:
        prog, status = 1., "\r\n"
    block = int(round(bl * prog))
    text = "\r[{}] {}% {}".format("=" * block + " " * (bl - block),
               np.round((prog*100), decimals = 0),status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def kill_process(process, log = False, wait = 10, sig = signal.SIGTERM):
    """
    Attempt to find and terminate a process group relating to process.pid().
    Else attempt to terminate the process.pid(). 
    """
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, sig)   
        print(('\nKilling processchunks group PID %s.' % pgid))
        if log:
            com = process.communicate()
            write_to_log(log, com[0].decode() + '\n' + com[1].decode() + 
                     'run_split_peet: Terminating group PID %s\n' % pgid)  
    except OSError:
    #os.getpgid should return OSError if group doesnt exist (e.g. single PID)
        try:
            os.kill(process.pid, sig)
            print(('\nKilling processchunks PID %s.' % pgid))
            if log:
                com = process.communicate()
                write_to_log(log, com[0] + '\n'.encode() + com[1] + 
                    'run_split_peet: Terminating PID %s\n'.encode() % process.pid)  
        except:
            print(('Unable to terminate process %s: No such process.'
                  % process.pid))
    except:
        print('killpg: Unhandled Exception.')
        raise
    

def run_generic_process(cmd, out_log = False, wait = True):
    """
    Run a shell command.
    Inputs:
        cmd [list of strings]
    Optional:
        out_log [str] path to output log.  Default False
        wait [bool] wait for process to finish. Default True
    """
    #after writing this I realised imod does something very similar 
    #with vmstopy...   
    
    if not isinstance(cmd, list):
        raise TypeError('run_generic_process: cmd must be of type list')
    try:
        process = Popen(cmd, stdout = PIPE, stderr = PIPE, 
                        preexec_fn=os.setsid)
        if out_log:
            if isfile(out_log):
                os.rename(out_log, out_log + '~')
            write_to_log(out_log, (' ').join(cmd) + '\n')
            for line in iter(process.stdout.readline, ''.encode()):
                # line = line.decode()
                # if line == '':
                #     break
                # else:
                write_to_log(out_log, line.decode().strip())
            com = process.communicate()                
            write_to_log(out_log, com[0].decode() + '\n' + com[1].decode())

            if process.poll() != 0:
                raise ValueError(('run_generic_process: Process' +
                            ' returned non-zero status.  See %s') % out_log)
        elif wait:
            com = process.communicate()
            p = process.poll()
            if p != 0:
                out_log = realpath('run_generic_process_error.log')
                #com = process.communicate()
                write_to_log(out_log, (' ').join(cmd))
                write_to_log(out_log, com[0].decode() + '\n' + com[1].decode())
                raise ValueError(('run_generic_process: Process returned' +
                                 ' non-zero status.  See %s') % out_log)
        else:
            print(('run_generic_process: WARNING: ' +
                   'Child process errors will not be caught.'))
    except ValueError:
        raise
    except KeyboardInterrupt:
        kill_process(process, out_log)      
        raise
    except:
        print('run_generic_process: Unhandled Exception.')
        kill_process(process)
        com = process.communicate()
        print(com[0] + '\n'.encode() + com[1])
        raise

def run_processchunks(base_name, out_dir, machines, log = False):   
    """
    Run com script(s) using processchunks
    Inputs:
        base_name [str] base name of com files 
        out_dir [str] path to directory containing com files
        machines [list of str] machine names for running, e.g. ['citra']*2
    Optional:
       logs [tuple of strings] names of output log files.  
           Default (False, False)
    """ 
    pwd = os.getcwd()
    os.chdir(out_dir)
    
    if not isinstance(machines, list):
        machines = [machines]
    try:
        cmd = ['/bin/sh', 'processchunks', '-g', '-n', '18', '-P', 
               (',').join(machines), base_name]
        process = Popen(cmd, stdout = PIPE, stderr = PIPE,
                        preexec_fn=os.setsid)
        if not log:
            c_log = join(out_dir, 'processchunks.out')
        else:
            c_log = realpath(log)
        if isfile(c_log):
            os.rename(c_log, c_log + '~')
        write_to_log(c_log, out_dir + '\n' + (' ').join(cmd) + '\n')

        total_chunks, chunks_done = 0, 0
        for line in iter(process.stdout.readline, ''.encode()):
            line = line.decode()
            # if line == '':
            #     break
            # else:
            write_to_log(c_log, line.strip())
            if line.split()[3:6] == ['DONE', 'SO', 'FAR']:
                total_chunks = int(line.split()[2])
                chunks_done = int(line.split()[0])
                progress_bar(total_chunks, chunks_done)
        com = process.communicate()
        write_to_log(c_log, com[0].decode() + '\n' + com[1].decode())
        if process.poll() != 0:
            raise ValueError(('run_processchunks: Process' +
                            ' returned non-zero status.  See %s') % c_log)
    except ValueError:
        raise
    except:
        kill_process(process, log = c_log)
        raise
    else:
        if total_chunks != chunks_done or chunks_done == 0:        
            raise Exception('Processchunks did not run to completion.') 
    os.chdir(pwd)

def run_split_peet(base_name, out_dir, base_name2, out_dir2, machines,
                   logs = (False, False)):   
    """
    Run 2 sets of com scripts in parallel using processchunks.
    Inputs:
        base_name [str] base name of com files 
        out_dir [str] path to directory containing com files
        base_name2 [str] base name of com files 
        out_dir2 [str] path to directory containing com files
        machines [list of str] machine names for running, e.g. ['citra']*2
    Optional:
       logs [tuple of strings] names of output log files.  
           Default (False, False)
    """   
    pwd = os.getcwd()
    if not isinstance(machines, list):
        machines = [machines]
    if len(machines) < 2:
        print(('run_split_peet: Warning: Only one machine specified. '+
               'Both processchunks will be sent to the same core.'))
        m1 = m2 = machines
    else:
        m1 = machines[:len(machines)//2]
        m2 = machines[len(machines)//2:]
    if not all(logs):
        c_log1 = join(out_dir, 'processchunks.out')
        c_log2 = join(out_dir2, 'processchunks.out')
    else:
        c_log1, c_log2 = realpath(logs[0]), realpath(logs[1])
    if isfile(c_log1):
        os.rename(c_log1, c_log1 + '~')
    if isfile(c_log2):
        os.rename(c_log2, c_log2 + '~')
    try:
        processchunks_terminated = 0
        #process 1
        os.chdir(out_dir)  
        cmd = ['/bin/sh', 'processchunks', '-g', '-n', '18', '-P', 
               (',').join(m1), base_name]
                    #%s %s'
                     #%((',').join(m1), base_name))
        process = Popen(cmd, stdout = PIPE, stderr = PIPE, 
                        preexec_fn=os.setsid)
        write_to_log(c_log1, out_dir + '\n' + (' ').join(cmd) + '\n')
        #process2
        os.chdir(out_dir2)  
        cmd2 = ['/bin/sh', 'processchunks', '-g', '-n', '18', '-P', 
               (',').join(m2), base_name2]
                     #%((',').join(m2), base_name2))
        process2 = Popen(cmd2, stdout = PIPE, stderr = PIPE,
                         preexec_fn=os.setsid)
        write_to_log(c_log2, out_dir + '\n' + (' ').join(cmd2) + '\n')
          
        total_chunks1, total_chunks2 = 0, 0
        chunks_done1, chunks_done2 = 0, 0
        for output1, output2 in zip_longest(
                        iter(process.stdout.readline, ''.encode()),
                        iter(process2.stdout.readline, ''.encode())):
            advance_bar = False
            if output1 != None:
                output1 = output1.decode()
                write_to_log(c_log1, output1.strip())
                if output1.split()[3:6] == ['DONE', 'SO', 'FAR']:
                    total_chunks1 = max(int(output1.split()[2]), total_chunks1)
                    chunks_done1 = max(int(output1.split()[0]), chunks_done1)
                    advance_bar = True
                elif output1.split()[:2] == ['ALL', 'DONE']:
                    total_chunks1 = chunks_done2 = total_chunks1
                    advance_bar = True                    
            if output2 != None:
                output2 = output2.decode()
                write_to_log(c_log2, output2.strip())
                if output2.split()[3:6] == ['DONE', 'SO', 'FAR']:
                    total_chunks2 = max(int(output2.split()[2]), total_chunks2)
                    chunks_done2 = max(int(output2.split()[0]), chunks_done2)
                    advance_bar = True
                elif output2.split()[:2] == ['ALL', 'DONE']:
                    total_chunks2 = chunks_done2 = total_chunks2
                    advance_bar = True                    
            if advance_bar:
                total_chunks = total_chunks1 + total_chunks2
                chunks_done = chunks_done1 + chunks_done2
                progress_bar(total_chunks, chunks_done)   
            panic_msg = (('').join(['#']*30) + '\n' +
                         'SOMETHING HAS GONE WRONG, DUMPING STDERR, STDOUT:\n')
            if process.poll() != None and output1 == None:
                #have to check if there is still output in the PIPE because
                #of race condition with process.poll()
                #some processchunks errors return 0 status:
                #when done, check return status but also if chunks are done
                if (process.poll() != 0
                or (process.poll() == 0 and chunks_done1 != total_chunks1)
                or (process.poll() == 0 and chunks_done1 == 0)):                    
                    com = process.communicate()
                    write_to_log(c_log1, 
                                 'Processchunks status %s' % process.poll())
                    write_to_log(c_log1, panic_msg)
                    write_to_log(c_log1, com[0] + '\n' + com[1])
                    if process2.poll() == None: 
                        kill_process(process2, log = c_log2)
                    processchunks_terminated = 1
                    raise ValueError('Processchunks returned non-zero status.')
            if process2.poll() != None and output2 == None:
                if (process2.poll() != 0
                or (process2.poll() == 0 and chunks_done2 != total_chunks2)
                or (process2.poll() == 0 and chunks_done2 == 0)):
                    com = process2.communicate()
                    print('#############%s' % str(com))                
                    write_to_log(c_log2, 
                                 'Processchunks status %s' % process2.poll())
                    write_to_log(c_log2, panic_msg)
                    write_to_log(c_log2, com[0] + '\n' + com[1])                    
                    if process.poll() == None:
                        kill_process(process, log = c_log1)
                    processchunks_terminated = 1
                    raise ValueError('Processchunks returned non-zero status.')
    except ValueError:
        if processchunks_terminated == 1:
            raise
        else:
            kill_process(process, log = c_log1)
            kill_process(process2, log = c_log2)
            raise
    except KeyboardInterrupt:
        kill_process(process, log = c_log1)
        kill_process(process2, log = c_log2)        
        raise
    except:
        print('run_split_peet: Unhandled Exception.')
        kill_process(process, log = c_log1)
        kill_process(process2, log = c_log2)    
        raise
    else:
        os.chdir(pwd)
    


##############################################################################

def in_plane_rotation(ref, query, angrange, out_dir, limit = 10, interp = 1,
                      order = 3, refine = True, edge = 3, mode = 'nearest'):
    
    #interp never used#
    
    """
    input 2d
    angrange should be even
    decimal
    edge removes 
    
    """
    def rotate_and_cc(angles):
        
        mat = np.zeros(len(angles))
        for x in range(len(angles)):
            r = rotate(ref, angles[x], (0,1), reshape = False, order = order)#, mode = mode)
            if edge:
                r = r[edge:-edge, edge:-edge]
            cc = ncc(query, r, limit, 1, outfile=False)
            mat[x] = cc.max()              
        fine = np.linspace(angles[0], angles[-1], 101)
        intp = interp1d(angles, mat, kind = 2)
        fintp = intp(fine)

        pkpos, pkh = find_peaks(fintp, np.min(fintp))
        pkh = pkh['peak_heights']

        if pkh.size == 0:
            bestphi = np.nan
            subdegree = False
        else:
            bestphi = fine[pkpos[np.argmax(pkh)]]
        return bestphi

    if angrange%2 != 0:
        angrange += 1
    angles = np.arange(-angrange//2, angrange//2 + 1)    
    #trimming helps with edge artefacts due to rotation
    if edge:
        query = query[edge:-edge, edge:-edge]
  
    bestphi = rotate_and_cc(angles)
    if refine:
        angles = np.linspace(bestphi - 0.5, bestphi + 0.5, 11)
        bestphi = rotate_and_cc(angles)
    return np.round(bestphi, decimals = 2)

def deal_with_excludelist(ali, tlt, defocus_file, xf, excludelist, 
                          base_name, rec_dir, out_dir):
    """ 
    excludelist is numbered from one!!!!!!!
    remove tilts based on excludelist
    all inputs should be optional...meaning that this will work even if
    only one input is specified... provided that it is at the correct position
    """
    
    #etomo comfiles have excludelist from 1. 
    excludelist = np.array(excludelist) - 1
    
    #aligned stack
    if ali:
        ali_path, ali_name = split(ali)
        new_ali = join(os.path.realpath(out_dir), 'excludelist_' + ali_name)
        check_output('newstack -input %s -output %s -exclude %s' % (
                ali, new_ali, (',').join([str(x) for x in excludelist])
                ), shell = True)   
    else:
        new_ali = False
    #tilt_angles
    if tlt:
        tilt_angles = [float(x.strip()) for x in open(tlt, 'r')]
        new_tilt_angles = [float(x.strip()) for x in open(tlt, 'r')]
        for x in excludelist:
            remove_angle = tilt_angles[x]
            new_tilt_angles.remove(remove_angle)
        new_tilt_angles = np.array(new_tilt_angles)    
        new_tlt = os.path.join(out_dir, base_name + '_excludelist.tlt')
        with open(new_tlt, 'w') as f:
            for x in new_tilt_angles:
                f.write(str(x) + '\n')
    else:
        new_tlt = False
    #xf
    if xf:
        new_xf = join(out_dir, base_name + '_excludelist.xf')
        with open(xf, 'r') as l:
            lines = []
            for x in l:
                lines.append(x)
            for x in excludelist:
                lines.pop(x)
            with open(new_xf, 'w') as l:
                for x in range(len(lines)):
                    l.write(lines[x])
    else:
        new_xf = False
    
    #defocus file
    if defocus_file:
        new_defocus_file = join(out_dir, base_name + '_excludelist.defocus')
        ex_bool = np.isin(np.array(list(range(len(tilt_angles)))), np.array(excludelist),
                          invert = True)
        with open(defocus_file, 'r') as ff:
            first_line = ff.readlines()[0]
        if any([a == 'CTFFind' for a in first_line.split()]):
            defocus_file = convert_ctffind_to_defocus(defocus_file, base_name,
                                                      rec_dir, out_dir)
        if str(first_line.split()[-1]) == '2':
            # defocus file format version 2
            df=[]        
            with open(defocus_file, 'r') as f:
                df = f.readlines()
                df = np.array(df)[ex_bool]
                if np.any(np.array(excludelist) == 0):
                    df[0] = ('\t').join(df[0].split()) + '\t2\n'
            with open(new_defocus_file, 'w') as f:
                for x in range(len(df)):
                    df[x] = '%s\t%s\t' % (x + 1, x + 1) + ('\t').join(df[x].split()[2:]) + '\n'
                    f.write(str(df[x]))
        else:
            df=[]
            #read defocus file (etomo astigmatism format)
            print('STILL UNTESTED!!!!')
            print('!!!!!!!!!!!!!!!!!!!')
            with open(defocus_file, 'r') as f:
                df = f.readlines()
                fl = df.pop(0) #pop
                df = np.array(df)[ex_bool]
            with open(new_defocus_file, 'w') as f:
                f.write(fl)
                for x in range(len(df)):
                    df[x] = ('%s\t%s\t' % (-x + len(df), -x + len(df))
                            + ('\t').join(df[x].split()[2:]) + '\n')
                    f.write(str(df[x]))
    else:
        new_defocus_file = False
    return new_ali, new_tlt, new_defocus_file, new_xf

'bfcc -side 1 -angles 0,0,0 -View 0,0,0,0 -Mask mw.mrc -verbose 1 -mode directional -limit 0 -shiftlimit 1 -Template 16.rec 16.rec'

def check_tmpfs(chunk_id, vols, tmpfsdir = '/dev/shm'):
    """Evaluate (return bool) whether a chunk will have enough space to
    write to /dev/shm.  
    YOU WILL HAVE TO CLEAN UP cunk_count-%03d.txt % chunk_id FILES AT 
    THE END OF EACH CHUNK!!!!
    
    chunk_id [int] unique chunk identifier
    vols [int] required space in bytes
    """            
    #import psutil
    #import os
    #import glob
    #import numpy as np
                
    if not os.path.isdir(tmpfsdir):
        os.makedirs(tmpfsdir)
    #free = psutil.disk_usage(tmpfsdir)[2]
    free = int(check_output(['df', tmpfsdir]).decode().split()[-3])
    #assuming 32bit, mask file should have the same size as each tomo

    #how many chunks can write into tmpfs at the same time
    nposs = int(np.floor(free/vols))

    #open empty text file to count how many chunks are trying to write
    #to /dev/shm
    chunk_counter = (tmpfsdir + '/chunk_count-%03d.txt' % chunk_id)
    a = open(chunk_counter, 'w')
    a.close()
    #check how many chunks in total are trying to use tmpfs
    chunk_count = glob.glob((tmpfsdir + '/chunk_count*.txt'))
    if len(chunk_count) > nposs:
        cids = np.sort([int(y[-7:-4]) for y in chunk_count])
        if np.where(cids == chunk_id)[0][0] < nposs:
            avail = True
        else:
            avail = False
    else:
        avail = True
    return avail


def orthogonal_tilts(tlt, n_orth = 6, d_orth = 90, excludelist = False,
                     return_mask = False):
    """
    For each angle, returns a list of angles {n_orth} angles long that are
    at least {d_ort} degrees away, plus the current angle. 
    E.g. if input angles are [-60, 27, 30, 33, 36], d_orth =  90 and n_orth = 3,
    the first list (out of 5) will be [-60, 30, 33, 36].
    Inputs
    d_orth [int] desired angular distance from query projection
    n_orth [int] number of projections to be used (plus query)
    """
    #exlude highest angles?
    

    tilt_angles = np.array([float(x.strip('\n ')) for x in open(tlt)])

    if not isinstance(excludelist, bool):
        excludelist = np.array(excludelist) - 1
        #remove tilts based on excludelist
        exc_mask = np.isin(list(range(len(tilt_angles))), excludelist, invert = True)
        tilt_angles = np.array(tilt_angles)[exc_mask]

    #the following code picks {n_orth} angles AROUND the orthogonal tilt 
    #(meaning the tilt that is {d_orth} degrees away from the current tilt)
    #adjust {d_orth} so that closest of the list of orthogonal angles
    #is actually {d_orth} degrees away
    diff = int(np.round(np.absolute(np.mean(np.diff(tilt_angles)))))
    d_orth += (n_orth//2) * diff

    #move all angles to positive range to simplify the math
    ta = tilt_angles + 90
    #direction of tilt series: positive values first?
    pfirst = (ta[0] - ta[-1] > 0)
    #reorder angles: min to max
    if pfirst:
        ta = np.sort(ta)
    
    angset = []
    mask = []
    for x in range(len(ta)):
        #get distances from x
        d = np.absolute(ta - ta[x])
        #sort by distances (furthest first)
        o = np.flip(np.argsort(d), axis = 0)
        
        #if there are enough tilts around the orthogonal tilt (meaning tilt 
        #{d_orth} degrees away from current tilt):
        #pick {n_orth}/2 on both sides of the orthogonal tilt
        #else pick {n_orth} most distant tilts

        #position of tilt {d_orth} degrees away
        n = find_nearest(d, d_orth)
        #position after sorting
        q = np.where(o == n)[0][0]

        #if the orthogonal tilt is within last {n_orth} tilts, just pick the 
        #{n_orth} most distant tilts
        if q < n_orth//2:
            orth_ind = o[:n_orth]
        #otherwise pick tilts on both side of the orthogonal tilt.
        else:
            orth_ind = o[q - n_orth//2 - n_orth%2 + 1:q + n_orth//2 + 1]
        orth_ind = np.hstack((orth_ind, x))
        orth_mask = np.isin(list(range(len(tilt_angles))), orth_ind)
        angset.append(tilt_angles[orth_mask])
        mask.append(orth_mask)

    if return_mask:
        return np.array(angset), np.array(mask)
    else:
        return np.array(angset)

        
def rec_orth_tomos(
        base_name,
        out_dir,
        tlt,
        ali,
        tomo_binning,
        thickness,
        mask,
        machines,
        nvlp_id = False,
        global_xtilt = 0,
        OFFSET = False,
        SHIFT = False,
        xfile = False,
        localxf = False,
        zfac = False,
        excludelist = False,
        fakesirt = 0,
        n_orth = 6,
        d_orth = 60,
        tmpfs = False):
    
    #default
    combine_comfiles = True


#    startTime = time.time()  
    tilt_angles = [float(x.strip('\n\r').strip(' ')) for x in open(tlt)]
        
    trd = join(out_dir, 'orthogonal_rec')
    if not os.path.isdir(trd):
        os.makedirs(trd)
    #generate tilt sets
    orth_ta = orthogonal_tilts(tlt, n_orth, d_orth, excludelist)
    
    #global excludelist mask
    if not isinstance(excludelist, bool):
        excludelist = np.array(excludelist) - 1
        #remove tilts based on excludelist
        gem = np.isin(list(range(len(tilt_angles))), excludelist, invert = True)
    else:
        gem = np.ones(len(tilt_angles), dtype = bool)

#
#    if tmpfs:
#        #when tmpfs space is limiting, decide which chunks go to tmpfs and 
#        #which are written out.  Since chunks are executed in order, a very 
#        #basic solution is to send every Nth chunk to memory where 
#        #N = ceiling({num cores}/{num cores that can fit in tmpfs}
#        mem_or_hd = int(np.ceil(len(machines)/float(nposs)))
#        mem_or_hd = range(len(orth_ta))[::mem_or_hd]
#        print mem_or_hd
    repali_list = []
    if tmpfs:
        path = (',').join(['"' + y + '"' for y in sys.path if y != ''])
        tmpfsdir = join('/dev/shm/', base_name + '_flexo')
    
    for x in range(len(orth_ta)):

        if tmpfs:    
            stmt = (
                    '>sys.path.extend([%s])' % path,
                    '>from definite_functions_for_flexo import check_tmpfs',
                    '>chunk_id = %s' % x,
                    '>vols = %s' % (os.stat(mask)[6]*2),
                    '>tmpfs = check_tmpfs(chunk_id, vols)',
                    '>tmpfsdir = "%s"' % tmpfsdir,
                    '>if not os.path.isdir(tmpfsdir):',
                    '>\tos.makedirs(tmpfsdir)',
                    '>if tmpfs:',
                    '>\trec_n = "%s"' % join(tmpfsdir,
                                         base_name + '_%03d.rec' % x),
                    '>\tmrec_n = "%s"' % join(tmpfsdir,
                                          base_name + '_m_%03d.rec' % x),
                    '>else:',
                    '>\trec_n = "%s"' % join(trd,
                                           base_name + '_%03d.rec' % x),
                    '>\tmrec_n = "%s"' % join(trd,
                                            base_name + '_m_%03d.rec' % x),
                    )
            #these need to be variables as each chunk decides whether
            #to write to tmpfs or disk
            rec_n = '%rec_n'
            mrec_n =  '%mrec_n'
        else:
            rec_n = join(trd, base_name + '_%03d.rec' % x)
            mrec_n = join(trd, base_name + '_m_%03d.rec' % x)
        #always reproject to disk
        repali_n = join(trd, base_name + '_%03d.ali' % x)
        maskcom = join(trd, 'clipmask-%03d.com' % x)
        comb_com = join(trd, 'orth-%03d.com' % x)
        repali_list.append(repali_n)
#        with open(tlt_n, 'w') as f:
#            for y in orth_ta[x]:
#                f.write('%s\n' % y)
            
        #need excludelist for tilt, combine with existin excludelist     
        exc_mask = np.isin(tilt_angles, orth_ta[x])
        exc_mask = np.logical_and(gem, exc_mask)
        #excludelist numbered from 1
        excludelist = (np.arange(len(tilt_angles)) + 1)[np.logical_not(
                                                                    exc_mask)]

        #reconstruction .com
        format_tilt(base_name, trd, ali, tlt, tomo_binning, thickness,
                    global_xtilt, OFFSET, SHIFT, xfile, localxf, zfac,
                    excludelist, rec_n, fakesirt,
                    com_ext = '-%03d' % x)
        #reprojection .com
        format_tilt(base_name, trd, ali, tlt, tomo_binning, thickness,
                    global_xtilt, OFFSET, SHIFT, xfile, localxf, zfac,
                    excludelist, rec_n, fakesirt,
                    '_rep-%03d' % x,
                    orth_ta[x],
                    mrec_n,
                    repali_n)

        #format masking comfiles
        with open(maskcom, 'w') as f:
            f.write('$clip multiply %s %s %s\n' % (mask, rec_n, mrec_n))
            f.write('$rm %s' % (rec_n))
        
        if combine_comfiles:
            #combine comfiles
            #This is True by default - the alternative would mean waiting
            #for all orth tomos to be reconstructed before they could be
            #masked, before these could be reprojected etc.
            cparts = [join(trd, 'tilt-%03d.com' % x),
                      maskcom,
                      join(trd, 'tilt_rep-%03d.com' % x)]
            with open(comb_com, 'w') as f:
                if tmpfs:
                    f.write(('\n').join(stmt))
                    f.write('\n')
                for y in range(len(cparts)):
                    with open(cparts[y]) as g:
                        f.write(g.read() + '\n\n')
                f.write('$rm %s\n' % (mrec_n))
                if tmpfs:
                    f.write('$rm %s\n' % ('/dev/shm/chunk_count-%03d.txt' % x))
                    f.write(('>if not os.listdir("%s") and' + 
                            ' glob.glob("/dev/shm/chunk_count*.txt") == []:\n')
                            % tmpfsdir)
                    f.write('$  rm -r %s' % tmpfsdir)


    #format filein list for newstack to extract query tilts
    nf = join(trd, base_name + '_orth_sync.txt')
    if isinstance(nvlp_id, bool):
        orth_ali = join(out_dir, base_name + '_orth_sub.ali')
    else:
        orth_ali = join(out_dir, 'subtracted_%02d.mrc' % nvlp_id)
        
    exc_ta = np.array(tilt_angles)[gem]
    with open(nf, 'w') as f:
        f.write('%s\n' % len(orth_ta))
        for x in range(len(orth_ta)):
            #repali_n = join(trd, base_name + '_%03d.ali' % x)
            f.write('%s\n' % repali_list[x])
            ind = np.where(orth_ta[x] == exc_ta[x])[0][0]
            f.write('%s\n' % ind)

    #combining comfiles matters when tmpfs == True
    if combine_comfiles:
        run_processchunks('orth', trd, machines)
    else:
        run_processchunks('tilt', trd, machines)
        run_processchunks('clipmask', trd, machines)
        run_processchunks('tilt_rep', trd, machines)

    if tmpfs:
        if os.path.isdir(tmpfsdir):
            #this doesn't work on remote machines...
            check_output('rm -r %s' % tmpfsdir, shell = True)

    check_output('newstack -filei %s -output %s' % (nf, orth_ali),
                 shell = True)
#    print 'Orthogonal subtraction execution time: %s s.' % int(np.round(
#                                (time.time() - startTime ), decimals = 0))
    
def reconstruct_binned_tomo(out_dir, base_name, binning, st, output_xf,
                            output_tlt, thickness, global_xtilt, SHIFT,
                            xfile, localxf, zfac, excludelist, defocus_file,
                            V, Cs, ampC, apix,
                            tomo_size,
                            deftol = 200,
                            interp_w = 4,
                            n_tilts = False,
                            machines = False,
                            no_ctf_convolution = False):
    """
    Purely for readability of flexo_processchunks.py
    Creates comscripts with _bin[binning].com extension so the original
    scripts are not overwritten.
    """
    pwd = os.getcwd()
    os.chdir(out_dir)
    output_rec = abspath(join(out_dir,
                              base_name + '_bin%s_full.rec' % binning))            
    out_tomo = join(out_dir, base_name + '_bin%s.rec' % binning)
    com_ext = '_bin%s' % int(binning)
    output_ali = abspath(join(out_dir, base_name + '_bin%s.ali' % int(binning)))
    
    #format comscripts
    output_ali = format_newst(base_name, out_dir, st, output_xf,
                              binning, tomo_size, output_ali, com_ext)
    #OFFSET has to be 0 in tlt.com if it's specified in align.com !
    output_rec = format_tilt(base_name, out_dir, output_ali, output_tlt,
                          binning, thickness, global_xtilt, 0, SHIFT,
                         xfile, localxf, zfac, excludelist,
                         output_rec,
                         com_ext = com_ext) 

    ctf_ali = format_ctfcorr(output_ali, output_tlt, output_xf, 
                             defocus_file, out_dir, base_name, V, Cs, ampC, 
                             apix,
                             deftol,
                             #to be added to inputs:
                             interp_w)
  
    #make ts
    imodscript('newst_bin%s.com' % int(binning), os.path.realpath(out_dir))
    #ctfcorrect
    if no_ctf_convolution:
        ctf_ali = output_ali
    else:
        warnings.warn('Ctfcorrection disabled!')
        check_output('splitcorrection -m %s ctfcorrection.com' % 
                     int(np.floor(n_tilts/len(machines))),
                     shell = True)
        run_processchunks('ctfcorrection', out_dir, machines)
        os.rename(ctf_ali, output_ali)
    #reconstruct
    check_output('splittilt -n %s tilt_bin%s.com' % (len(machines),
                            int(binning)), shell = True)
    run_processchunks('tilt_bin%s' % int(binning), out_dir, machines)
    check_output("clip rotx %s %s" % (output_rec, out_tomo), shell = True)
    
    os.chdir(pwd)
    
    return output_rec, out_tomo, output_ali

def get_tiltrange(tiltlog):
    with open(tiltlog) as f:
        aa = f.readlines()
    aline = False
    alist = []
    for x in range(len(aa)):    
        if aa[x] == ' Projection angles:\n':
            aline = x + 2
        if not isinstance(aline, bool):
            if x >= aline:
                if aa[x] == '\n' and aa[x + 1] == '\n':
                    break
                else:
                    alist.append(aa[x])
    trange = [float(alist[0].split()[0]), float(alist[-1].split()[-1].strip())]
    return trange

def check_refpath(path):
    """
    Returns absolute path or raises error if file doesnt exist.
    
    When generating prm file using e.g. remdup the relative
    paths do not get updated.  The reference/mask are often ../ 
    or ../../
    
    """
    if os.path.isabs(path):
        if not isfile(path):
            raise Exception('File for %s not found.' % join(path))
    else:
        if isfile(path):
            path = realpath(path)
        elif isfile('../' + path):
            path = realpath('../' + path)
        elif isfile('../../' + path):
            path = realpath('../../' + path)
        else:
            raise Exception('File for %s not found.' % join(path))
    return path
    
def prepare_prm(prm, ite, tom_n, out_dir, base_name, st, new_prm_dir,
                        search_rad = False,
                        phimax_step = False,
                        psimax_step = False,
                        thetamax_step = False,
                        hicutoff = False, #expect list/tuple
                        lowcutoff = False, #expect list/tuple
                        refthr = False,
                        tomo = False #if False, default naming used
                        ):
    """
    This function is inteded to prepare a set of PRM files for FSC
    determination in order to detect change in alignment of the Flexo 
    aligned tomogram.
    
    Specify either both half-data set PRM files or a single PRM file in 
    which case this file will be split in two.

    The alignment should already be pretty good.
    Inputs:
        prm [str] path to PEET parameter file
        ite [int] PEET iteration (uses PEETPRMFile)
        tom_n - [int or list] numbered from 1, use 'all' to include all
            This option is available in case there are e.g. multiple model
            files relating to the same tomogram.
        out_dir [str] path to output directory
        base_name [str] desired output base name
        st [str] path to original tilt series stack
        new_prm_dir [str] path to desired output directory
    Optional:
        search_rad [int, tuple or list of 3 ints] translation search radius
            default False
        phimax_step [list or tuple of 2 ints] phi rotation max and step
            default False
        psimax_step [list or tuple of 2 ints]
            default False
        thetamax_step [list or tuple of 2 ints]
            default False
        lowcutoff [list or tuple of 2 floats]
            default False
        hicutoff [list or tuple of 2 floats]
            default False
        lowcutoff [list or tuple of 2 floats]
            default False
        refthr [int] number of particles to be included in the average
            default False
        tomo [str, list, bool or 'init'] path to tomo(s)
            default False
    Returns:
        [numpy.ndarray] frequency at cutoff, resolution at cutoff,
                        FSC at nyquist
    """
    prmdir, prmname = os.path.split(prm)
    prm = PEETPRMFile(prm)
    motls = prm.get_MOTLs_from_ite(ite)   
    cwd = os.getcwd()
#    new_peet_dir = join(out_dir, 'peet')
    os.chdir(prmdir)
    if not os.path.isabs(motls[0]):
        for x in range(len(motls)):
            motls[x] = realpath(motls[x])
   
    modfiles = prm.prm_dict['fnModParticle']
    if not os.path.isabs(modfiles[0]):
        for x in range(len(modfiles)):
            modfiles[x] = realpath(modfiles[x])
            
    reference = prm.prm_dict['reference']
    reference = check_refpath(reference)
    mask = prm.prm_dict['maskType']
    if not np.isin(mask, ['none', 'sphere', 'cylinder']):
        mask = check_refpath(mask)

    #determine binning of tomo used for peet run, used for volume naming
    r_apix, r_size = get_apix_and_size(reference)
    st_apix, st_size = get_apix_and_size(st)
    #at what binning was the peet project actually run?
    peet_bin = int(np.round(r_apix/st_apix, decimals = 0))


    
    #which tomograms/mods/csvs should be used (all stored as list of strings)
    if isinstance(tom_n, str):
        if tom_n == 'all':
            tom_n = np.arange(len(modfiles))
        else:
            raise ValueError('prepare_prm: Unexptected tom_n value')  
    else:
        if isinstance(tom_n, int):
            if tom_n == 0:
                print(('prepare_prm: Tomograms for FSC are numbered from 1! '
                       + 'Using tomogram no. 1'))
                tom_n = [tom_n]
            tom_n = [tom_n - 1]
        else:
            if tom_n[0] == 0:
                print(('prepare_prm: Zero found for tomogram number. '
                       + 'Assuming entered numbering is from zero, not 1.'))
                tom_n = np.array(tom_n)
            else:                
                tom_n = np.array(tom_n) - 1


    mod = [modfiles[n] for n in tom_n]
    csv = [motls[n] for n in tom_n]
    
    if isinstance(tomo, bool):
        #The point of this is to run peet on flexo tomograms, so the default:
        tomo = [join(out_dir, base_name + '_bin%s.rec' % int(peet_bin))
                for m in range(len(mod))]
    elif isinstance(tomo, str):
        if tomo == 'init':
            tomo = prm.prm_dict['fnVolume']
        tomo = [realpath(tomo[tom_n[t]]) for t in range(len(tom_n))]          
    elif (isinstance(tomo, list) or isinstance(tomo, tuple)
            or isinstance(tomo, (np.ndarray, np.generic))):
        tomo = list(tomo)
        if len(tomo) != len(tom_n):       
            raise ValueError('prepare_prm: Number of tomogram paths does ' +
                             'not match the number of requested tomograms.')
    #tilt range - stored as list of lists
    #assuming Flexo tomogram is being used 
    trange = get_tiltrange(join(out_dir, 'tilt.log'))
    trange = [trange for x  in range(len(tomo))]
        

    new_prm = prm.deepcopy()
    new_prm.prm_dict['fnModParticle'] = mod
    new_prm.prm_dict['initMOTL'] = csv
    new_prm.prm_dict['fnVolume'] = tomo
    new_prm.prm_dict['tiltRange'] = trange
    new_prm.prm_dict['fnOutput'] = base_name
    new_prm.prm_dict['reference'] = reference
    new_prm.prm_dict['maskType'] = mask
    
    if isinstance(phimax_step, bool):
        phimax_step = 0, 1
    if isinstance(psimax_step, bool):
        psimax_step = 0, 1
    if isinstance(thetamax_step, bool):
        thetamax_step = 0, 1        
    angrange = ['-%s:%s:%s' % (phimax_step[0], phimax_step[1], phimax_step[0]),
                '-%s:%s:%s' % (psimax_step[0], psimax_step[1], psimax_step[0]),
                '-%s:%s:%s' % (thetamax_step[0],
                               thetamax_step[1], thetamax_step[0])]
    
    new_prm.prm_dict['dPhi'] = [str(angrange[0])]
    new_prm.prm_dict['dPsi'] = [str(angrange[1])]
    new_prm.prm_dict['dTheta'] = [str(angrange[2])]
           
        
    if isinstance(hicutoff, bool):    
        hicutoff = [0.45, np.round(max(1./r_size[0], 0.03), decimals = 3)]
    #list of lists
    try:
        new_prm.prm_dict['hiCutoff'] = [list(hicutoff)]
    except:
        raise ValueError ('prepare_prm: Unexpected hicutoff value')

    if isinstance(lowcutoff, bool):
        #Reduce to a single entry in case there were multiple iterations
        if len(new_prm.prm_dict['lowCutoff']) != 1:
            new_prm.prm_dict['lowCutoff'] = [
                                        new_prm.prm_dict['lowCutoff'][0]]
    else:
        if isinstance(lowcutoff, int):
            new_prm.prm_dict['lowCutoff'] = [0,0.05]
        else:
            raise ValueError('prepare_prm: Unexpected lowcutoff value.')
            #new_prm.prm_dict['lowCutoff'] = lowcutoff
    
    if isinstance(search_rad, bool):
        #default False to make it easier to work with defaults in higher order
        #functions
        #search_rad = [0, 0, 0]
        #it makes no sense for the default to be 0. If someone forgets to
        #set the search rad (like me) then the FSC output will be wrong
        search_rad = [r_size[0]/4]*3
    elif isinstance(search_rad, int):
        search_rad = [search_rad, search_rad, search_rad]
#    if len(search_rad) == 1:
#        search_rad = [search_rad, search_rad, search_rad]
    new_prm.prm_dict['searchRadius'] = ' {%s}' % search_rad
    
    if isinstance(refthr, bool):
        #Reduce to a single entry in case there were multiple iterations
        if len(new_prm.prm_dict['refThreshold']) != 1:
            new_prm.prm_dict['refThreshold'] = [
                                        new_prm.prm_dict['refThreshold'][0]]
    else:
        if isinstance(refthr, int):
            new_prm.prm_dict['refThreshold'] = [refthr]
        else:
            raise ValueError('prepare_prm: Unexpected refthr value.')
        
    #nParticles required for calcUnbiasedFSC
    #get number of particles by reading particle indices from last lines of 
    #csv files and adding them together
    num_p = 0
    for x in tom_n:
        num_p += int(check_output(['tail', '-1', '%s' % csv[x]]).decode().split(',')[3])
    new_prm.prm_dict['nParticles'] = num_p
    
    new_prmpath = join(new_prm_dir, base_name + '.prm') 
    new_prm.write_prm_file(new_prmpath)
            
    os.chdir(cwd)
    return peet_bin, new_prmpath, r_apix

def combined_fsc_halves(prm1, prm2, tom_n, out_dir, ite,
                        combine_all = False):
    """
    combines model files and motive lists split for fsc
    """
    def mod_and_csv(prm, ite = 2):
        """get motls and modfiles from prm, check if they exist and convert
        to abspath
        """
        prmdir, prmname = os.path.split(prm)
        if isinstance(prm, str):
            prm = PEETPRMFile(prm)
        motls = prm.get_MOTLs_from_ite(ite)
        if not isfile(motls[0]):
            os.chdir(prmdir)
        if not isfile(motls[0]):
            if os.path.isabs(motls[0]):
                tmotl = os.path.split(motls[0])[-1]
            else:
                tmotl = motls[0]
            tmotl = ('_').join(tmotl.split('_')[:-1]) + '_Iter*.csv'
            last_motl =  glob.glob(tmotl)[-1]
            ite = int(last_motl.split('_')[-1].strip('Iter.csv'))
            motls = prm.get_MOTLs_from_ite(ite)        

        if not os.path.isabs(motls[0]):
            for x in range(len(motls)):
                motls[x] = realpath(motls[x])

        modfiles = prm.prm_dict['fnModParticle']
        if not os.path.isabs(modfiles[0]):
            for x in range(len(modfiles)):
                modfiles[x] = realpath(modfiles[x])

        return motls, modfiles

    if not isdir(out_dir):
        os.makedirs(out_dir)

    motls1, modfiles1 = mod_and_csv(prm1, ite)
    motls2, modfiles2 = mod_and_csv(prm2, ite)

    new_mods = []
    new_motls = []
    if combine_all:
        tom_numbers = np.arange(len(modfiles1))
    else:
        tom_numbers = np.arange(tom_n -1, tom_n)
    for x in tom_numbers:
        fname = ('_').join(os.path.split(modfiles1[x])[1].split('_')[:-1])
        outmod = join(out_dir, fname + '_combined.mod')
        outcsv = join(out_dir, fname + '_combined.csv')
        new_mods.append(outmod)
        new_motls.append(outcsv)

        #read in csv halves and interleave
        csv1 = PEETMotiveList(motls1[x])
        csv2 = PEETMotiveList(motls2[x])
        new_arr = np.zeros((len(csv1) + len(csv2), 20))
        new_arr[::2] = csv1
        new_arr[1::2] = csv2
        #zero offsets
        new_arr[:, 10:13] = 0.

        #add to csv and renumber
        new_csv = PEETMotiveList()
        for y in range(len(new_arr)):
            new_csv.add_pcle(new_arr[y])
        new_csv.renumber()
        new_csv.write_PEET_motive_list(outcsv)
    
        #combine models
        mod1 = PEETmodel(modfiles1[x]).get_all_points()
        mod2 = PEETmodel(modfiles2[x]).get_all_points()
        mod1 += csv1.get_all_offsets()
        mod2 += csv2.get_all_offsets()
        new_arr = np.zeros((len(mod1) + len(mod1), 3))
        new_arr[::2] = mod1
        new_arr[1::2] = mod2
        new_mod = PEETmodel()
        for y in range(len(new_arr)):    
            new_mod.add_point(0, 0, new_arr[y])
        new_mod.write_model(outmod)
    
    if combine_all:
        return new_mods[tom_n - 1], new_motls[tom_n - 1]
    else:
        return new_mods[0], new_motls[0]



def get_resolution(fsc, fshells, cutoff, apix = False, fudge_ctf = True,
                   get_area = 'cutoff'):
    """
    Reduce FSC plot to a value closest to cutoff
    
    Inputs:
        fsc [str, list or 1d array] str is interpreted as path to 'arrFSC.txt'
            else list of FSC values
        fshells [str, list or 1d array] str is interpreted as path to 
            'freqShells.txt', else list of frequency values
    Optional:
        apix [float] pixel size. if specified, returns resolution value
        fudge_ctf: disregard dips in ctf (potentially) due to CTF effects. I.e.
            if FSC dips below [cutoff] but then goes up again above [cutoff],
            ignore the previous dip(s).
        get_area: [int, float or 'cutoff'] specify the number of resolution
            bins used for calculating the area under FSC.  If 'cutoff', the
            last bin is where FSC == cutoff.
    Output:
        [numpy array] (sampling frequency at cutoff, resolution,
                        area under FSC, last bing for area calculation)
    
    """
    if isinstance(fsc, str):
        fsc = np.array(open(fsc).read().strip('\n\r').split(),
                           dtype = float)
    if isinstance(fshells, str):
        fshells = np.array(open(fshells).read().strip('\n\r').split(),
                           dtype = float)
    if len(fsc) != len(fshells):
        raise ValueError('get_resolution: input array length mismatch')
    #I want to ignore trophs in FSC surrounded by very high FSC
    #e.g. capsids with or without DNA
    reliable_fsc = (1 - cutoff)/2 + cutoff
    hi_pk = find_peaks(fsc, height = reliable_fsc)[0]
    if hi_pk.size == 0:
        hi_pk = np.where(fsc == np.max(fsc))[0]
    #try to find low FSC "noise" peaks.  This is used to identify mask 
    #correlation at higher frequencies
    lo_pk = find_peaks(fsc, height = (-0.3, cutoff))[0]
    if lo_pk.size == 0:
        lo_pk = np.array([len(fsc)])
    if not np.all(hi_pk > max(lo_pk)):
        hi_pk = hi_pk[hi_pk < max(lo_pk)]
    if not np.any(hi_pk):
        hi_pk = np.array([0])
    search_fsc = fsc[max(hi_pk): min(lo_pk) + 1]
    search_shells = fshells[max(hi_pk): min(lo_pk) + 1]
    #truncate search range to the first negative slope. +1 to include the first
    #noise peak value.  This should make at least the last gradient value 
    #positive. 
    curr_grad = np.gradient(search_fsc)[:-1]
    while curr_grad[-1] >= 0:
        curr_grad = curr_grad[:-1]
    search_fsc = search_fsc[:len(curr_grad)]
    search_shells = search_shells[:len(curr_grad)]
    
    if not fudge_ctf:
        #truncate search range to where it stops being > cutoff, + 1
        search_end = np.where((search_fsc > cutoff) == False)[0][0] + 1
        search_fsc = search_fsc[:search_end]
        search_shells = search_shells[:search_end]
 
    #only check negative slope
    #I think this is redundant if fudge_ctf == False but shouldn't hurt
    grad_mask = np.gradient(search_fsc) <= 0
    search_fsc = search_fsc[grad_mask]
    search_shells = search_shells[grad_mask]
    
    res = find_nearest(search_fsc, cutoff)
    #    if len(res) > 1:
#        #this was meant to do the same as only taking negative slope
#        #but still useful?
#        res = res[0]
    if res.size > 1:
        raise ValueError('get_res: multiple values where fsc == cutoff. %s'
                         % res)
    if res >= len(search_fsc) - 2:
        print(('get_resolution: WARNING: detected resolution too close to ' +
            'resolution at Nyquist.  Consider using smaller pixel size.'))
        
    #area under curve
    if get_area == 'cutoff':
        cutoff_idx = int(np.where(fsc == search_fsc[res])[0] + 1)
    elif isinstance(get_area, int) or isinstance(get_area, float):
        cutoff_idx = get_area
    area = np.trapz(fsc[:cutoff_idx])
    
    i = interp1d(search_shells, search_fsc)
    new_fshells = np.linspace(search_shells[0], search_shells[-1],
                              len(search_shells)*10)
    new_fsc = i(new_fshells)
    res = new_fshells[find_nearest(new_fsc, 0.143)]
    
#    res = search_fsc[res]
#    #need to get the fsc value here to be able to find its index in the 
#    #whole FSC list not just search_fsc
#    res = np.where(fsc == res)[0]
    #res: (sampling frequency, resolution, area under FSC, last bing for area)
    res = np.array((res, 0, area, cutoff_idx), dtype = float)    
    res = np.round(res, decimals = 3)
    if apix:
        res[1] = np.round(apix/res[0], decimals = 1)
    return res

def plot_fsc(peet_dirs, out_dir, cutoff = 0.143, apix = False,
             fshells = False, fsc = False):  
    apix = float(apix)
    if isinstance(peet_dirs, str):
        peet_dirs = [peet_dirs]
    if isinstance(fshells, bool):
        fshells = [join(peet_dirs[x], 'freqShells.txt')
                    for x in range(len(peet_dirs))]
    if isinstance(fsc, bool):
        fsc = [join(peet_dirs[x], 'arrFSCC.txt')
                for x in range(len(peet_dirs))]
    fig, axs = plt.subplots(1, 1, figsize = (7, 7))
    axs.axhline(0, color = 'black', lw = 1)
    axs.axhline(cutoff, color = 'grey', lw = 1, linestyle='dashed')
    axs.set(xlabel = 'Fraction of sampling frequency')
    res = []
    get_area = 'cutoff'
    for x in range(len(peet_dirs)):
        with open(fshells[x], 'r') as f:
            ar_fshells = np.array(f.read().strip('\n\r').split(), dtype = float)
        with open(fsc[x], 'r') as f:
            ar_fsc = np.array(f.read().strip('\n\r').split())
            if ar_fsc[ar_fsc != 'NaN'].size == 0:
                warnings.warn('FSC determination failed.')
                return False
            else:
                ar_fsc = np.array(ar_fsc, dtype = float)
        
        axs.plot(ar_fshells, ar_fsc, label = x + 1)
        c_res = get_resolution(ar_fsc, ar_fshells, cutoff = cutoff,
                               apix = apix, get_area = get_area)
        get_area = int(c_res[3])
        res.append(c_res)
    res = np.array(res)
    axs.legend(loc = 'upper right')
    axs.set_ylim(None, top = 1)
    if apix:
        #also plot resolution values
        axR = axs.twiny()
        axR.set_xlim(axs.get_xlim())    
        top_xticks = axs.get_xticks()
        allowed_mask = top_xticks > 0
        zero_mask = top_xticks == 0
        disallowed_mask = top_xticks < 0     
        top_xticks[allowed_mask] = apix/top_xticks[allowed_mask]
        top_xticks[zero_mask] = 'inf'
        top_xticks[disallowed_mask] = None       
        for x in range(len(top_xticks)):
            if isinstance(top_xticks[x], float):
                top_xticks[x] = np.format_float_positional(
                                            top_xticks[x], precision = 2)
        axR.set_xticklabels(top_xticks)
        axR.set(xlabel = 'Resolution [Angstrom]')
    if apix:
        fig.suptitle('Current iteration resolution at %s FSC: %s' %
                     (cutoff, res[-1, 1]))
    out_fig = join(out_dir, 'fsc_plot.png')
    if os.path.isfile(out_fig):
        os.rename(out_fig, out_fig + '~')
    plt.savefig(out_fig)
    plt.close()
    return res
#    plt.setp(axes, xticks = arf)



#############################################################################

def ncc(target, probe, max_dist, interp = 1, outfile = False,
        subpixel = 'full_spline', testnorm = False):
    """Outputs normalised cross-correlation map of two images.
    Input arguments:
        target/reference image [numpy array]
        probe/query image [numpy array]
        maximum distance to search [even integer]
        interpolation [integer]
        outfile: if intered, writes an MRC image of the map     
        subpixel: 
            'zoom': uses ndimage.interpolation.zoom to upscale the CC map
            'cropped_spline' first crops the CC map (to 2*max_dist*interp)
                            then converts to spline
            'full_spline' converts to spline then accesses the central part
                            equivalent to 2*max_dist*interp
    Returns CC map [numpy array]
    """
    if np.std(target) == 0 or np.std(probe) == 0:
        raise ValueError('ncc: Cannot normalise blank images')    
    if max_dist > min(target.shape)//2 - 1:
        max_dist = min(target.shape)//2 - 1
#    if interp > 1:
#        target, probe = zoom(target, interp), zoom(probe, interp)
    #norm  
    target = (target - np.mean(target))/(np.std(target))
    probe = (probe - np.mean(probe))/(np.std(probe))
        
    cc_fft = fftn(target) * np.conj(fftn(probe))
    if testnorm:
        d1 = np.absolute(fftn(target))
        d2 = np.absolute(fftn(probe))
        d1[d1 == 0] = 1
        d2[d2 == 0] = 1
        ncc_fft = cc_fft/d1*d2 
    else:
        fft_abs = np.absolute(cc_fft)
        fft_abs[fft_abs == 0] = 1 #avoid divison by zero
        ncc_fft = cc_fft/fft_abs
    ncc = ifftshift(ifftn(ncc_fft).real)
    
#    #rectangular data should be handled correctly
#    if ncc.shape[0] != ncc.shape[1]:
#        overhang = (max(ncc.shape) - min(ncc.shape))/2
#        if ncc.shape[0] < ncc.shape[1]:
#            ncc = ncc[:, overhang:-overhang]
#        else:
#            ncc = ncc[overhang:-overhang]
    
    if subpixel == 'zoom':
        ncc = zoom(ncc, interp)
        edge = int((ncc.shape[0] - (max_dist * 2 * interp))//2) #only square iamges!!!!!!
        ncc = ncc[edge:-edge, edge:-edge]
    elif subpixel == 'cropped_spline':
        edge = int((ncc.shape[0] - (max_dist * 2))//2)
        cropped = ncc[edge:-edge, edge:-edge]
        spline = RectBivariateSpline(
            np.arange(cropped.shape[0]), np.arange(cropped.shape[1]), cropped)
        ncc = spline(np.arange(0, cropped.shape[0], 1./interp),
                     np.arange(0, cropped.shape[1], 1./interp))
    elif subpixel == 'full_spline':
        edge = int((ncc.shape[0] - (max_dist * 2))//2) 
        spline = RectBivariateSpline(
                np.arange(ncc.shape[0]), np.arange(ncc.shape[1]), ncc)
        ncc = spline(np.arange(edge, ncc.shape[0] - edge, 1./interp),
                     np.arange(edge, ncc.shape[1] - edge, 1./interp))
    if outfile:
        with mrcfile.new(outfile, overwrite=True) as mrc:
            ncc = ncc.copy(order='C')
            mrc.set_data(np.float16(ncc))
    return ncc

def get_peaks(cc_map, n_peaks = 100):
    """Finds peaks in 2D array.
    Input arguments:
        cc_map [2d array]
        max_peaks [int] maximum number of peaks to be stored. also defines
            size of output array
    Returns:
        out_peaks [n_peaks by 3 numpy array] 
                [:,0] peak X coord
                [:,1] peak Y coord
                [:,2] peak value
    """
    out_peaks = np.zeros((n_peaks, 3), dtype = float)
    out_peaks[:,2] = 0  

    peak_pos = peak_local_max(cc_map, 1)
    if len(peak_pos) == 0:
        #take map maximum if peak_local_max fails to find a peak
        peak_pos = np.array(np.where(cc_map == np.max(cc_map))).squeeze()
        peak_val = np.max(cc_map)
    else:        
        peak_val = cc_map[peak_pos[:, 0], peak_pos[:, 1]] #in decending order
        sort_ind = np.argsort(peak_val)
        peak_val = peak_val[sort_ind[::-1]]
        peak_pos = peak_pos[sort_ind[::-1]]
    idx = min(n_peaks, len(peak_pos))
    #peak coordinates are YX, swap:
    if peak_pos.ndim == 1:
        out_peaks[0, 0] = peak_pos[1]
        out_peaks[0, 1] = peak_pos[0]
        out_peaks[0, 2] = peak_val    
    else:
        out_peaks[:idx, 0] = peak_pos[:idx, 1]
        out_peaks[:idx, 1] = peak_pos[:idx, 0]
        out_peaks[:idx, 2] = peak_val[:idx]
    return out_peaks

def p_extract_and_cc(     
        alis,
        plotbacks,
        reprojected_mod,
        defocus_file,
        tlt,
        box,
        out_dir,        
        #file, pcle numbering
        chunk_base = False,
        chunk_id = False,
        n_pcls = False,       
        pcl_indices = False,
        group_ids = False,
        excludelist = False,
        #imaging params
        zero_tlt = False, 
        dosesym = False,
        orderL = True,
        dose = 0,
        pre_exposure = 0, 
        apix = False,
        V = 3E5,
        Cs = 2.7E7 ,
        ampC = 0.07,
        ps = 0,
        #cc and filtering params
        butter_order = 4,  
        limit = 10,
        interp = 4, 
        centre_bias = 0.1, 
        thickness_scaling = 1,
        #dev and output
        debug = 0,
        plot_pcls = False,
        test_dir = False,
        allow_large_jumps = False, 
        xf = False,
        orthogonal_subtraction = False,
        n_peaks = 100,
        ccmap_filter = 800,
        no_ctf_convolution = False
        ):
    """
    alis [str or list of str] query image file(s)
        if group_ids are specified, format has to be whatevername_??.extension
        where ?? is an integer of any length, 
    plotbacks [str or list of str] ref image file(s)
        if group_ids are specified, format has to be whatevername_??.extension
        where ?? is an integer of any length, 
    reprojected_mod [str or numpy array] 
        [str]: path to fiducial model.  This model
            has to have strictly the number of points/contour = number of tilts.
            Each contour is assumed to be a single pcl.
        [3d numpy array] [number of tilts:number of particles:
                        xcoord, ycoord, zcoord (tilt number)]
    defocus_file [str] path to IMOD (new/old) style defocus file
    tlt [str] path to .tlt file
    box [list of two ints] extraction box size
    out_dir [str] path to output directory
    Optional arguments:
    chunk_base [str] base of output filename
    chunk_id [bool or int] controls numbering of output csv file. e.g.
        chunk_id = 1 will produce flexo-001_tmp.csv
        if False, chunk_id = 0
    n_pcls [int] controls number of digits of output particle names. 
        e.g. n_pcls = 1000 will result in names containing 5 digits
    pcl_indices [bool or list of ints] specify subset of pcl indices to be used
        if False, use all pcls
        numbered from 0
    group_ids [bool or list of ints] #if not false, will look for a tilt series
        that has the same group number (expected filename is whatever_$$.ext
        where $$ is the group id).  This is to ensure that the particle 
        is exctracted from the correct file when non_overlapping_pcls is used.
    excludelist [list of ints] tilt numbers to be excluded (numbered from 1)
    zero_tlt [bool or int] starting tilt (lowest dose) of tilt series.
        Numbered from 1!
        If False, will use the tilt closest to 0 degrees.
        tilt {zero_tlt} and surrounding tilts are used for initial mean shift 
        estimation.  It is also the starting point for doseweighting (i.e.
        the tilt with the least amount of tiltering).
    dosesym [bool] was TS acquired using dosesymmetric scheme?
    orderL [bool] True: the order of bidirecitonal TS acquisition  is from 
        {zero_tlt} to tilt number 1. False: order is towards the last tilt.
    dose [int or float] fluence (electrons/angstrom) per tilt
    pre_exposure [int or float]
    apix [bool or float] pixel size of tilt series.
        False: will attempt to read it from  image files.
    V [int or float] accelerating voltage in V (typically 300000 or 3E5)
    Cs [int or float] hromatic aberration in mm (Krios 2.7E7)
    ampC [float] amplitude contrast (fraction of 1)
    ps [float] phase shift (functionality not tested properly...)
    butter_order [int] butterworth filter order
    limit [int] maximum allowed particle shift.  Controls cross-correlation 
        map size.
    interp [int] controls precision of subpixel shift.  Will scale particles by
        this value.
    centre_bias [float] adjust c in gaussian: exp(-(x-b)**2/(2c)**2) 
        CC maps are multiplied by this 2D gaussian in order to favour peaks
        closer to the median shift of previous tilts
    n_peaks [int] number of peaks stored for each img

    plot_pcls [list of ints] if debug == 2, will only generate graphical
        output for particles whose pcl_indices == plot_pcls

    orthogonal_subtraction [bool] True if using orthogonal subtraction.  
        get_pcl_defoci expects input stacks of size = original aligned stack.
        Orthogonal subtraction removes excludelist entries, meaning it can
        be different size.

    output:
        chunk_base-???_tmp.csv
        output shift values are of ref relative to query, i.e. ref shifted by
        -2, -2 relative to query will produce -2, -2 x and y shifts
    
    
    """
    plt.ioff()


#    step = 5 #think this has to be odd
    
    #check file inputs
    if isinstance(alis, str):
        alis = [alis]
    if isinstance(plotbacks, str):
        plotbacks = [plotbacks]     
    #read in and format fiducial model, or pass on if it's numpy array
    if isinstance(reprojected_mod, str):
        mod = PEETmodel(reprojected_mod).get_all_points()
        max_tilt = int(np.round(max(mod[:,2])+1, decimals = 0))
        rsorted_pcls = np.array([np.array(mod[x::max_tilt])
                                    for x in range(max_tilt)])
        #rsorted_pcls shape [number of tilts, number of particles, 3]
    else:
        rsorted_pcls = reprojected_mod
    
    #select a subset of particles if pcl_indices are specified
    if not isinstance(pcl_indices, bool):
        if len(pcl_indices) == 1:
            #need to keep the array size the same
            tmp = np.zeros((rsorted_pcls.shape[0], 1, rsorted_pcls.shape[2]))
            tmp[:,0] = rsorted_pcls[:, pcl_indices]
            rsorted_pcls = tmp
        else:
            rsorted_pcls = rsorted_pcls[:, np.array(pcl_indices)]
    else:
        pcl_indices = np.arange(rsorted_pcls.shape[1])

    #    convert group_ids into file identifiers
    if not isinstance(group_ids, bool):
        if len(alis) == 1:
            #skip all this nonsense if there is only one file
            file_ids = np.zeros(rsorted_pcls.shape[1], dtype = int)  
        else:
            #extract file identifiers from paths
            #VP 8/6/2021: alis can now be the initial .ali
            #fa = [int(x.split('_')[-1].split('.')[0]) for x in alis]          
            fp = [int(x.split('_')[-1].split('.')[0]) for x in plotbacks]   
            #if not (np.unique(fa) == np.unique(fp)).all():
            #    raise Exception('Ali and plotback numbering do not match.')
            if len(np.unique(fp)) != len(np.unique(group_ids)):
                raise Exception('Numbers of groups and files do not match.')
            #make sure files are in ascending order
            #alis = [alis[np.argsort(fa)[x]] for x in range(len(alis))]
            plotbacks = [plotbacks[np.argsort(fp)[x]]
                            for x in range(len(plotbacks))]
            uid = np.unique(group_ids)
            test = np.isin(uid, fp)
            if not np.all(test):
                raise ValueError('Group IDs do not match file names')
            #renumber group IDs to match indices of [alis]
            id_dic = np.zeros(np.max(uid + 1), dtype = int)
            for x in range(len(uid)):
                id_dic[uid[x]] = x
            file_ids = [id_dic[group_ids[x]] for x in range(len(group_ids))]
    else:
        file_ids = np.zeros(rsorted_pcls.shape[1], dtype = int)  
    #read in image files
    #mrcfile doesn't like files processed in e.g. bsoft, use permissive mode
    
    if np.unique(alis).size == 1:
        #if the initial ali is passed on multiple times
        read_ali = deepcopy(mrcfile.open(alis[0], permissive = True).data)
        one_ali = True
    else:
        read_ali = [deepcopy(mrcfile.open(x, permissive = True).data)
                for x in alis]
        one_ali = False
    read_plotback = [deepcopy(mrcfile.open(x, permissive = True).data)
                    for x in plotbacks]

    #check imaging params
    #if apix is not specified, get from first ali
    if isinstance(apix, bool):
        apix = get_apix_and_size(alis[0])[0]
    wl = (12.2639/(V+((0.00000097845)*(V**2)))**0.5)

    tilt_angles = [float(x.strip('\n\r').strip(' ')) for x in open(tlt)]
    #VP 13/06/2020 using plotback instead of subtracted stack as input for 
    #get_pcl_defoci to avoid issues with excludelist when using
    #orthogonal subtraction
    if no_ctf_convolution:
        defoci = np.zeros((len(tilt_angles), rsorted_pcls.shape[1]))
    else:
        defoci = get_pcl_defoci(False, tlt, False, out_dir, rsorted_pcls,
                plotbacks[0], apix, defocus_file, excludelist, verbose = False,
                )     
    
    if isinstance(zero_tlt, bool):
        zero_tlt = find_nearest(tilt_angles, 0) + 1
    if isinstance(n_pcls, bool):
        n_pcls = 3
#    if isinstance(test_dir, bool):
#        peak_out = join(out_dir, 'extracted_particles')
#    else:
#        peak_out = test_dir
#    if isinstance(chunk_base, bool):
#        chunk_base = 'flexo'
        
    #remove excludelist tilts and pcl coords
    if not isinstance(excludelist, bool):
        #excludelist is numbered from one, fix:
        excludelist = np.array(excludelist) - 1
        excludelist_mask = np.isin(
                list(range(rsorted_pcls.shape[0])), excludelist, invert = True)    
        rsorted_pcls = rsorted_pcls[excludelist_mask]
        for x in range(len(alis)):
            if not orthogonal_subtraction:
                #excludelist entries are removed during orthogonal subtraction
                if one_ali:
                    read_ali = read_ali[excludelist_mask]
                else:
                    read_ali[x] = read_ali[x][excludelist_mask]
            #VP 2/12/2020 I think this indentation is correct (i.e not part
            #of if statement).  The reprojected tilt series that come from
            #orthogonal subtraction don't have excludelist views...
            read_plotback[x] = read_plotback[x][excludelist_mask]


    for x in range(rsorted_pcls.shape[1]):
        cc_peaks = np.zeros((rsorted_pcls.shape[0], n_peaks, 3))
        fcc_peaks = np.zeros((rsorted_pcls.shape[0], n_peaks, 3))
        #cc_peak shape [number of tilts, number of peaks, 3
        #(x coord, y coord, peak value)]
        #fcc_peaks - lowpassed cc maps
        
        #extract particle stacks
        if one_ali:
            query = extract_2d_simplified(read_ali, rsorted_pcls[:, x], box)
        else:
            query = extract_2d_simplified(read_ali[file_ids[x]], 
                                               rsorted_pcls[:, x], box)
        ref = extract_2d_simplified(read_plotback[file_ids[x]],
                                                  rsorted_pcls[:, x], box)

              
        #write out particle stacks
        if test_dir:
            testpcl = join(test_dir, 'testquery%0*d.mrc'
                           % (len(str(n_pcls)), pcl_indices[x]))
            testref = join(test_dir, 'testref%0*d.mrc'
                           % (len(str(n_pcls)), pcl_indices[x]))
            #if not os.path.isfile(testpcl):
            write_mrc(testpcl, query)
            write_mrc(testref, ref)
        
        ref = ctf_convolve_andor_dosefilter_wrapper(ref, zero_tlt, dose,
                apix, V, Cs, wl, ampC, ps, defoci[:, x], butter_order, dosesym,
                orderL, pre_exposure)
        query = ctf_convolve_andor_dosefilter_wrapper(query, zero_tlt, dose,
                apix, V, Cs, wl, ampC, ps, 0, butter_order, dosesym,
                orderL, pre_exposure)
        #write out filtered particle stacks
        if test_dir:
            ccmaps = []
            testpcl = join(test_dir, 'ftestquery%0*d.mrc'
                           % (len(str(n_pcls)), pcl_indices[x]))
            testref = join(test_dir, 'ftestref%0*d.mrc'
                           % (len(str(n_pcls)), pcl_indices[x]))
            #if not os.path.isfile(testpcl):
            write_mrc(testpcl, query)
            write_mrc(testref, ref)
    
        #these plots are extremely slow:
        #debug = 3 means a plot is generated for every particle
        if debug > 2:
            debug = 2
        #debug = 2, a plot is generated only for particles included in 
        #the plot_pcls list
        if debug == 2:
            if isinstance(plot_pcls, bool):
                debug = 1
            elif np.isin(pcl_indices[x], plot_pcls):
                debug = 2
            else:
                debug = 1
        #CC########################################
        for y in range(len(ref)):
            ccmap = ncc(ref[y], query[y], limit, interp)
            cc_peaks[y] = get_peaks(ccmap, n_peaks)
            fmap = butter_filter(ccmap, ccmap_filter, apix)
            fmap *= np.max(ccmap)/np.max(fmap)
            #adjust filtered max to be ~max of ccmap
            fcc_peaks[y] = get_peaks(fmap, n_peaks)
            if test_dir:
                ccmaps.append(ccmap)
        #convert peak coords to shifts from 0
        cc_peaks[:,:,:2] = np.divide(cc_peaks[:,:,:2],
                                    float(interp))-float(limit)
        fcc_peaks[:,:,:2] = np.divide(fcc_peaks[:,:,:2],
                                    float(interp))-float(limit)
        tmp_dir = join(out_dir, 'cc_peaks')
        tmp_out = join(tmp_dir, chunk_base + '-%0*d_peaks.npy' %
                       (len(str(n_pcls)), pcl_indices[x]))
        tmp_out_f = join(tmp_dir, chunk_base + '-%0*d_f_peaks.npy' %
                       (len(str(n_pcls)), pcl_indices[x]))
        
        if not isdir(tmp_dir):
            os.makedirs(tmp_dir)
        np.save(tmp_out, cc_peaks)
        np.save(tmp_out_f, fcc_peaks)
        
        if test_dir:
            ccmap_out = join(test_dir, 'ccmap_%0*d.mrc'
                       % (len(str(n_pcls)), pcl_indices[x]))        
            write_mrc(ccmap_out, np.array(ccmaps))




def shifts_by_cc(pcl_n, ref, query, limit, interp, tilt_angles,
                  zero_tilt, centre_bias = 0.1, thickness_scaling = 1.5,
                  step = 5, out_dir = False, debug = 0,
                  log = False, #new
                  allow_large_jumps = False):

    """
    previously cc_but_better
    Intended to take in data for 1 particle.
    
    What does the output mean? Having ref = [img] and query = [bint -translate
    1,2 img img2], the output shift will be [-1, -2]
    IMPORTANT: Throughout the function X and Y values are swapped because 
    data read in by mrcfile has their XY axes swapped (but not Z!, so its YXZ).
    This means that peaks coordinates/shifts are YX throughout, but get
    swapped to XY at the end.
    """
    if not log and debug > 0:
        print('cc_but_better: No log file specified, logging disabled.')
        debug = 0
        
    debug_out = []
    map_size = limit*interp*2
    max_iters = 8
    iter_n = 0
    imp_mat = []
    imp_bounds = 1., 1.4
    #imp values > ~1.4 are unreasonably high and suggest something failed
    starting_centre_bias = centre_bias
    
    #define middle tilts, zero_tlt is numbered from 1
    if len(ref) < step:
        bot, top = 0, len(ref)
    else:
        bot = int(zero_tilt - 1 - (step - 1)//2)
        top = int(zero_tilt - 1 + (step - 1)//2 + 1)
    
    #make stack of ccmaps
    ccmaps = np.array([(ncc(ref[x], query[x], limit, interp))
                        for x in range(len(ref))])
            
#    #CC values need to be shifted to positive values!
#    ccmaps = ccmaps - np.min(ccmaps, axis = (1,2))[:, None, None]

    #######################################################################
    #initial CC of middle section: distance matrix biased towards 0,0 shift
    out_name = "%02d_mid_compare_ccs.png" % pcl_n
    write_to_log(log, (('#'*5 + 'Particle %s' + '#'*5 +
                       '\nAligning tilts %s-%s (numbered from 1)')
                       % (pcl_n, bot + 1, top + 1)), debug)
    
    shifts, imp = cc_middle(pcl_n, ref, query, tilt_angles, ccmaps, interp,
                            limit, map_size, top, bot, centre_bias,
                            thickness_scaling, out_dir, debug,
                            'init', out_name)
    #increase centre_bias if imp < 1
    while (imp < imp_bounds[0] or imp > imp_bounds[1]) and iter_n < max_iters:
        iter_n += 1
        #break if nothing changes after 2 iterations
        if np.any(np.unique(imp_mat, return_counts = True)[1] > 2):
            write_to_log(log, ("compare_ccs: peak value has not changed in"
                    " the last three iterations (%.4f).  Continuing with"
                    " current shifts.") % imp, debug)
            #reset centre bias...debatable
            centre_bias = starting_centre_bias
            break     
        ocb = centre_bias
        #TBD!!!!
        centre_bias = failed_peak_scaling(centre_bias)        
        if imp > imp_bounds[1]:
            debug_out.extend(
            ("compare_ccs: averaged peak improvement\t %.4f.\nThis is an"
             "unreasonable amount and something probably went wrong." % imp))

        else:
            debug_out.extend('compare_ccs: negative peak change %.4f.' % imp)
        debug_out.extend(
            ("Increasing centre_bias from %.2f to %.2f. (iteration %s of %s)" %
                                    (ocb, centre_bias, iter_n, max_iters - 1)))

        med = (np.median(shifts[bot : top, 0]),
               np.median(shifts[bot : top, 1]))
        compare_out_name = "%02d_mid_compare_ccs_iter%s.png" % (pcl_n, iter_n)
        biased_out_name = out_name = '%02d_mid_%s-%s_iter%s.png' % (
                                            pcl_n, bot + 1,top + 1, iter_n)
        shifts, imp = cc_middle(pcl_n, ref, query, tilt_angles, ccmaps, interp,
                                limit, map_size, top, bot, centre_bias,
                                thickness_scaling, out_dir, debug, med,
                                compare_out_name, biased_out_name, iter_n)     
        imp_mat.append(imp)
    else:
        debug_out.extend("compare_ccs: averaged peak improvement %s" % imp)
        #print wmessage   
         
    #walk up   
    if len(ref) > step:
        shifts, wmsg = cc_walk(pcl_n, ccmaps, shifts, interp, limit,
                       len(ref), top, 1, step, tilt_angles, centre_bias,
                       thickness_scaling, out_dir, debug,
                       allow_large_jumps)
        out_name = ("%02d_%s-%s_compare_ccs.png"
                    % (pcl_n, top + 1, len(ref) + 1))
        imp, avg_ccmaps = compare_ccs(pcl_n, ref, query, len(ref), top, shifts,
                                      interp, limit, out_dir, debug)
        if debug > 0:
            debug_out.extend("\nStarting cc_walk up.")
            debug_out.extend(wmsg)
            wmessage = ["Particle %s tilts %s-%s averaged peak change %.4f" %
                        (pcl_n, top + 1, len(ref) + 1, imp)]
            debug_out.extend(wmessage)
            #print wmessage
        #walk down
        shifts, wmsg = cc_walk(pcl_n, ccmaps, shifts, interp, limit,
                       bot, 0, -1, step, tilt_angles, centre_bias,
                       thickness_scaling, out_dir, debug,
                       allow_large_jumps)
        out_name = "%02d_%s-%s_compare_ccs.png" % (pcl_n, 1, bot + 1)
        imp, avg_ccmaps = compare_ccs(pcl_n, ref, query, bot, 0, shifts,
                                      interp, limit, out_dir, debug)
        if debug > 0:
            debug_out.extend("\nStarting cc_walk down.")
            debug_out.extend(wmsg)
            debug_out.extend(
                    "Particle %s tilts %s-%s averaged peak change %.4f" %
                        (pcl_n, top + 1, len(ref) + 1, imp))
            #print wmessage
    shifts = np.divide(shifts, float(interp))-float(limit)
    #xy is flipped
    shifts = np.flip(shifts, axis = 1)
    return shifts, debug_out, ccmaps


def cc_middle(pcl_n, ref, query, tilt_angles, ccmaps, interp, limit, map_size,
              top, bot, centre_bias, thickness_scaling, out_dir, debug,
              med = 'init', comp_out_name = False,
              biased_out_name = False, iter_n = False): 
    

    """
    Check shifts of middle tilts.
    
    """
    
    def biased_cc(pcl_n, tilt_angles, ccmaps, interp, limit,
                  centre_x, centre_y, top, bot, centre_bias,
                  thickness_scaling = 1.5,
                  out_dir = False, debug = 0, out_name = False,
                  iter_n = False):
        """
        Multiplies CC map with a centre-biasing function before feeding it to
        buggy_peaks.
        Inputs:
            {ccmaps} - CC maps in an array with len = len(ref)
            {interp, limit} - image interpolation
            {centre_x, centre_y} - coordinates for the centre of the function
                                    e.g. {buggy_peaks}[0,1], {buggy_peaks}[0,0]
            {bot, top} - use tilts [bot:top]
            {centre_bias, power} - matches distance_bias inputs
    
        """
        init_peaks = []
        
        if debug == 2:
            fig, axs = plt.subplots(3, top-bot, figsize = ((top-bot)*3, 11))
            plot_title = 'CCs of middle tilts (%s-%s)' % (bot + 1, top + 1)
            if iter_n:
                plot_title += " iteration %s." % iter_n
            fig.suptitle(plot_title)
            
        for x in range(bot, top):
            cos_centre_bias = bias_increase(tilt_angles[x], centre_bias,
                                            thickness_scaling)
            db = g2d(centre_x, centre_y, interp*limit*2, cos_centre_bias)                
            ccmap = ccmaps[x]*db
            if debug < 2:
                pk, val, warn = buggy_peaks(ccmap)
            elif debug > 1:           
                print("this breaks is there are no nice peaks")
                #first, plot unbiased ccmaps[x]
                pk, val, warn = buggy_peaks(ccmaps[x])
                axs[0, x-bot].imshow(ccmaps[x], cmap = 'Greys')
                axs[0, x-bot].add_artist(circle(pk[0,1], pk[0,0], interp))
                axs[0, x-bot].title.set_text('Raw #%s: peak[%s,%s]'
                                               % (x, pk[0,1], pk[0,0]))
                axs[2, x-bot].plot(
                        norm(np.max(ccmaps[x], axis = 0)), label = 'raw')
                #then plot ccmap (ccmap = ccmaps[x]*distance bias)
                pk, val, warn = buggy_peaks(ccmap)
                axs[1, x-bot].imshow(ccmap, cmap = 'Greys')
                axs[1, x-bot].add_artist(circle(pk[0,1], pk[0,0], interp))
                axs[1, x-bot].title.set_text('biased #%s: peak[%s,%s]'
                                               % (x, pk[0,1], pk[0,0]))     
                axs[2, x-bot].plot(
                        norm(np.max(ccmap, axis = 0)), label = 'biased')
                axs[2, x-bot].title.set_text('X max proj.')
                axs[2, x-bot].legend(loc = 'lower right')
                fig.tight_layout()
            if pk.ndim > 1:
                pk = pk[0]
            init_peaks.append(pk)
            if len(pk) == 0:
                raise ValueError(
                    ('get_cc_map, particle %s tilt %s returned a flat array.' +
                    "This can happen if the input model and volumes don't match")
                    % (pcl_n, x))
        if debug > 1: 
            if not out_name:
                out_name = '%02d_mid_%s-%s.png' % (pcl_n, bot + 1,top + 1)
            fig.savefig(join(out_dir, out_name), bbox_inches = 'tight')   
            plt.close()
    
        return np.array(init_peaks)

    #end def######################

    shifts = np.zeros((len(ccmaps), 2))
    #skip initial peaks if median shift is specified
    if med == 'init':  
        init_peaks = biased_cc(pcl_n, tilt_angles, ccmaps, interp, limit,
                             map_size//2, map_size//2, top, bot, centre_bias,
                             thickness_scaling, out_dir, debug) 
        shifts[bot : top] = init_peaks
        med = (np.median(shifts[bot : top, 0]),
               np.median(shifts[bot : top, 1]))
    #now bias CC map to median shift instead of 0,0
    refined_peaks = biased_cc(pcl_n, tilt_angles, ccmaps, interp, limit,
                              med[1], med[0], top, bot, centre_bias,
                              thickness_scaling, out_dir, debug,
                              biased_out_name, iter_n)    
    shifts[bot : top] = refined_peaks      
    #check for improvement        
    imp, avg_ccmaps = compare_ccs(pcl_n, ref, query, top, bot, shifts, interp,
               limit, out_dir, debug, comp_out_name)
    return shifts, imp        

class extracted_particles:
    """
    
    """
    def __init__(self, sorted_pcls, cc_peaks = False, out_dir = False,
                 base_name = False,
                 chunk_base = False, n_peaks = 100, excludelist = [],
                 groups = False, tilt_angles = False, model_3d = False,
                 apix = False):
        """
        
        """
        self.sorted_pcls = sorted_pcls
        self.out_dir = out_dir
        self.base_name = base_name
        self.chunk_base = chunk_base
        self.n_peaks = n_peaks
        self.excludelist = excludelist
        self.groups = groups
        self.tilt_angles = tilt_angles
        self.tilt_subset = False
        self.dist_matrix = False
        self.dist_score_matrix = False        
        self.shifts = False
        self.shift_mask = False
        self.cc_values = False
        self.model_3d = model_3d
        self.apix = apix
        
        self.flg_excludelist_removed = False  
        self.flg_outliers_removed = False
        self.read_3d_model()        
        self.remove_tilts_using_excludelist()
        self.remove_group_outliers()
        self.update_indices()   

    
    def update_indices(self):
        if isinstance(self.groups, str):
            self.groups = np.load(self.groups)
        if isinstance(self.sorted_pcls, str):
            self.sorted_pcls = np.load(self.sorted_pcls)        
        self.num_tilts = self.sorted_pcls.shape[0]
        self.tilt_indices = np.array(self.sorted_pcls[:, 0, 2], dtype = int)
        self.num_pcls = self.sorted_pcls.shape[1]
        self.pcl_ids = np.array(self.sorted_pcls[0, :, 3], dtype = int)  
        #pcl_ids are original pcl indices (prior to removing group outliers)
        if isinstance(self.tilt_angles, str):
            #tilt angles can be specified as path to .tlt
            self.tilt_angles = np.array([float(x.strip('\n\r').strip(' '))
                                for x in open(self.tilt_angles)])

    def remove_tilts_using_excludelist(self, excludelist = []):
        #this must be run exactly once
        if not self.flg_excludelist_removed:
            if not (isinstance(excludelist, bool)
                    or isinstance(self.excludelist, bool)):
                tmp_excludelist = np.array(self.excludelist) - 1
                exc_mask = np.isin(self.tilt_indices, tmp_excludelist,
                                   invert = True)  
                self.sorted_pcls = self.sorted_pcls[exc_mask]
                self.tilt_angles = self.tilt_angles[exc_mask]
                self.update_indices()
                self.flg_excludelist_removed = True
        
    def remove_group_outliers(self, groups = False):
        """
        remove particles that will not be extracted due to
        
        using non_overlapping_pcls can cause particles to be excluded
        """
        if not self.flg_outliers_removed:
            if not isinstance(groups, bool):
                if isinstance(groups, str):
                    self.groups = np.load(groups)
                else:
                    self.groups = groups
            elif isinstance(self.groups, str):
                self.groups = np.load(self.groups)
                
            if isinstance(self.sorted_pcls, str):
                self.sorted_pcls = np.load(self.sorted_pcls)
            if self.groups.ndim == 1:
                #in case there is only one group
                gmask = self.groups
            else:
                gmask = np.any(self.groups, axis = 0)
                #gmask has len(particles) - len(excludelist)
                #this removes outliers from self.groups, meaning that this can be
                #executed multiple times safely.
                #But it does not protect from re-reading groups
                #--> introduced flg_outliers_removed
                #self.groups = self.groups[:, gmask]
            self.sorted_pcls = self.sorted_pcls[:, gmask]
            self.flg_outliers_removed = True
            self.update_indices()
            if not isinstance(self.model_3d, bool):
                self.model_3d = self.model_3d[gmask]

    
    def read_3d_model(self, model_3d = False):
        if isinstance(model_3d, str):
            self.model_3d = model_3d
        if isinstance(self.model_3d, str):
            if isfile(self.model_3d):
                mod_path = self.model_3d  
        elif isinstance(self.model_3d, bool) and self.base_name:
            #model name defined in rotate_model()
            mod_path = join(self.out_dir, 'bin_%s.mod' % self.base_name)
            if not isfile(mod_path):
                mod_path = False
        else:
            mod_path = False
        if mod_path:
            self.model_3d = np.array(PEETmodel(mod_path).get_all_points())
            self.remove_group_outliers()
        #VP 7/5/2021
        #temoporarily? removed:
        #if (np.absolute(np.mean(self.sorted_pcls[:,:,1] - self.model_3d[:, 1]))
        #    > 1):
        #    raise ValueError('3D model does not match particle coordinates.')
        
    def read_cc_peaks(self, out_dir = False, chunk_base = False,
                      spec_path = False, name_ext = 'peaks'):
        """
        Reads in pickled arrays of cc_peaks.  Can be specified as tuple
        of paths or using output directory and chunk base.
        
        self.shifts has shape [num_tilts, num_pcls, n_peaks, 3
                                 (xshift, yshift)]
        """
        
        self.shifts = np.zeros((self.num_tilts, self.num_pcls,
                                  self.n_peaks, 2))
        self.cc_values = np.zeros((self.num_tilts, self.num_pcls,
                                  self.n_peaks))
        
        if out_dir:
            self.out_dir = out_dir
        if chunk_base:
            self.chunk_base = chunk_base
            
        for x in range(self.num_pcls):
            if self.out_dir and self.chunk_base:
                tmp_path = realpath(
                        join(self.out_dir, 'cc_peaks', self.chunk_base
                             + '-%0*d_%s.npy' % (len(str(self.num_pcls)),
                                                 self.pcl_ids[x], name_ext)))
            elif isinstance(spec_path, tuple):
                tmp_path = spec_path[x]
            else:
                raise ValueError('Input paths not specified')
            tmp_arr = np.load(tmp_path)
            self.shifts[:, x] = tmp_arr[:, :, :2]
            self.cc_values[:, x] = tmp_arr[:, :, 2]
        
#        #mask ccc <= 0
#        self.cc_values = np.ma.masked_less_equal(self.cc_values, 0)
#        cc_mask = np.ma.getmask(self.cc_values)
#        cc_mask = np.stack((cc_mask, cc_mask), axis = 3)
#        self.shifts = np.ma.masked_where(cc_mask, self.shifts)
        
    def split_for_processchunks(self, pcls_per_core):
        #rearrange ssorted_pcls by groups, only operate with indices
        order = np.argsort(self.sorted_pcls[0,:,4])
        group_ordered = np.array(self.sorted_pcls[0,:,3:5][order], dtype = int)
        c_tasks = []
        for x in np.arange(self.num_pcls + 1, step = pcls_per_core):
            tmp_task = group_ordered[x:x + pcls_per_core]
            #no point generating new chunk for very small groups, add to existing
            if len(tmp_task) <= pcls_per_core/5 and x != 0:
                c_tasks[-1] = np.vstack((c_tasks[-1], tmp_task))
            else:
                c_tasks.append(tmp_task)
           
        #c_tasks = [group_ordered[x:x + pcls_per_core]    
        #        for x in np.arange(self.num_pcls + 1, step = pcls_per_core)]
        
        return c_tasks  
    
    def fit_cosine(self, init_scale = 1, init_const = 0, return_model = True):
        """Author: Daven Vasishtan
        """    
        peak_median = np.median(self.cc_values[:, :, 0], axis = 1)        
        def f(p):
            model = p[0] * np.cos(np.radians(self.tilt_angles)) + p[1]
            return np.sum((peak_median - model)**2)
        new_params = minimize(f, [init_scale, init_const])
        if not return_model:
            return new_params
        else:
            return (new_params.x[0] * np.cos(np.radians(self.tilt_angles))
                    + new_params.x[1])


    def pick_shifts_basic_weighting(self, neighbour_distance = 50,
                                    n_peaks = 5,
                                    cc_weight_exp = 5,
                                    plot_pcl_n = False,
                                    figsize = (12,12)):

        def weighted_mean(vals, distances, exp = 2, axis = 0, stdev = False):
            weights = 1/np.array(distances, dtype = float)**exp
            if not isinstance(vals, np.ndarray):
                vals = np.array(vals)
            weighted_mean = (np.sum(vals*weights, axis = axis)/
                             np.sum(weights, axis = axis))
            if stdev:
                weighted_stdev = np.sqrt(np.sum(
                    weights*(vals - np.expand_dims(weighted_mean, axis))**2,
                                                                axis = axis)
                    /(((np.float(vals.shape[axis]) - 1)/vals.shape[axis])
                                            *np.sum(weights, axis = axis))
                                                )
                return weighted_mean, weighted_stdev
            else:
                return weighted_mean  

        def cc_distance_weight_neighbours(self, pcl_index, dst, pos,
                                          n_peaks = 5, cc_weight_exp = 3,
                                          cc_weight = True):
            """
            Shifts are weighted sequentially be CCC and then (3D) distance from
            neighbouring particles
            """

            #pick neighbours within distance limit. First index is pcl_index,
            #filler value is 1 + num_pcls 
            nbr_indices = np.unique(pos[pcl_index][1:])[:-1]
            nbr_shifts = self.shifts[:, nbr_indices]
            nbr_shifts = nbr_shifts[:, :, :n_peaks]
            nbr_cccs = self.cc_values[:, nbr_indices]
            nbr_cccs = nbr_cccs[:, :, :n_peaks]
            nbr_dist = dst[pcl_index][dst[pcl_index] != np.inf][1:]
            
            if cc_weight:
                #use fraction of neighbour max ccc instead of raw ccc
                cc_ratios = (nbr_cccs/
                             nbr_cccs[:, :, 0][:, :, None])**cc_weight_exp
                #weight shifts of each neighbouring particle using ccc_ratios
                cc_w_mn = (np.sum(nbr_shifts*cc_ratios[:, :, :, None], axis = 2)
                           /np.sum(cc_ratios[:, :, :, None], axis = 2))
                #weighted mean of the ccc weighted shifts
                wmn, wstd = weighted_mean(cc_w_mn, nbr_dist[None, :, None],
                                          axis = 1, stdev = True)
            else:
                wmn, wstd = weighted_mean(nbr_shifts[:, :, 0],
                            nbr_dist[None, :, None], axis = 1, stdev = True)
            return wmn, wstd
        
        def tilt_weighted_mean(wshifts, window = 5):
            """
            Rolling weighted mean.
            Inputs:
                wshifts [num_tilts, 2]
            """
            if window%2 != 1:
                window += 1
            hw = int(window/2)
            #define arbitrary distance between tilts
            tilt_distance = np.hstack((np.arange(1 + hw, 1, -1), 1,
                                       np.arange(2, 2 + hw)))
            #deal with array edges by defining the min and max index of 
            #tilt_distance for each loop
            lo_dst_indices = np.hstack((np.arange(hw, 0, -1),
                                    np.zeros(len(wshifts) - hw, dtype = int)))
            hi_dst_indices = np.hstack((np.zeros(len(wshifts) - hw,
                                                    dtype = int) + window,
                                        np.arange(-1, -1 - hw, -1)))  
            tilt_weighted = np.zeros(wshifts.shape)
            for y in range(len(wshifts)):
                tmp_dst = tilt_distance[lo_dst_indices[y]:hi_dst_indices[y]]
                min_idx = max(0, y - hw)
                max_idx = min(len(wshifts), y + hw + 1)
                tmp_wshifts = wshifts[min_idx:max_idx]
                tilt_weighted[y] = weighted_mean(tmp_wshifts, tmp_dst[:, None])
            return tilt_weighted
        
        def score_shifts(self, wshifts, pcl_index, n_peaks = 10, 
                          cc_weight_exp = 5, dist_weight_exp = 2,
                          return_mask = False):
            
            """
            Score of shifts from a reference shift (wshift):
                [euclidian distance]**exp1/[ratio of max CCC]***exp2
                
            """
            
            def euc_dist(a, b):
                return np.sqrt((b[..., 0] - a[..., 0])**2
                                + (b[..., 1]-a[..., 1])**2)
                
            distance = euc_dist(self.shifts[:, pcl_index, :n_peaks],
                                wshifts[:, None])
            cc_ratios = (self.cc_values[:, pcl_index, :n_peaks]
                         /self.cc_values[:, pcl_index, 0, None])
            sc = (distance**dist_weight_exp)/cc_ratios**cc_weight_exp
            
            m = np.array(np.where(sc == np.min(sc, axis = 1)[:, None],
                                      np.ones(sc.shape), 0), dtype = bool)
            if return_mask:
                return m
            else:
                return self.shifts[:, pcl_index, :n_peaks][m]
            
        def plot_weighted_pcl(self, pcl_index, tilt_mean, weighted_std, 
                              figsize = (12,12), out_dir = False):
            f, ax = plt.subplots(2, 1, figsize = figsize)
            xvals = np.array(self.tilt_angles, dtype = int)
            for axis in range(2):
                ax[axis].axhline(0, c = 'k')
                #weighted mean
                ax[axis].scatter(xvals, tilt_mean[:, axis],
                                  c = 'tab:blue', alpha = 0.6,
                                  label = 'weighted mean')  
                ax[axis].plot(xvals, tilt_mean[:, axis],
                  c = 'tab:blue', alpha = 0.6)
                #weighted std
                ax[axis].fill_between(xvals,
                      tilt_mean[:, axis] + weighted_std[:, axis],
                      tilt_mean[:, axis] - weighted_std[:, axis], alpha = 0.5,
                      label = 'neighbour STD') #color = 'tab:blue', 
                #max ccc shifts
                ax[axis].scatter(xvals, self.shifts[:, pcl_index, 0, axis],
                                  c = 'tab:purple', alpha = 0.6, 
                                  label = 'max CCC shift')
                ax[axis].plot(xvals, self.shifts[:, pcl_index, 0, axis],
                  c = 'tab:purple', alpha = 0.6)
                #best scoring shift
                masked_shift = self.shifts[:, pcl_index, :, axis][
                                        self.shift_mask[:, pcl_index, :, axis]]
                ax[axis].scatter(xvals, masked_shift, c = 'tab:orange',
                  label = 'max scoring shift')  
                ax[axis].plot(xvals, masked_shift, c = 'tab:orange')
                  
                ax[axis].legend()

                ax[axis].set_ylabel('shift [pixel]')
                ax[axis].set_xlabel('tilt angle [degrees]')
            
            ax[0].set_title('X axis shift')
            ax[1].set_title('Y axis shift')                
            if out_dir:
                plt.savefig(join(self.out_dir, 'shift_scoring_pcl%0*d.png' % (
                            len(str(self.num_pcls)), pcl_index)))
                plt.close()                

                
        if isinstance(self.shifts, bool):
            try:
                self.read_cc_peaks()
            except:
                raise ValueError(
                    'No CC data. Use extracted_particles.read_cc_peaks()')             
        self.shift_mask = np.zeros(self.shifts.shape, dtype = bool)
        self.shift_mask[:, :, n_peaks:] = 0 
        
        tree = KDTree(self.model_3d)
        dst, pos = tree.query(self.model_3d,
                              self.num_pcls)

    #set minimum number of neighbours to 2
        for x in range(len(dst)):
            farenuf = np.where(dst[x] >= min(neighbour_distance, np.max(dst[x])))[0][0]
            dst[x][max(3, farenuf):] = np.inf
            pos[x][max(3, farenuf):] = dst.shape[0] + 1

        
        for pcl_index in range(self.num_pcls):
            cc_dst_mean, cc_dst_std = cc_distance_weight_neighbours(self,
                    pcl_index, dst, pos, n_peaks = n_peaks,
                    cc_weight_exp = cc_weight_exp, cc_weight = True)
            
            tilt_mean = tilt_weighted_mean(cc_dst_mean)
            pcl_mask = score_shifts(self, tilt_mean, pcl_index,
                                    n_peaks = n_peaks,
                                    cc_weight_exp = cc_weight_exp,
                                    return_mask = True)
            self.shift_mask[:, pcl_index, :n_peaks] = pcl_mask[..., None]
            
            #this is the incorrect std to use here, but good enough for eyeballing
            if not isinstance(plot_pcl_n, bool):
                if np.isin(pcl_index, plot_pcl_n):
                    plot_weighted_pcl(self, pcl_index, tilt_mean, cc_dst_std,
                              figsize = figsize, out_dir = True)

    def write_fiducial_model(self, ali = False):
    
        shifts = self.shifts[self.shift_mask].reshape(self.num_tilts, 
                                                            self.num_pcls, 2)
        shifted_pcls = self.sorted_pcls[:, :, :3]
        shifted_pcls[:, :, :2] = (shifted_pcls[:, :, :2] - shifts)

        outmod = PEETmodel() 
        for p in range(self.num_pcls):
            #I suspect an contour is already created with PEETmodel() instance,
            #no need to add it for the first pcl
            if p != 0:
                    outmod.add_contour(0)        
            for r in range(self.num_tilts):
                outmod.add_point(0, p, shifted_pcls[r,p])
                
        outmod_name = abspath(join(self.out_dir, 'flexo_ali.fid'))
        outmod.write_model(outmod_name)
    
        if ali:
            #set image coordinate information from the given image file
            check_output('imodtrans -I %s %s %s' % 
                         (ali, outmod_name, outmod_name), shell = True)
        return outmod_name


    def nice_tilts(self, zero_tlt = False):
        #define middle tilts based on middle tilt or median CCC
        if zero_tlt:
            step = max(5, self.num_tilts/4)
            bot = int(zero_tlt - 1 - (step - 1)//2)
            #zero_tlt is numbered from 1, so take 1 off
            top = int(zero_tlt - 1 + (step - 1)//2 + 1)   
            self.tilt_subset = np.arange(bot, top)
        else:
            #pick tilts with the highest median CCCs
            tilt_medians = np.median(self.cc_values[:, :, 0], axis = 1)
            tilt_median_order = np.argsort(tilt_medians)[::-1]
            num_hi_tilts = int(np.max((3, self.num_tilts/4)))
            gappy = np.sort(tilt_median_order[:num_hi_tilts])
            self.tilt_subset = np.arange(gappy[0], gappy[-1])

                    
    def get_shift_magnitude(self, padding_shift = False,
                            n_peaks = 10, scoring = 'mean'):
        """
        get pairwise distances between peaks of neighbouring tilts
        to preserve array shape, the first set of distances are from
        reference_value
        
        scoring:
            'prod' - product of ccc
            'euclid' - euclidian distance between ccc
            'mean' - mean of ccc
        
        return 
            self.dist_matrix [num_tilts, num_pcls, n_peaks, n_peaks]
                pairwise distances between tilt x and x - 1
            self.dist_score_matrix [num_tilts, num_pcls, n_peaks, n_peaks]
                distance score
        """
        if isinstance(self.shifts, bool):
            try:
                self.read_cc_peaks()
            except:
                raise ValueError(
                    'No CC data. Use extracted_particles.read_cc_peaks()')
#        if not pairwise:
#            self.shift_magnitude = np.zeros(self.shifts.shape[:-1])
#            
#            for x in range(self.num_tilts):
#                #this is wrong, I shouldn't be starting from 0,0...
#                #maybe it's ok for initial estimation...
#                if x == 0:
#                    self.shift_magnitude[x] = np.linalg.norm(
#                        0 - self.shifts[0], axis = self.shifts.ndim - 2)
#                else:
#                    self.shift_magnitude[x] = np.linalg.norm(
#                            self.shifts[x - 1] - self.shifts[x],
#                            axis = self.shifts.ndim - 2)
#        else:          
#            

        def ccc_scoring(d1, d2):
            if scoring == 'prod':
                return np.outer(d1, np.absolute(d2))
            elif scoring == 'euclid':
                return np.sqrt(np.add.outer(d1**2, d2**2))
            elif scoring == 'mean':
                return np.add.outer(d1, d2)/2.
            else:
                raise ValueError('extracted_particles.get_shift_magnitude:'
                    + ' unrecognised scoring method "%s"' % scoring)
        
        self.dist_matrix = np.zeros((self.num_tilts,
                                self.num_pcls, n_peaks, n_peaks))
        self.dist_score_matrix = np.zeros((self.num_tilts,
                                self.num_pcls, n_peaks, n_peaks))
        
        for y in range(self.num_pcls):
            for x in range(self.num_tilts):
                if x == 0:
                    pad_values = np.zeros((n_peaks, 2))
                    if isinstance(padding_shift, bool):
                        pad_values += np.median(self.shifts[:, y, 0],
                                                axis = 0)
                    else:
                        pad_values += padding_shift
                    dist = cdist(self.shifts[x, y, :n_peaks], pad_values)
                    ccc_prod = ccc_scoring(self.cc_values[x, y, :n_peaks],
                                        self.cc_values[x, y, :n_peaks])
                    #multiply by themselves
                else:
                    dist = cdist(self.shifts[x, y, :n_peaks],
                                 self.shifts[x - 1, y, :n_peaks])
                    ccc_prod = ccc_scoring(self.cc_values[x, y, :n_peaks],
                                        self.cc_values[x - 1, y, :n_peaks])
                
                self.dist_matrix[x, y] = dist
                self.dist_score_matrix[x, y] = ccc_prod



    def dst_mat_to_shift_mask(self, mat_mask, pcl_mask = False):
        
        """
        The mask from gmm clustering represents peaks that are acceptable (distance and
        score-wise) between tilts n and n-1. Now lets make sure that these acceptable 
        peaks (between tilts n-1 and n) are also acceptable between n and n+1.
        In the distance matrix the columnds of n correspond to rows of n-1, so if any
        peaks were eliminated at tilt n (i.e. the rows are all False), the whole
        respective column has to be False in tilt n + 1. In other words the columns
        of mask at tilt n will be multiplied by np.any(tilt n-1, axis = 1) 
        (i.e. row-wise any())
        """
        if isinstance(pcl_mask, bool):
            pcl_mask = np.ones(mat_mask.shape[1], dtype = bool)
        if isinstance(mat_mask, np.ma.MaskedArray):
            #mask of masked_array has to be done with logical inverse
            mat_mask = np.logical_not(mat_mask.mask)
        #len(mat_mask) has to be 1 shorter because of pariwise comparison
        row_proj = np.any(mat_mask[:-1], axis = 3)
        row_to_col = np.repeat(row_proj[:, :, None], 10, axis = 2)
        continuous_mask = mat_mask[1:]*row_to_col
        
        #output mask to have the same shape as self.shifts (apart from the last dim...)
        shift_mask = np.zeros(self.shifts.shape[:-1], dtype = bool)
        shift_mask[1:, pcl_mask, :mat_mask.shape[2]] = np.any(continuous_mask, axis = 3)
        #fill in the missing 0th tilt mask
        shift_mask[0, pcl_mask, :mat_mask.shape[2]] = np.any(mat_mask[0], axis = 2)
        return shift_mask    

    def classify_initial_shifts(self, gmm_scalar = 10,
                                zero_tlt = False, plot = False,
                                out_dir = True,
                                shift_apix_weight0 = 1,
                                shift_apix_weight1 = 1,
                                training_peak_indices = [0, -1],
                                max_input_values = 2500,
                                n_peaks = 10,
                                gap_mask = True,
                                figsize = (12,6)):

        if out_dir and isinstance(out_dir, bool):
            out_dir = self.out_dir
        if isinstance(self.cc_values, bool):
            #can't just catch and read_cc_peaks
            raise Exception('CCC values have not been read.')
        if isinstance(self.model_3d, bool):
            try:
                self.read_3d_model()
            except:
                raise Exception('3D model has not been read.')
        if isinstance(self.dist_matrix, bool):
            #this should run if none of the previous exceptions are triggered
            #VP: this is pointless, classify_initial_shifts breaks
            #if get_shift_magnitude had not been executed already
            self.get_shift_magnitude(n_peaks = n_peaks)

        if isinstance(self.tilt_subset, bool):
            self.nice_tilts(zero_tlt = zero_tlt)

    #start with medians/std to select a trustworthy subset of particles
        med_ccc = np.median(self.cc_values[self.tilt_subset, :, 0], axis = 0)
        std_shift = (np.std(self.shifts[self.tilt_subset, :, 0, 0], axis = 0)
                + np.std(self.shifts[self.tilt_subset, :, 0, 1], axis = 0))/2
        
        #this is needed for dispaly purposes
        if plot:
            init_minmax = (np.min(med_ccc), np.max(med_ccc),
                       np.min(std_shift), np.max(std_shift))

        #gmm doesn't work well with very small/very large numbers
        med_ccc = norm(med_ccc)*gmm_scalar
        std_shift = norm(std_shift)*gmm_scalar
            
        pcl_mask, _, _ = self.bgmm_cluster(med_ccc,
                            std_shift, apix = self.apix,
                            shift_apix_weight = shift_apix_weight0,
                            prenorm_xyminmax = init_minmax,
                            suptitle = 'Initial particle GMM',
                            x_text = 'median CCC',
                            y_text = 'shift ' + r'$\sigma$',
                            out_dir = out_dir,
                            plot = plot)
    
    #now train a gmm using the highest peaks and the smallest two peaks
    #the low peaks are used as a "boundary" negative population
        n = training_peak_indices
            
        if plot:
            tmp_ccc = self.dist_score_matrix[:, :, n][:, :, :, n]
            tmp_dst = self.dist_matrix[:, :, n][:, :, :, n]
            init_minmax = (np.min(tmp_ccc), np.max(tmp_ccc),
                           np.min(tmp_dst), np.max(tmp_dst))
            tmp_ccc = tmp_dst = None
        
        norm_sc = norm(self.dist_score_matrix)*gmm_scalar
        norm_dst = norm(self.dist_matrix)*gmm_scalar


        if gap_mask:
            #gaps are filled in with 0.
            #(e.g. if dist_matrix is shape 31,300,100 and there are only 6 peaks left
            #in a cc map after filtering)
            gap_mask = self.dist_score_matrix[:, pcl_mask] != 0

            #exclude training_peak_indices that point to self.dist_score_matrix
            #with mostly zeros
            any_peak_mask = np.diagonal(np.mean(gap_mask, axis = (0,1))) > 0.1
            any_peaks = np.arange(n_peaks)[any_peak_mask]
            n = np.unique(any_peaks[n])

            train_sc = norm_sc[:, :, n][:, :, :, n]
            train_dst = norm_dst[:, :, n][:, :, :, n]
            train_sc = train_sc[:, pcl_mask][gap_mask[:, :, n][:, :, :, n]]
            train_dst = train_dst[:, pcl_mask][gap_mask[:, :, n][:, :, :, n]]
        else:
            train_sc = norm_sc[:, :, n][:, :, :, n]
            train_dst = norm_dst[:, :, n][:, :, :, n]
            train_sc = np.ravel(train_sc[:, pcl_mask])
            train_dst = np.ravel(train_dst[:, pcl_mask])

        _, map_gmm, accepted_classes = self.bgmm_cluster(train_sc,
                            train_dst, apix = self.apix,
                            shift_apix_weight = shift_apix_weight1,
                            prenorm_xyminmax = init_minmax,
                            max_input_values = max_input_values,
                            suptitle = 'Global peak GMM',
                            x_text = 'distance score',
                            y_text = 'shift ' + r'$\sigma$',
                            out_dir = out_dir,
                            plot = plot)
        
        
        in_all_sc = norm_sc[:, pcl_mask, :n_peaks, :n_peaks]
        in_all_dst = norm_dst[:, pcl_mask, :n_peaks, :n_peaks]
        acc_shape = in_all_sc.shape
        in_all_sc = np.ravel(in_all_sc)
        in_all_dst = np.ravel(in_all_dst)
        
        accepted_mask = self.predict_given_classes(map_gmm, (in_all_sc, in_all_dst),
                                              accepted_classes)
        accepted_mask = accepted_mask.reshape(acc_shape)
        if not isinstance(gap_mask, bool):
            print(gap_mask.shape)
            print(accepted_mask.shape)
            accepted_mask = np.logical_and(gap_mask, accepted_mask)
        self.shift_mask = self.dst_mat_to_shift_mask(accepted_mask,
                                                     pcl_mask = pcl_mask)
        
        if plot:
            unmasked = np.sum(self.shift_mask[:, pcl_mask], axis = (2))
            f, ax = plt.subplots(1, 2, figsize = figsize)
            t1 = np.sort(np.min(unmasked, axis = 0))
            s1 = np.min(unmasked, axis = 1)
            t3 = np.sort(np.mean(unmasked, axis = 0))
            s3 = np.mean(unmasked, axis = 1)
            
            ax[0].plot(t1, label = 'minimum number of peaks/map')
            ax[0].plot(t3, label = 'mean number of peaks/map')  
            
            ax[1].plot(s1, label = 'minimum number of peaks/map')
            ax[1].plot(s3, label = 'mean number of peaks/map')  
            
            f.suptitle('Leftover peaks after classification.')
            ax[0].set_title('Mean across tilts')
            ax[1].set_title('Mean across particles')
            ax[0].set_xlabel('sorted particles')
            ax[1].set_xlabel('tilt number')
            ax[0].legend()
            ax[1].legend()      

    def reshape_bgmm_inputs(self, inp):
        #reshape inputs to (nsamples, nfeatures)
        if isinstance(inp, tuple) and not isinstance(inp[0], (int, float)):
            if not isinstance(inp[0], np.ndarray):
                inp = [np.array(inp[x]) for x in range(len(inp))]
            if inp[0].ndim == 2:
                if inp[0].shape[1] == 1:
                    pass
                elif inp[0].shape[0] == 1:
                    inp = np.hstack(inp)
                else:
                    inp = [np.ravel(inp[x]) for x in range(len(inp))]
                    inp = np.vstack(inp).T
            else:
                inp = np.vstack(inp).T
        else:
            inp = np.array(inp)
            if inp.ndim == 1:
                inp = inp.reshape(-1, 1)
        return inp
          

    def predict_given_classes(self, gmm, samples, classes):
        """generate mask of a sample using a pre-existing gmm and known class ids
        Parameters:
                gmm: scipy.mixture.BayesianGaussianMixture object
                samples: ndarray, shape(num samples, 2)
        Returns:
                out: bool ndarray, shape (num_samples)
        """
        classes = np.array(classes)
        samples = self.reshape_bgmm_inputs(samples)
        pred = gmm.predict(samples)
    
        if classes.size > 1:
            tmp_mask = np.zeros((len(classes), len(pred)))
            for x in range(len(classes)):
                tmp_mask[x] = np.ma.masked_equal(pred, classes[x]).mask
            accepted_mask = np.any(tmp_mask, axis = 0)
        else:
            accepted_mask = np.ma.masked_equal(pred, classes).mask
        return accepted_mask



    def bgmm_cluster(self, xval, yval, apix, cutoff = 1.5e-2, n_components = 15,
                figsize = (12,12), plot = False, suptitle = 'GMM clustering',
                max_input_values = 2500,
                shift_apix_weight = 0.5,
                max_pop_size_limit = 0.1,
                x_text = 'ccc', y_text = 'shift', out_dir = True,
                limit_filter = True, prenorm_xyminmax = False):
        
        """
        """
        
        ##########
        def perpendicular_line(m, b, x2, y2):
            m2 = -1./m
            b2 = y2 + (1./m)*x2
            return m2, b2
        
        def line_project(m, b, x2, y2):
            #https://math.stackexchange.com/questions/62633/orthogonal-projection-of-a-point-onto-a-line
            m = np.array(m, dtype = np.float128)
            b = np.array(b, dtype = np.float128)
            x2 = np.array(x2, dtype = np.float128)
            y2 = np.array(y2, dtype = np.float128)
            
            m2, b2 = perpendicular_line(m, b, x2, y2)
            x3 = -(b - b2)/(m + (1./m))
            y3 = m*x3 + b
            return np.array([x3, y3]).T
        
        def parabola_y(x, cc_min = 0.05, max_sh = 3, cc_mean = 0.2, cc_mean_y = 0,
                       yoffset = 0):
            x = np.array(x, dtype = np.complex)
            max_sh -= yoffset
            def fa(cc_mean, max_s, c = 0.05):
                return (cc_mean - c)/max_s**2
            a = fa(cc_mean, max_sh, cc_min)
            cc_min = cc_min - a*(cc_mean_y - yoffset)**2
            return (np.sqrt((4*a*(x - cc_min)))/(2*a)).real + yoffset
        
        def bgmm(inp, max_iter = 500, n_components = 10,
                 weight_concentration_prior = None):       
            inp = self.reshape_bgmm_inputs(inp)
            n_components = np.min((n_components, np.max((len(inp)/10, 2))))
            gmm = BayesianGaussianMixture(n_components = n_components,
                covariance_type = 'full', max_iter = max_iter,
                weight_concentration_prior = weight_concentration_prior).fit(inp)
            return gmm  
        ################
        
        xy_val = np.vstack((xval, yval)).T
        
        if len(xval) > max_input_values:
            #train using a subset of data where nsamples > 2500
            frac = (2/3.*max_input_values)/len(xval)
            rand_mask = np.random.choice([True, False],
                                        size = len(xval), p = [frac, 1 - frac])
            xval = xval[rand_mask]
            yval = yval[rand_mask]
        
        #min/max is used a lot, precompute
        min_xval, max_xval = np.min(xval), np.max(xval)
        min_yval, max_yval = np.min(yval), np.max(yval)
        
        #train gmm
        init_gmm = bgmm((xval, yval), n_components = n_components)
    
        #order classes using weights
        weights = init_gmm.weights_
        classes = np.arange(len(weights))
        w_mask = weights > cutoff
        init_means = init_gmm.means_[w_mask]
        init_covs = init_gmm.covariances_[w_mask]
        
        #define a line with negative slope for projection
        shift_weight = apix/float(1./shift_apix_weight)
        init_regress = stats.linregress(xval, yval*shift_weight)
        r_slope = init_regress.slope
        if r_slope > 0:
            r_slope = -r_slope
        if np.absolute(r_slope) > 10000:
            #this reduces clustering precision. reduce shift_weight
            warnings.warn(
                'bgmm_cluster: Slope of projection line exceeds safe limits.')
            
        #project input values onto this line and cluster using 2 components
        proj_ccc = line_project(r_slope, init_regress.intercept,
                                xval, yval)[:, 0] #use only Y coords
        proj_gmm = bgmm(proj_ccc, n_components = 2)
        proj_means = proj_gmm.means_.reshape(2)
        proj_stds = np.sqrt(proj_gmm.covariances_.reshape(2))
        
        #now project init_gmm means so that they can be predicted using proj_gmm
        proj_init_means = line_project(r_slope,
            init_regress.intercept, init_means[:, 0], init_means[:, 1])[:, 0]
        
            #things become awkward if there is a substantial overlap between the two
            #populatinos - the accepted population can be in the middle of the other,
            #meaning that the init_gmm population with the largest proj_init_means
            #could be rejected
        narrower_cls = np.argsort(proj_stds)[0]
        broader_cls = np.argsort(proj_stds)[1]
        overlap_limits = (proj_means[narrower_cls] - proj_stds[narrower_cls],
                          proj_means[narrower_cls] + proj_stds[narrower_cls])
            #if the two populations overlap, keep all init_gmm.  This means that 
            #limit filter is triggered
        if (overlap_limits[0] < proj_means[broader_cls] < overlap_limits[1]):
            proj_cls1_mask = np.ones(proj_init_means.shape, dtype = bool)
            limit_filter = True
            ##alternative way would be just to take the largest...
            #proj_cls1_mask = np.array(np.where(
            #        proj_means == np.max(proj_means), proj_means, 0), dtype = bool)
        else:
            #otherwise take all proj_init_means predicted to belong to the proj_gmm
            #population with the largest mean
            proj_cls1_id = np.where(proj_means == np.max(proj_means))[0][0]  
            #then get the init_means that fall within the high ccc class
            pred_init_means = proj_gmm.predict(proj_init_means.reshape(-1, 1))
            proj_cls1_mask = np.ma.masked_equal(pred_init_means, proj_cls1_id).mask
        
        #generate mask of accepted init_gmm classes
        accepted_init_classes = classes[w_mask][proj_cls1_mask]
        #now remove init_gmm accepted classes that are worse than 5? sigma of the mean
        #of the "best" class. Do this in both X and Y, but it's likely that only shifts
        #can have this much spread. The "best" class is the one with the highest
        #projected_init_mean, unless it is 1/max_size_limit smaller than the
        #second best class etc.
        #most populated class
        if accepted_init_classes.size > 1 and limit_filter:
            #get class sizes
            cls_sizes = np.array([
                    np.sum(self.predict_given_classes(init_gmm, xy_val, int(x)))
                    for x in list(accepted_init_classes)])
            
            #scale population sizes by the order of their means.
            #Second is max_size_limit times smaller than first etc
            cls_order = np.argsort(init_means[proj_cls1_mask][:, 1])
            cls_size_scalar = np.zeros(len(cls_order))
            for x in range(len(cls_order)):
                cls_size_scalar[cls_order[x]] = (1/max_pop_size_limit)**-x          
            scaled_cls_sizes = cls_sizes*cls_size_scalar
            best_accepted_cls = np.where(scaled_cls_sizes == np.max(scaled_cls_sizes))[0][0]
            
            #get cls stds and means
            x_std = np.sqrt(init_covs[proj_cls1_mask][:, 0, 0])
            y_std = np.sqrt(init_covs[proj_cls1_mask][:, 1, 1])
            cls_std = np.vstack((x_std, y_std)).T
            accepted_init_means = init_means[proj_cls1_mask]
            
            #get min x and max y limit
            best_mean = accepted_init_means[best_accepted_cls]
            best_std = cls_std[best_accepted_cls]
            para_xlim = best_mean[0] - 3*best_std[0]
            para_ylim = best_mean[1] + 5*best_std[1]
            
            #get limiting curve parameters
            para_x = np.linspace(min_xval, max_xval, 100)
            para_y = parabola_y(para_x, para_xlim, para_ylim,
                                best_mean[0], best_mean[1], min_yval)
            
            #first, anything worse than para_xlim is removed 
            limit_mask = init_means[proj_cls1_mask][:, 0] > para_xlim
            considered_means = init_means[proj_cls1_mask][limit_mask]
            #then take means below the line
            score = parabola_y(considered_means[:, 0], para_xlim, para_ylim,
                best_mean[0], best_mean[1], min_yval) - considered_means[:, 1]
            score_mask = score > 0
            limit_mask[limit_mask] = score_mask
            
            accepted_init_classes = accepted_init_classes[limit_mask]
        else:
            limit_mask = np.ones(len(accepted_init_classes), dtype = bool)
        
        #predict all input
        accepted_mask = self.predict_given_classes(init_gmm,
                                              xy_val, accepted_init_classes)

        
        if out_dir:
            if isinstance(out_dir, bool):
                out_dir = self.out_dir
        if plot or out_dir:
            proj_weight = proj_gmm.weights_
            proj_mean = proj_gmm.means_
            proj_covar = proj_gmm.covariances_
            #proj axis
            ax_xaxis = np.linspace(np.min(proj_ccc), np.max(proj_ccc), 100) 
            #grid for contours
            xx = np.linspace(min_xval, max_xval, 50)
            yy = np.linspace(min_yval, max_yval, 50)
            X, Y = np.meshgrid(xx, yy)
            zr = np.vstack([X.ravel(), Y.ravel()]).T

            #this should set the probability area to 1
            #except it's not right...
            Z = np.exp(init_gmm.score_samples(zr))*(max_xval - min_xval)*(max_yval - min_yval)
            Z = Z.reshape(X.shape)
            hist_bins = (np.linspace(min_xval, max_xval, 40),
                         np.linspace(min_yval, max_yval, 40))
            
            #move lines to roughly the middle
            mid_x = np.mean((min_xval, max_xval))
            mid_y = np.mean((min_yval, max_yval))
            new_intercept = mid_y - r_slope*mid_x
            m2, b2 = perpendicular_line(r_slope, init_regress.intercept,
                                        mid_x, mid_y)
            inv_mask = np.logical_not(proj_cls1_mask)
            inv_limit_mask = np.logical_not(limit_mask)
            
            f, ax = plt.subplots(2, 2, figsize = figsize)
            
            ax[0, 0].hist2d(xval, yval, bins = hist_bins, cmap = 'Greys')
            ax[0, 0].plot(xval, new_intercept + r_slope*xval, color = 'black',
                          label='projection line')
            ax[0, 0].plot(xval, b2 + m2*xval, color = 'silver',
                          label = 'projection direction')
            ax[0, 0].scatter(init_means[:, 0][proj_cls1_mask][limit_mask],
                             init_means[:, 1][proj_cls1_mask][limit_mask],
                             c = 'r',  label = 'mean of accepted classes')
            if np.any(inv_limit_mask):
                ax[0, 0].scatter(init_means[:, 0][proj_cls1_mask][inv_limit_mask],
                             init_means[:, 1][proj_cls1_mask][inv_limit_mask],
                             c = 'purple',
                             label = 'mean of secondary rejected classes')
                ax[0, 0].plot(para_x, para_y, 'k--', alpha = 0.5,
                              label = 'secondary rejection limit')
            ax[0, 0].scatter(init_means[:, 0][inv_mask],
                             init_means[:, 1][inv_mask], c = 'tab:blue',
                             label = 'mean of rejected classes')
                
            #contours
            con = ax[0, 1].contour(X, Y, Z, cmap = 'Spectral')
            ax[0, 1].clabel(con, inline=1, fontsize=10, fmt='%1.1f')
            
            #projected data
            #this seems to solve a np.hist bug:
            proj_ccc = np.array(proj_ccc, dtype = 'float32')
            ax[1, 0].hist(proj_ccc, 30, density = True,
                          alpha = 0.3, color = 'black')
            for y in range(len(proj_mean)):
                pdf_y = proj_weight[y]*stats.norm.pdf(ax_xaxis, proj_mean[y],
                                   np.sqrt(proj_covar[y])).ravel()
                ax[1, 0].plot(
                        ax_xaxis, pdf_y, color = 'black')
            ax[1, 0].scatter(proj_init_means[proj_cls1_mask],
                             np.zeros(np.sum(proj_cls1_mask)), c = 'r',
                             label = 'mean of accepted classes')
            ax[1, 0].scatter(proj_init_means[inv_mask],
                             np.zeros(np.sum(inv_mask)), c = 'tab:blue',
                             label = 'mean of rejected classes')
            
            #output data
            ax[1, 1].hist2d(xy_val[:, 0][accepted_mask],
                            xy_val[:, 1][accepted_mask],
                            bins = hist_bins, cmap = 'Greys')
            
            #labels
            f.suptitle(suptitle)
            ax[0, 0].set_title('2D hist of input')
            ax[0, 0].set_ylabel(y_text)
            ax[0, 0].set_xlabel(x_text)
            ax[0, 0].legend(loc = 'upper right')
            #
            ax[0, 1].set_title('gaussian mixture model PDF')
            ax[0, 1].set_ylabel(y_text)
            ax[0, 1].set_xlabel(x_text)    
            #
            ax[1, 0].legend()
            ax[1, 0].set_title('projected input PDF')
            ax[1, 0].set_xlabel('projected %s [arbitrary units]' % x_text)
            #
            #ax[1, 1].set_ylim(ax[0, 0].get_ylim())
            #ax[1, 1].set_xlim(ax[0, 0].get_xlim())
            ax[1, 1].set_title('2D hist of accepted data')
            ax[1, 1].set_ylabel(y_text)
            ax[1, 1].set_xlabel(x_text)
            
            #change ticks to pre-normalisation
            if not isinstance(prenorm_xyminmax, bool):
                xt = np.linspace(min_xval, max_xval, 5)
                yt = np.linspace(min_yval, max_yval, 5)
                xt_label = np.round(np.linspace(prenorm_xyminmax[0],
                                    prenorm_xyminmax[1], 5), decimals = 2)
                yt_label = np.round(np.linspace(prenorm_xyminmax[2],
                                    prenorm_xyminmax[3], 5), decimals = 1)
                plt.setp((ax[0, 0],ax[0, 1],ax[1, 1]), xticks = xt,
                         xticklabels = xt_label, yticks = yt,
                         yticklabels = yt_label)
            
            if out_dir:
                f.savefig(join(self.out_dir,
                               '%s.png' % ('_').join((suptitle).split(' '))))
                plt.close()                
        return accepted_mask, init_gmm, accepted_init_classes
                    
        
            
        
                    

    def plot_median_cc_vs_tilt(self, out_dir = True, figsize = (12,8)):
        #x values are not continuous but easier to look at as line plot...
        #gaps in X (tilts) are filled in and not ommited.  Something to play 
        #with in the future...
        if out_dir and isinstance(out_dir, bool):
            out_dir = self.out_dir
        f, ax = plt.subplots(1, figsize = figsize)
        peak_median = np.median(self.cc_values[:, :, 0], axis = 1)
        q25, q75 = np.percentile(self.cc_values[:, : ,0], [25, 75], axis = 1)
        pmin = np.min(self.cc_values[:, :, 0], axis = 1)
        pmax = np.max(self.cc_values[:, :, 0], axis = 1)
        xvals = np.array(self.tilt_angles, dtype = int)
        if self.tilt_angles[0] > self.tilt_angles[-1]:
            plt.gca().invert_xaxis()
        ax.plot(xvals, peak_median, color = 'midnightblue',
                label = 'median')
        ax.fill_between(xvals, pmax, pmin, color = 'lightsteelblue',
                         label = 'max/min')
        ax.fill_between(xvals, q25, q75, color = 'cornflowerblue',
                         label = 'quartiles')
        ax.set_xlabel('specimen tilt [degrees]')
        ax.set_ylabel('CCC')
        ax.legend(loc = 'upper right')
        f.suptitle('CCC median, quartiles and min/max.')
        if out_dir:
            plt.savefig(join(self.out_dir, 'median_tilt_cc_values.png'))
            plt.close()
            
    def plot_global_shifts(self, out_dir = True, figsize = (15,4)):
        #x values are not continuous but easier to look at as line plot...
        #gaps in X (tilts) are filled in and not ommited.  Something to play 
        #with in the future...
        if out_dir and isinstance(out_dir, bool):
            out_dir = self.out_dir
            
        f, ax = plt.subplots(1, 3, figsize = figsize, sharey = True)
        med_sh = np.ma.median(self.shifts[:, :, 0], axis = 1)
        mean_sh = np.ma.mean(self.shifts[:, :, 0], axis = 1)
        std_sh = np.ma.std(self.shifts[:, :, 0], axis = 1)
        xvals = np.array(self.tilt_angles, dtype = int)
        ax[0].plot(xvals, med_sh[:, 0], label = 'x shift')
        ax[0].plot(xvals, med_sh[:, 1], label = 'y shift')
        ax[1].plot(xvals, mean_sh[:, 0], label = 'x shift')
        ax[1].plot(xvals, mean_sh[:, 1], label = 'y shift')
        ax[2].plot(xvals, std_sh[:, 0], label = 'x shift')
        ax[2].plot(xvals, std_sh[:, 1], label = 'y shift')
        
        for x in range(len(ax)):
            ax[x].set_xlabel('specimen tilt [degrees]')
            ax[x].legend(loc = 'upper right')
        ax[0].set_ylabel('shift [px]')
        ax[0].set_title('median shift')
        ax[1].set_title('mean shift')
        ax[2].set_title('stdev')
        if out_dir:
            plt.savefig(join(self.out_dir, 'global_shifts.png'))
            plt.close()


    def plot_particle_med_cc(self, out_dir = True, figsize = (12,8)):
        if out_dir and isinstance(out_dir, bool):
            out_dir = self.out_dir
        f, ax = plt.subplots(1, figsize = figsize)
        peak_median = np.median(self.cc_values[:, :, 0], axis = 0)
        order = np.argsort(peak_median)[::-1]
        peak_median = peak_median[order]
        q25, q75 = np.percentile(self.cc_values[:, :, 0],[25, 75], axis = 0)
        q25, q75 = q25[order], q75[order]
        pmin = np.min(self.cc_values[:, :, 0], axis = 0)[order]
        pmax = np.max(self.cc_values[:, :, 0], axis = 0)[order]
        xvals = np.array(self.pcl_ids, dtype = int)[order]
        ax.plot(list(range(len(xvals))), peak_median, color = 'midnightblue',
                label = 'median')
        ax.fill_between(list(range(len(xvals))), pmax, pmin,
                        color = 'lightsteelblue', label = 'max/min')
        ax.fill_between(list(range(len(xvals))), q25, q75,
                        color = 'cornflowerblue', label = 'quartiles')
        ax.set(xlabel = 'ordered particles', ylabel = 'CCC')
        ax.legend(loc = 'upper right')
#        ax.set(xlabel = 'ordered particles')
#        plt.ylabel('CCC')
        f.suptitle('CCC median, quartiles and min/max.')
        if out_dir:
            plt.savefig(join(self.out_dir, 'tilt_cc_values.png'))
            plt.close()
     
    def plot_shift_magnitude_v_cc(self, n_peaks = 1,
            cc_multiplier = 'auto', out_dir = True,
            figsize = (12,8)):
        """
        
        cc_multiplier [int] scaling of CCC axis (x axis)
            default 'auto': cc_multiplier = max(y axis)/max(x axis)
        """
        

        if isinstance(self.dist_matrix, bool):
            self.get_shift_magnitude()
        if out_dir and isinstance(out_dir, bool):
            out_dir = self.out_dir
        n_peaks = min(n_peaks, self.dist_matrix.shape[2])
        
        def remove_zero_entries(n_peaks):
            truncated_shifts = self.dist_matrix[:, :, :n_peaks, :n_peaks]
            truncated_cc_values = self.dist_score_matrix[:, :, :n_peaks, :n_peaks]
            nonzero_shifts = truncated_shifts[truncated_cc_values != 0.]
            nonzero_cc_values = truncated_cc_values[truncated_cc_values != 0.]
            return nonzero_cc_values, nonzero_shifts
        
        def heatmap2d(n_peaks, cc_multiplier):
            if n_peaks == False:
                n_peaks = self.dist_matrix.shape[2]
            nonzero_cc_values, nonzero_shifts = remove_zero_entries(n_peaks)
            if cc_multiplier == 'auto':
                cc_multiplier = np.max(nonzero_shifts)/np.max(nonzero_cc_values)
            heatmap, xedges, yedges = np.histogram2d(
                                    nonzero_cc_values*cc_multiplier,
                                    nonzero_shifts, bins = 50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]            
            return heatmap.T, extent, cc_multiplier

#       VP edit: This is not going to work since dist_matrix is 
#               shape [num_tilts, num_pcls, n_peaks, n_peaks]
#               and not [num_tilts, num_pcls, n_peaks]
#        if scatter_3d:
#            if n_peaks == False:
#                n_peaks = self.dist_matrix.shape[2]
#            nonzero_cc_values, nonzero_shifts = remove_zero_entries()
#            z_vals = np.ones(self.dist_matrix[:, :, :n_peaks].shape
#                 )*range(self.dist_matrix[:, :, :n_peaks].shape[-1])
#            plt.figure()
#            ax = plt.subplot(111, projection = '3d')
#            if cc_multiplier == 'auto':
#                cc_multiplier = np.max(nonzero_shifts)/np.max(nonzero_cc_values)
#            ax.scatter(nonzero_cc_values*cc_multiplier, nonzero_shifts,
#                       z_vals)
#        else:
        f, ax = plt.subplots(1, 2, figsize = figsize)
        ax[0].set(xlabel = 'CCC',
                  ylabel = 'shift magnitude [pixels]')
        ax[1].set(xlabel = 'CCC')
        heatmap, extent, cc_multiplier = heatmap2d(False, cc_multiplier)
        ax[0].title.set_text('all CCCs')
        ax[0].imshow(heatmap, extent = extent,
                      origin = 'lower', cmap = 'viridis')
        heatmap, extent, cc_multiplier = heatmap2d(n_peaks, cc_multiplier)
        ax[1].title.set_text('%s largest CCC(s)' % n_peaks)
        ax[1].imshow(heatmap, extent = extent,
                      origin = 'lower', cmap = 'viridis')
        xticks0 = ax[0].get_xticks()
        xticks2 = ax[1].get_xticks()
        ax[0].set_xticklabels(
                np.round(xticks0/cc_multiplier, decimals = 3))
        ax[1].set_xticklabels(
                np.round(xticks2/cc_multiplier, decimals = 3))
        
        if out_dir:
            plt.savefig(join(self.out_dir,
                             'shift_magnitude_v_ccc.png'))
            plt.close()
            
    
            
        
