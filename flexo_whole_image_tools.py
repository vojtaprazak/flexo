# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:27:36 2022

@author: vojta
"""
import os
from os.path import split, join, isdir, isfile
from copy import deepcopy
import mrcfile
import MapParser_f32_new
from PEETModelParser import PEETmodel
from PEETMotiveList import PEETMotiveList
from EMMap_noheadoverwrite import Map
import numpy as np
from scipy.ndimage.interpolation import shift, zoom
import warnings
from flexo_tools import write_mrc, get_apix_and_size
from subprocess import check_output
from skimage.filters import threshold_yen, threshold_isodata
from scipy.ndimage.morphology import grey_dilation, grey_erosion
from scipy.ndimage import convolve

from definite_functions_for_flexo import run_generic_process, imodscript, corrsearch
from IMOD_comfile import IMOD_comfile

def replace_pcles(average_map, tomo_size, csv_file, mod_file, outfile, apix,
                  rotx = True):

    #remove inputs: group_mask = False, extra_bin = False, average_volume_binning = 1
    """
    Plotback particles using IMOD model and motive list. average_map voxel
    size does not necessarily need to match the tomogram pizel size.
    
    Expecting IMOD model in rotated orientation (not _full.rec)

    Parameters
    ----------
    average_map : str or MapParser
        Density map of reference particle. White on black, segmented.
    tomo_size : list of three ints
        XYZ tomogram size.
    csv_file : str or PEETMotiveList
        PEET motive list.
    mod_file : str or PEETmodel
        PEET model file.
    outfile : str
        Path to output file.
    apix : float
        Voxel size (isotropic only) of tomogram.
    average_volume_binning : float, optional
        Average binning relative to tomogram binning. E.g. 0.5 with average
        pixel size 1 and tomogram pizel size 2. The default is 1.
    rotx : bool, optional
        If True, write in rotated orientation (90 degrees around X axis).
        If False, write in _full.rec orientation (default initial tilt output).
        The input model should be in the corresponding orientation.
        The default is False.

    Returns
    -------
    None.

    """ 
    
    if isinstance(average_map, str):
        ave = MapParser_f32_new.MapParser.readMRC(average_map)
    else:
        ave = average_map #intended for parallelisation
    
    average_apix = ave.apix
    average_volume_binning = np.round(average_apix/apix, decimals = 3)
    if average_volume_binning > 1:
        warnings.warn('Average volume binning is %s, meaning that a larger pixel size map is being placed in a lower pixel size volume. This is suspicious.' % average_volume_binning)
    xsize = ave.x_size()
    ysize = ave.y_size()
    zsize = ave.z_size()  
    
    if average_volume_binning != 1:
        xsize = xsize * average_volume_binning
        ysize = ysize * average_volume_binning
        zsize = zsize * average_volume_binning
    
    if type(csv_file) == str:
        motl = PEETMotiveList(csv_file)
    elif isinstance(csv_file, PEETMotiveList):
        motl = csv_file
        
    if type(mod_file) == str:
        mod = PEETmodel(mod_file).get_all_points()
    elif isinstance(mod_file, PEETmodel):
        mod = mod_file       
        
    mat_list = motl.angles_to_rot_matrix()
    
    offsets = motl.get_all_offsets()
    if np.max(offsets) != 0.:
        warnings.warn('Motive list contains non-zero offsets. These WILL NOT be added to particle positions.')
    offsets *= 0
    
    border = int(xsize//2)
    tomo_size = np.array(tomo_size, dtype = int) + int(border*2)
    mod += border
    tomo = Map(np.zeros(np.flip(tomo_size, 0), dtype='float32'),[0,0,0],
               apix,'replace_pcles') #replaced ave.apix

    if mod.max() > np.array(tomo_size).max():
        print('Maximum model coordinates exceed volume size. %s %s'\
        % (mod.max(),  np.array(tomo_size).max()))
    if mod.ndim == 1:
        mod = mod[None]

    for p in range(len(mod)):
        x_pos = int(round(offsets[p][0] + mod[p][0]))
        y_pos = int(round(offsets[p][1] + mod[p][1]))
        z_pos = int(round(offsets[p][2] + mod[p][2]))

        x_offset = offsets[p][0] + mod[p][0] - x_pos
        y_offset = offsets[p][1] + mod[p][1] - y_pos
        z_offset = offsets[p][2] + mod[p][2] - z_pos     
        
        new_ave = ave.copy()
        shifted_ave = new_ave.rotate_by_matrix(mat_list[p], ave.centre(), cval = 0)
        
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
        
        # x_p_min = np.math.floor(max(0, x_pos - xsize / 2))
        # x_p_max = np.math.ceil(min(tomo_size[0], x_d + x_pos + xsize / 2))
        # y_p_min = np.math.floor(max(0, y_pos - ysize / 2))
        # y_p_max = np.math.ceil(min(tomo_size[1], y_d + y_pos + ysize / 2))
        # z_p_min = np.math.floor(max(0, z_pos - zsize / 2))
        # z_p_max = np.math.ceil(min(tomo_size[2], z_d + z_pos + zsize / 2))

        x_p_min = np.math.ceil(max(0, x_pos - xsize / 2))
        x_p_max = np.math.floor(min(tomo_size[0], x_d + x_pos + xsize / 2))
        y_p_min = np.math.ceil(max(0, y_pos - ysize / 2))
        y_p_max = np.math.floor(min(tomo_size[1], y_d + y_pos + ysize / 2))
        z_p_min = np.math.ceil(max(0, z_pos - zsize / 2))
        z_p_max = np.math.floor(min(tomo_size[2], z_d + z_pos + zsize / 2))
        
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
            #27 
            tomo.fullMap[z_p_min:z_p_max, y_p_min:y_p_max, x_p_min:x_p_max]\
        += shifted_ave.fullMap[z_n_min:z_n_max, y_n_min:y_n_max, x_n_min:x_n_max]
        except:
            raise ValueError('Particle model coordinates are outside specified region bounds.  Please make sure input 3D model fits the input tomogram.')
    tomo.fullMap = tomo.fullMap[border:-border, border:-border, border:-border]
    if not rotx:
        tomo.fullMap = np.rot90(-tomo.fullMap, k = -1)
    else:
        tomo.fullMap = -tomo.fullMap
    if outfile:
        print('Writing MRC file %s' % (outfile))
    
        write_mrc(outfile, tomo.fullMap)
        
def reproject_volume(output_file, tomo = False, ali = False, tlt = False,
                     thickness = False, add_tilt_params = [], tiltcom = False,
                     excludelist = []):
    """
    Reproject a volume into tilt series. Expecting volume in "full" orientation.
    This funciton operates two modes: 
    If tiltcom is specified (IMOD_comfile),
    it will use it's attributes and ignore other optional arguments apart from 
    tomo (volume to reproject).
    Otherwise, it requires tomo, ali, tlt and thickness.'

    Parameters
    ----------
    output_file : str
        Path to output file.
    tomo : str, optional
        Path to input (_full.rec) tomogram. The default is False.
    ali : str, optional
        Path to aligned tilt series. The default is False.
    tlt : str, optional
        Path to tilt angle file (.tlt). The default is False.
    thickness : int, optional
        Tomogram thickness. The default is False.
    add_tilt_params : list of str, optional
        Optional argumetns to pass to IMOD tilt. The default is [].
    tiltcom : IMOD_comfile, optional
        IMOD_comfile instance with tilt.com parameters. The default is False.
    excludelist : list if int, optional
        List of views to exclude, numbered from 1. The default is [].
    write_comfile : bool, optional
        If true and IMOD_comfile is specified, write command file. The default is False.


    Returns
    -------
    None.

    """
    #operates in 2 modes: with and without tiltcom. tiltcom overrides optional inputs
        
    if not tiltcom and (not ali or not tlt or not tomo):
        raise Exception('Either tiltcom (IMOD_comfile with tilt.com parameters),' +
                        ' or tomogram, aligned stack and tilt file is required.')

    if tiltcom:
        if isinstance(tiltcom, str):
            tiltcom = IMOD_comfile(split(tiltcom)[0], split(tiltcom)[1])
        exclude_key = ['OutputFile']
        add_tilt_params = tiltcom.get_command_list(
            append_to_exclude_keys = exclude_key)[1:] #exclude 'tilt'
        if not tomo:
            tomo = tiltcom.dict['OutputFile']
        tlt = tiltcom.dict['TILTFILE']
        excludelist = tiltcom.excludelist

    str_tilt_angles = [str(x.strip('\n\r').strip(' ')) for x in open(tlt)]
    str_tilt_angles = [str_tilt_angles[x] for x in range(len(str_tilt_angles))
                       if x + 1 not in excludelist] #numbered from 1

    # if write_comfile and tiltcom:
    #     tiltcom.dict['RecFileToReproject'] = tiltcom.dict['OutputFile']
    #     tiltcom.separator_dict['RecFileToReproject'] = None
    #     tiltcom.dict['REPROJECT'] = [float(x) for x in str_tilt_angles]
    #     tiltcom.separator_dict['REPROJECT'] = ','
    #     tiltcom.dict['OutputFile'] = output_file
    #     out_dir = os.path.split(output_file)[0]
    #     if not out_dir:
    #         out_dir = os.getcwd()
    
    cmd_list = ['tilt',
                '-REPROJECT', (',').join(str_tilt_angles),
                '-RecFileToReproject', tomo,
                '-OutputFile', output_file
                ]    
    if not tiltcom:
        cmd_list.extend(['-InputProjections', ali,
                         '-TILTFILE', tlt,
                         '-THICKNESS', thickness])

    cmd_list.extend(add_tilt_params)
    #check_output((' ').join(cmd_list), shell = True)
    run_generic_process(cmd_list)
    
    
def mask_from_plotback(volume, out_mask, size = 5,
                           lamella_mask_path = False,
                           mean_filter = True,
                           invert = False):
    """
    Thresholds a plotback and generates a smooth mask.

    Parameters
    ----------
    volume : str
        Path to plotback.
    out_mask : str
        Path to output file.
    size : int, optional
        Dilation size. Negative values erode. The default is 5.
    lamella_mask_path : str, optional
        Path to lamella mask, which will be multiplied with the output mask.
        The default is False.

    Returns
    -------
    None.

    """

    if type(volume) == str:
        m = deepcopy(mrcfile.open(volume).data)
    else:
        m = volume
    
    #threshold in a sensible? way.  
    #First flatten volume into an image to get thr
    shortest_axis = np.where(np.array(m.shape) == min(m.shape))[0][0]
    thr = threshold_yen(np.mean(m, axis = shortest_axis))
    m = m < thr

    if size < 0:
        size = -size
        m = grey_erosion(m, size)
    else:
        m = grey_dilation(m, size)
    if invert:
        m = np.logical_not(m)
    write_mrc(out_mask, m)

    tmp_mask = out_mask + '~'
    #smooth edges
    os.rename(out_mask, tmp_mask)
    if not mean_filter:
        check_output('mtffilter -3 -l 0.001,0.03 %s %s' %
                     (tmp_mask, out_mask), shell = True)   
    else:
        check_output('clip smooth -n -3 -l 7 %s %s' % (tmp_mask, out_mask), shell = True)
        warnings.warn('3d mask smooting is hardcoded')
    os.remove(tmp_mask)
    os.rename(out_mask, tmp_mask)
    check_output('newstack -scale 0,1 %s %s' % (tmp_mask, out_mask), shell = True)
    os.remove(tmp_mask)

    if lamella_mask_path:
        os.rename(out_mask, tmp_mask)
        check_output('clip multiply %s %s %s' % (tmp_mask,
                    lamella_mask_path, out_mask), shell = True)   
        
def get2d_mask_from_plotback(ts, out_2dmask, dilation = 10):
    """
    Generate a soft-edge mask from reprojected plotback using threshold_yen.

    Parameters
    ----------
    ts : str
        Path to reprojected plotback.
    out_2dmask : str
        Path to output mask.
    dilation : TYPE, optional
        Size of border around thresholded object. The default is 10.

    Returns
    -------
    None.

    """
    
    mf = int(np.ceil(dilation*0.5))
    
    m = deepcopy(mrcfile.open(ts).data)
    for t in range(len(m)):
        thr = threshold_isodata(m[t])
        m[t] = m[t] < thr
        m[t] = grey_dilation(m[t], dilation)
        m[t] = convolve(m[t], np.ones((mf,mf)))
        
    m = m/mf**2 # scale to max 1
        
    write_mrc(out_2dmask, m)
        

def match_tomos(ref_tiltcom, query_tiltcom, out_dir,
                niters = 3, angrange = 20, plot = True):
    """
    Searches for relative orientations of two tomograms with slightly different
    orientations. Inputs are two tilt comfiles in the same directory, the
    comfiles and output tomograms should have different names. The outputs
    can then be fed to align.com

    Parameters
    ----------
    ref_tiltcom : str
        Name of "reference" tilt comfile.
    query_tiltcom : str
        Name of tilt comfile to be aligned.
    out_dir : TYPE
        DESCRIPTION.
    niters : TYPE, optional
        DESCRIPTION. The default is 3.
    angrange : TYPE, optional
        DESCRIPTION. The default is 20.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    SHIFT : TYPE
        DESCRIPTION.
    OFFSET : TYPE
        DESCRIPTION.
    global_xtilt : TYPE
        DESCRIPTION.

    """
    
    #two comfiles
    
    """
    returns translation and rotation between two tomograms:
        1) extracts parameters from rec_dir and out_dir comfiles
        2) makes (binned) versions
        3) checks rotation and translation by comparing strips of each tomo
             or use 3D  cc (corrsearch3d), use_corr = True
    How tiny: additional binning relative to input tomo
    works fine at bin 16 (60 apix), accurate to ~3 unbinned pixels and 0.2 degrees
    copy_orig [str] path to correctly binned ref tomo for matching
        if not False, skips reference tomo generation
    """

    #angrange = 20 #+/- 10 degrees

    imodscript(ref_tiltcom, out_dir)  

    imodscript(query_tiltcom, out_dir)  

    rtiltcom = IMOD_comfile(out_dir, ref_tiltcom)
    qtiltcom = IMOD_comfile(out_dir, query_tiltcom)
    
    if 'SHIFT' not in qtiltcom.dict.keys():
        qtiltcom.dict['SHIFT'] = [0.0, 0.0]
        qtiltcom.separator_dict['SHIFT'] = ' '
    if 'OFFSET' not in qtiltcom.dict.keys():
        qtiltcom.dict['OFFSET'] = 0.0
        qtiltcom.separator_dict['OFFSET'] = None
    if 'XAXISTILT' not in qtiltcom.dict.keys():
        qtiltcom.dict['XAXISTILT'] = 0.0
        qtiltcom.separator_dict['SHIFT'] = None     
    
    for i in range(1, niters + 1):
        nSHIFT, nOFFSET, nglobal_xtilt = corrsearch(rtiltcom.dict['OutputFile'],
                                                 qtiltcom.dict['OutputFile'], out_dir,
                                                 iter_n = i,
                                                 plot = plot)
            
        SHIFT = np.round(np.array(qtiltcom.dict['SHIFT'])
                         - nSHIFT*qtiltcom.dict['IMAGEBINNED'], decimals = 2)
        OFFSET = np.round((qtiltcom.dict['OFFSET'] - nOFFSET), decimals = 2)
        global_xtilt = np.round((qtiltcom.dict['XAXISTILT'] - nglobal_xtilt), decimals = 2)
        # print('iter_n %s' % i)
        # print('corrsearch calc SHIFT, OFFSET, global_xtilt %s,%s,%s'
        #       % (nSHIFT, nOFFSET, nglobal_xtilt*qtiltcom.dict['IMAGEBINNED']))
        # print('corrsearch curr SHIFT, OFFSET, global_xtilt %s,%s,%s'
        #       % (SHIFT, OFFSET, global_xtilt))
        
        new_qtiltcom = deepcopy(qtiltcom)
        new_qtiltcom.dict['OFFSET'] = OFFSET
        new_qtiltcom.dict['SHIFT'] = SHIFT
        new_qtiltcom.dict['XAXISTILT'] = global_xtilt
        new_qtiltcom.write_comfile(out_dir)

        if i != niters:
            imodscript(query_tiltcom, out_dir) 

    return SHIFT, OFFSET, global_xtilt        

def tomo_subtraction(tilt_comfile, out_dir, mask = False, plotback3d = False, ali = False,
                     iterations = 1, supersample = True, dilation_size = 0,
                     fakesirt = 20
                     ):

    def update_comdict(orig_dict, mod_dict):
        for key in mod_dict.keys():
            orig_dict[key] = mod_dict[key]
        return orig_dict    
    
    def subtract_mean_density(tomo):
        mean = float(check_output('header -mean %s' % tomo, shell = True).split()[0])
        #mean = float(out[0])
        mean = -mean
        tmp = tomo + '~'
        if isfile(tmp):
            os.remove(tmp)
        check_output('newstack -multadd 1,%s %s %s' % (mean, tomo, tmp), shell = True)
        os.rename(tmp, tomo)
        
    if not isdir(out_dir):
        os.makedirs(out_dir)
    
    if isinstance(tilt_comfile, IMOD_comfile):
        tiltcom = tilt_comfile
    else:
        tiltcom = IMOD_comfile(split(tilt_comfile)[0], 'tilt.com')
        
    
    if not mask:
        if plotback3d:
            mask = join(out_dir, 'mask.mrc')
            mask_from_plotback(plotback3d, mask, size = dilation_size, invert = True)
        else:
            raise Exception('mask or 3d plotback for mask generation required')
    
    if not ali:
        ali = tiltcom.dict['InputProjections']
    
    #ali_mode = int(check_output('header -mode %s' % (ali), shell = True).split()[0])
    out_tiltcom = deepcopy(tiltcom)
    for ite in range(iterations + 1):
        
        out_tom = join(out_dir, 'tomo%02d.mrc' % ite)
        prev_iter_tom = join(out_dir, 'tomo%02d.mrc' % (ite - 1))
        rep_tom = join(out_dir, 'rep%02d.mrc' % ite)
        sub_ts = join(out_dir, 'ts%02d.mrc' % ite)
        prev_iter_ts = join(out_dir, 'ts%02d.mrc' % (ite - 1))
        
        a_dict = {'OutputFile': out_tom,
              'InputProjections': sub_ts
                  }
            
        if ite != 0:
            subtract_mean_density(prev_iter_tom)
            tmp = prev_iter_tom + '~'
            if isfile(tmp):
                os.remove(tmp)
            check_output('clip multiply %s %s %s' % (prev_iter_tom, mask, tmp), shell = True)
            os.rename(tmp, prev_iter_tom)
            
            reproject_volume(rep_tom, tomo = prev_iter_tom, tiltcom = out_tiltcom)

            check_output('clip subtract %s %s %s' % (prev_iter_ts, rep_tom, sub_ts
                                                     ), shell = True)
            
        elif ite == 0:
            check_output('newstack -mode 2 %s %s' % (ali, sub_ts), shell = True)
            # if os.path.islink(sub_ts):
            #     os.unlink(sub_ts)
            # os.symlink(ali, sub_ts)

            if supersample:
                if 'SuperSampleFactor' not in out_tiltcom.dict:
                    a_dict['SuperSampleFactor'] = 2
                if 'ExpandInputLines' not in out_tiltcom.dict:
                    a_dict['ExpandInputLines'] = None

            if fakesirt:
                if 'FakeSIRTiterations'  not in out_tiltcom.dict:
                    a_dict['FakeSIRTiterations'] = fakesirt
                
            if 'RADIAL' in out_tiltcom.dict:
                out_tiltcom.dict['RADIAL'] = [0.5,0.05]
                a_dict['SuperSampleFactor'] = 2
        
        out_tiltcom.dict = update_comdict(out_tiltcom.dict, a_dict)
        out_tiltcom.write_comfile(out_dir, change_name = 'sub_tilt.com')
        imodscript('sub_tilt.com', out_dir)
            
            
            
            
            
            # a_dict = {'OutputFile': out_tom,
            #       'InputProjections': sub_ts
            #           }
            
            # if supersample:
            #     if 'SuperSampleFactor' not in out_tiltcom.dict:
            #         a_dict['SuperSampleFactor'] = 2
            #     if 'ExpandInputLines' not in out_tiltcom.dict:
            #         a_dict['ExpandInputLines'] = None
            # if fakesirt:
            #     if 'FakeSIRTiterations'  not in out_tiltcom.dict:
            #         a_dict['FakeSIRTiterations'] = fakesirt
                    
            # out_tiltcom.dict = update_comdict(out_tiltcom.dict, a_dict)
            # out_tiltcom.write_comfile(out_dir, change_name = 'sub_tilt.com')
            # imodscript('sub_tilt.com', out_dir)



