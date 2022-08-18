# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:48:06 2022

@author: vojta
"""
import os
import sys
import signal
import datetime
from copy import deepcopy
from itertools import zip_longest
from os.path import isfile, join, realpath, split, isdir
import glob
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from subprocess import check_output, Popen, PIPE
import mrcfile
from PEETPRMParser import PEETPRMFile 
from PEETModelParser import PEETmodel
from PEETMotiveList import PEETMotiveList
from mdoc_parser import Mdoc_parser
import warnings
from scipy import fftpack as fft
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from numpy.fft import fftfreq
from scipy.signal import butter, freqs, find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import zoom, shift, rotate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline#,interp1d
from scipy.signal import convolve2d
from skimage.feature import peak_local_max
#from transformations import euler_from_matrix

import MapParser_f32_new
from EMMap_noheadoverwrite import Map
from skimage.filters import threshold_yen, threshold_isodata
from scipy.ndimage.morphology import grey_dilation, grey_erosion
from scipy.ndimage import convolve
from IMOD_comfile import IMOD_comfile

def norm(x):
    """normalise to 0 mean and 1 standard deviation"""
    return (x - np.mean(x))/np.std(x)

def machines_from_imod_calib():
    """
    Get list of machines for paralelisation from IMOD_CALIB_DIR cpu.adoc.

    Returns
    -------
    machines : list of str
        list of machine names.
    numbers : list of int
        Number of available cores.

    """
    adoc = join(os.environ['IMOD_CALIB_DIR'], 'cpu.adoc')
    if isfile(adoc):
        machines = []
        numbers = []
        with open(adoc) as f:
            for line in f.readlines():
                if line.startswith('[Computer'):
                    machines.append(line.split()[-1].strip(']'))
                if line.startswith('number'):
                    numbers.append(int(line.split('=')[-1]))
    else:
        machines = []
        numbers = []
    return machines, numbers

def find_nearest(array, value):
    """Find the index of the closest value in an array.
    modified from stackoverflow 2566412
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def bin_model(input_model, output_model, binning, motl = False, out_motl = False):
    """
    Parameters
    ----------
    input_model : str
        Path to IMOD model.
    output_model : str
        Path to output file.
    binning : int or float
        Binning value. Values smaller than 0 scale up.
    motl : str, optional
        Motive list path. Used to transfer offsets.
    out_motl : str, optional
        Write motive list too.

    """
    m = np.array(PEETmodel(input_model).get_all_points())
    if motl:
        motl = PEETMotiveList(motl)
        offsets = motl.get_all_offsets()
        m += offsets
    m = m*1./binning
    mout = PEETmodel()
    for p in range(len(m)):
        mout.add_point(0, 0, m[p])
    mout.write_model(output_model)   
        
    if motl and out_motl:
        for p in range(len(offsets)):
            motl.set_offsets_by_list_index(p, [0,0,0])
        motl.write_PEET_motive_list(out_motl)
    

def get_mod_motl_and_tomo(prm, ite = 2):
    """
    Get absolute paths for model files and motive lists from prm.
    
    Parameters
    ----------
    prm : str
        Path to PEET parameter file.
    ite : int, optional
        Desired iteration output. Use 0 for initial files. Attempt to find 
        last iteration files if specified iteration does not exist.
        The default is 2.

    Returns
    -------
    motls : ndarray
        Motive list paths.
    modfiles : ndarray
        Model file paths.  
    tomos : ndarray
        Tomogram paths.

    """
    cwd = os.getcwd()
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
        last_motl =  glob.glob(tmotl)
        if len(last_motl) > 0:
            last_motl = last_motl[-1]
            ite = int(last_motl.split('_')[-1].strip('Iter.csv'))
            motls = prm.get_MOTLs_from_ite(ite)   
        else:
            if ite == 0:
                #if there are no csv files with ite 0 then that's a problem
                raise Exception('No MOTLS found %s'
                                % join(prmdir, prmname))
            else:
                #Weird recursion but ok...
                motls, modfiles, tomos = get_mod_motl_and_tomo(join(prmdir, prmname), 0)
                return motls, modfiles, tomos
     
    if not os.path.isabs(motls[0]):
        for x in range(len(motls)):
            motls[x] = realpath(motls[x])

    modfiles = prm.prm_dict['fnModParticle']
    if not os.path.isabs(modfiles[0]):
        for x in range(len(modfiles)):
            modfiles[x] = realpath(modfiles[x])
    tomos = prm.prm_dict['fnVolume']
    if not os.path.isabs(tomos[0]):
        for x in range(len(tomos)):
            tomos[x] = realpath(tomos[x])            
    os.chdir(cwd)
    return motls, modfiles, tomos

def write_mrc(out_name, data, voxel_size = 1.0, origin = (0,0,0), mode = False,
              set_float = True):
    """
    Write ndarray as MRC file.

    Parameters
    ----------
    out_name : str
        output file path.
    data : ndarray
        3D array.
    voxel_size : float
        Output voxel (isotropic) size.
    origin : three ints or floats
        Output origin. The signs are inverted to keep header values the same as input.
    mode : int, optional
        Set MRC file mode. E.g. 1 is 16 bit int, 2 is 32 bit float.

    """
    with mrcfile.new(out_name, overwrite=True) as mrc:
        if set_float:
            mrc.set_data(np.float32(data))
        else:
            mrc.set_data(data)
        mrc.voxel_size = voxel_size
        mrc.header.origin = tuple(-np.array(origin))
        if mode:
            mrc.header.mode = mode

def get_apix_and_size(path, origin = False):
    """
    Run IMOD header -p -s
    
    Parameters
    ----------
    path : str
        Path to MRC file.
    origin : bool
        Also return origin 

    Returns
    -------
    p : float
        Pixel size (Angstrom).
    s : ndarray
        1D array containing 3 int values: X, Y, Z axis size.
        
    Conditional return
    -------
    o : ndarray
        If origin == True, return header origin.

    """
    out = check_output('header -p -s -o %s' % path, shell = True).split()
    s = np.array([int(x) for x in out[:3]])
    p = float(out[3])
    o = np.array([float(x) for x in out[6:]])
    if origin:
        return p, s, o
    else:
        return p, s  

def fit_pts_to_plane(voxels):  
    """
    Fit a plane to given coordinates.
    from https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    
    Parameters
    ----------
    voxels : ndarray
        2D array (n by 3), X,Y,Z coordinates for each point.

    Returns
    -------
    fit : ndarray
        Plane fit parameters.
    residual : float
        Fit residual.

    """

    xy1 = (np.concatenate([voxels[:, :-1],
           np.ones((voxels.shape[0], 1))], axis=1))
    z = voxels[:, -1].reshape(-1, 1)
    fit = np.matmul(np.matmul(
            np.linalg.inv(np.matmul(xy1.T, xy1)), xy1.T), z)
    errors = z - np.matmul(xy1, fit)
    residual = np.linalg.norm(errors)
    return fit, residual

def make_lamella_mask(surface_model, tomo_size = False, out_mask = False, plot = False,
              rotx = True, tomo_path = False, filter_mask = True):
    """
    Generate a slab binary mask from coordinates arranged on two surfaces.

    Parameters
    ----------
    surface_model : str
        Path to IMOD model file. Can be e.g. tomopitch.mod (used for positioning)
    tomo_size : list of ints, optional
        X, Y, Z size of desired size of output mask. The default is False,
        but either tomo_size or tomo_path is needed.
    out_mask : str, optional
        path to write to. The default is False, in which case 3D ndarray
        is returned.
    plot_planes : bool, optional
        Plot results if True. The default is False.
    rotx : bool, optional
        True if tomogram has been rotated by 90 degrees around X axis.
        The default is False.
    tomo_path : str, optional
        Path to volume, the size and header is used. The default is False.
    filter_mask : bool, optional
        Low-pass filter the resulting mask to smooth edges.
    Returns
    -------
    mask  : ndarray
        3D mask. Returned only when output path is not specified.

    """
    
    def def_plane(fit, tomo_size):
        X, Y = np.meshgrid(
                np.arange(0, tomo_size[0]), np.arange(0, tomo_size[1]))
        Z = fit[0]*X + fit[1]*Y + fit[2]
        return X, Y, Z
    
    if isinstance(tomo_size, bool):
        if not tomo_path:
            raise Exception('tomo_size or tomo_path is required')
        else:
            apix, tomo_size, origin = get_apix_and_size(tomo_path, origin = True)
    else:
        apix = 1
        origin = (0,0,0)

    allpts = PEETmodel(surface_model).get_all_points()
    
    #sort Z points into 2 surfaces
    if not rotx:
        allpts = allpts[:, [0, 2, 1]]
        tomo_size = np.array(tomo_size)[[0, 2, 1]]
    
    fit_pts = allpts[:, 2].reshape(-1,1) #fit 1D
    km = KMeans(2).fit(fit_pts)
    tree = KDTree(km.cluster_centers_)
    dst, pos = tree.query(fit_pts)
    upper = allpts[pos == 0]
    lower = allpts[pos == 1]
    if np.median(upper[:,2:3]) < np.median(lower[:,2:3]):
        upper, lower = lower, upper
        
    fit, _ = fit_pts_to_plane(upper)
    X1, Y1, Z1 = def_plane(fit, tomo_size)
    fit, _ = fit_pts_to_plane(lower)
    X2, Y2, Z2 = def_plane(fit, tomo_size)
        
    if plot:
        plt.subplot(111, projection = '3d')
        ax = plt.subplot(111, projection = '3d')
        ax.scatter(upper[:,0], upper[:,1], upper[:,2], c = 'r')
        ax.scatter(lower[:,0], lower[:,1], lower[:,2], c = 'b')
        ax.plot_wireframe(X1, Y1, Z1, color = 'r')
        ax.plot_wireframe(X2, Y2, Z2, color = 'b')
        
    empty_mask = np.ones((tomo_size[2], tomo_size[1], tomo_size[0]))    
    mask = (empty_mask*np.arange(tomo_size[2])[:, None, None])
    mask = np.logical_and(mask < Z1[:,:], mask > Z2[:,:])

    if not rotx:
        mask = np.flip(mask, 0)
        mask = np.rot90(mask, k = -1, axes = (0,1))

    if out_mask:
        write_mrc(out_mask, mask, voxel_size = apix, origin = origin)
        if filter_mask:
            tmp_mask = out_mask + '~'
            os.rename(out_mask, tmp_mask)
            check_output('mtffilter -3 -l 0.001,0.1 %s %s' %
                         (tmp_mask, out_mask), shell = True)   
            os.remove(tmp_mask)
            os.rename(out_mask, tmp_mask)
            check_output('newstack -scale 0,1 %s %s' % (tmp_mask, out_mask), shell = True)
            os.remove(tmp_mask)
    else:
        return mask

def convert_ctffind_to_defocus(defocus_file, base_name, tlt, out_dir):
    """
    Converts to IMOD defocus 3.
    Tilts are listed in reverse order (see IMOD ctfphaseflip)

    Parameters
    ----------
    defocus_file : str
        Path to file.
    base_name : str
        Output file base name (IMOD base name).
    tlt : str
        Path to tilt angle file (.tlt).
    out_dir : str
        Path to output directory.

    Returns
    -------
    out_defocus_file : str
        Path to converted defocus file.

    """
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

def read_defocus_file(defocus_file, base_name = False, tlt = False,
                      out_dir = False):
    """
    Parsse defocus file.

    Parameters
    ----------
    defocus_file : str
        Path to .defocus file.
    base_name : str, optional
        Base name of output file. This entry is required only if the input 
        file is in ctffind format. The default is False.
    tlt : str, optional
        Path to tilt file. This entry is required only if the input 
        file is in ctffind format. The default is False.
    out_dir : str, optional
        Path to output directory. This entry is required only if the input 
        file is in ctffind format. The default is False.

    Returns
    -------
    df : ndarray
        [number of tilts, 7]: [tilt ordinal, tilt ordinal, tilt angle, 
                               tilt angle, defocus1 (nm), defocus2, astig angle]

    """
    #First check what type of defocus file it is.  
    #CTFFind is converted to .defocus 3.  
    with open(defocus_file, 'r') as ff:
        first_line = ff.readlines()[0]
    if any([a == 'CTFFind' for a in first_line.split()]):
        if not all((base_name, tlt, out_dir)):
            raise Exception('read_defocus_file: optional arguments required'
                            + ' to read CTFFind format.')
        defocus_file = convert_ctffind_to_defocus(defocus_file, base_name,
                                                  tlt, out_dir)

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

def get_pcl_defoci(ssorted_pcls, defocus_file, ali, 
                   base_name = False, out_dir = False,
                   apix = False, tlt = False
                   ):
    """
    Calculate particle defoci based on tilt angles and measured defocus.

    Parameters
    ----------
    ssorted_pcls : ndarray
        Output from fid2ndarray.
    defocus_file : str
        Path to IMOD defocus file. Ctffind is supported (in theory), in which case
        read_defocus_file optional arguments are required.
    ali : str
        Path to aligned stack (.ali). Used to get image size.
    base_name : str, optional
        Same as read_defocus_file
    out_dir : str, optional
        Same as read_defocus_file
    apix : float, optional
        Image pixel size. It is read from ali by default.
    tlt : str, optional
        Same as read_defocus_file

    Returns
    -------
    all_defoci : ndarray
        Defocus in angstroms. Array shape is [num tilts, num particles]

    """
    tilt_angles = ssorted_pcls[:, 0, 5]#assuming tilt angles were set in ssorted_pcles, I don't think it makes sense to deal with the alternative
    stack_apix, stack_size = get_apix_and_size(ali)
    if not apix:
        apix = stack_apix
    all_defoci = np.zeros((ssorted_pcls.shape[:2]))
    
    df = read_defocus_file(defocus_file, base_name, tlt, out_dir)
    defocus_excludelist = np.isin(df[:, 0], ssorted_pcls[:, 0, 2] + 1) #imod numbered from 1
    # defocus_excludelist = np.isin(np.round(df[:, 2], decimals = 0),
    #                               np.round(tilt_angles, decimals = 0))
    df = df[defocus_excludelist]
    
    meandf = np.mean((df[:, 4], df[:, 5]), axis = 0) * 10 # mean defocus in angstroms
    sin_tlt = np.sin(np.radians(np.array(tilt_angles))) #sine of tilt angles
    xpos = np.negative(ssorted_pcls[:, :, 0] - stack_size[0]//2) * apix
      
    for x in range(ssorted_pcls.shape[1]):
        #distance from tilt axis at 0 degrees in angstroms 
        dst = (sin_tlt * xpos[:, x])
        all_defoci[:, x] = meandf - dst

    return all_defoci

def fid2ndarray(fiducial_model, tlt = False, defocus_file = False, 
                #for defocus file:
                ali = False, excludelist = False,
                base_name = False, out_dir = False,
                **kwargs):
    """
    Reads an IMOD fiducial model into an ndarray. All contours must have
    the same number of particles.

    Parameters
    ----------
    fiducial_model : str
        Path to fiducial model.
    tlt : str, optional
        Path to tilt angle file (.tlt)
    defocus_file : str, optional
        Path to defocus file. This can be ctffind format, in which case the
        remaining keyword arugmnets are required.
    ali : str, optional
        Path to aligned stack (.ali)
    base_name : str, optional
        Base name of newly converted ctffind defocus file. For flexo, this
        should be the same as the IMOD base name.
    out_dir : str, optional
        Path to output directory, only used with ctffind input.

    Returns
    -------
    ssorted_pcls : ndarray
        Shape:
            [number of tilts:model points per tilt:7
            0=xcoords, 1=ycoords, 2=tilt number, 3=particle index, 4=group id (from 0), 5=tilt angle, 6=defocus)]
            group id == len(groups) indicates unallocated pcls    

    """

    #fiducial model to numpy array        
    rawmod = PEETmodel(fiducial_model).get_all_points()
    #rawmod[:, 2] = np.array(np.round(rawmod[:, 2], decimals = 0), dtype = int)
    tilt_numbers = np.unique(rawmod[:, 2])
    
    sorted_pcls = []
    for x in tilt_numbers:
        tmp = rawmod[rawmod[:, 2] == x]
        sorted_pcls.append(tmp)
        
    sorted_pcls = np.array(sorted_pcls)
    ssorted_pcls = np.dstack((sorted_pcls,
                              np.zeros((sorted_pcls.shape[0],
                                        sorted_pcls.shape[1], 4))))
    ssorted_pcls[:,:,3] =  list(range(sorted_pcls.shape[1]))
    
    
    excluded_views_mask = np.isin(np.arange(ssorted_pcls.shape[0]), np.array(excludelist) - 1, invert = True)
    # excluded_views_mask = np.ones(ssorted_pcls.shape[0], dtype = bool)
    # #remove excluded views. This could also be done using excludelist from tilt.com....but I prefer this??? maybe?
    # #I've seen excluded view coordinates being given either a really small coordinate value
    # #or insanely large (positive or negative) or "-0"  or "5.21" X coord and Z coord or...seemingly at random?
    # for x in range(ssorted_pcls.shape[0]):
    #     if np.allclose(0, np.mean(ssorted_pcls[x, :, :2], axis = 1), atol=1e-4):
    #         excluded_views_mask[x] = False
    #         print('excluded', x)
    #     elif np.absolute(ssorted_pcls[x, 0, 2]) > len(tilt_numbers) + 100:
    #         excluded_views_mask[x] = False
    #         print('excluded large', x)
    #     elif np.all(ssorted_pcls[x, 0, 0] == ssorted_pcls[x, 0, 2]):
    #         excluded_views_mask[x] = False
    #         print('excluded same', x)
            
    ssorted_pcls = ssorted_pcls[excluded_views_mask]
            
    if tlt:
        tilt_angles = np.array([float(x.strip()) for x in open(tlt, 'r')])
        tilt_angles = tilt_angles[excluded_views_mask]
        if len(tilt_angles) != ssorted_pcls.shape[0]:
            raise Exception('Number of tilts and number of tilt angles does not match.')
        ssorted_pcls[:, :, 5] = tilt_angles[:, None]
    if defocus_file:
        if not tlt:
            raise Exception('Tilt angle file (tlt) is required to set particle defoci.')
        defoci = get_pcl_defoci(ssorted_pcls, defocus_file, ali, **kwargs)
        #defoci = defoci[excluded_views_mask]
        ssorted_pcls[:, :, 6] = defoci

    return ssorted_pcls
    
def make_non_overlapping_pcl_models(fiducial_model, box_size, out_dir,
                                    threshold = 0.01, model3d = False,
                                    motl3d = False, **kwargs):
    """
    Split particles that would otherwise overal in a tilt series into
    independent models. All contours must have the same number of particles.

    Parameters
    ----------
    fiducial_model : str or ndarray
        Path to IMOD fiducial model (or reprojected 3D model) or 
        ndarray output from fid2ndarray. Ragged contours not allowed (contours
        of different lengths).
    box_size : int or list of two ints
        Isotropic box size. This controls the maximum allowed distance between
        particles.
    out_dir : str
        Path to output file.
    threshold : float, optional
        Fraction of particles too small to write a model for. E.g. with 
        threshold = 0.01, groups of non-overlapping particles smaller than 1%
        of total particles will be excluded.
    model3d : str, optional
        Path to 3D particle model. It will be split into groups.
    motl3d : str, optional (required with model3d)
        Path to PEET motive list.
    **kwargs
        Takes optional arguments for fid2ndarray

    Returns
    -------
    outmods : list of str
        Paths to separated models.
    groups : ndarray
        Array of masks, one for each group of particles.
    remainder : ndarray
        Indices of excluded particles (I think).
    ssorted_pcls : ndarray
        Array used to keep track of particles (e.g. original indices, group
                                               id etc.)
        Shape:
            [number of tilts:model points per tilt:
            xcoords, ycoords, tilt number, particle index, group id (from 0))]
            group id == len(groups) indicates unallocated pcls                                                
    
    Conditional return
    --------
    split3d_models : list of str
        Paths to separated 3d models.
    split3d_motls : list of str
        Paths to separated motive lists.
    """

    if isinstance(box_size, int):
        box_size = [box_size, box_size]
    #force square box
    if box_size[0] != box_size[1]:
        box_size = [max(box_size), max(box_size)] 
    
    if isinstance(fiducial_model, str):
        ssorted_pcls = fid2ndarray(fiducial_model, **kwargs)
    elif isinstance(fiducial_model, np.ndarray):
        ssorted_pcls = fiducial_model
    init_indices = np.arange(ssorted_pcls.shape[1])
    indices = np.arange(ssorted_pcls.shape[1])

    #for each tilt, for each particle, generate a list of particles 
    #(and distances) that are within {box_size}
    pvp = np.zeros((ssorted_pcls.shape[0],
                    ssorted_pcls.shape[1],
                    ssorted_pcls.shape[1], 2))
    for x in range(ssorted_pcls.shape[0]): 
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
    mm = np.zeros((ssorted_pcls.shape[1], ssorted_pcls.shape[1]))    
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
        #len(indices) + 1 is is used to filter out particle indices 
        #that have been allocated to a group
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
    toosmall = groups.sum(axis = 1)
    groups = groups[toosmall != 0] #I'm not sure why this required...
    toosmall = toosmall[toosmall != 0]/float(mm.shape[0]) < threshold
    if np.any(toosmall):
        print(('%s group(s) had less than %s %% the total number of '
               +'particles and were removed.') % (sum(toosmall), threshold*100))
        remainder = init_indices[np.sum(groups[toosmall], axis = 0, dtype = bool)]

        #set unallocated particle group id to len(groups) + 1
        ssorted_pcls[:, remainder, 4] = len(groups) + 1
        groups = groups[np.logical_not(toosmall)]
        #combined filtered masks and apply to list of indices
        #--> unused indices
    else:
        remainder = []
    print('Number of non-overlapping models ' + str(len(groups)))
    remainder = [int(r) for r in remainder] #json doesn't like numpy types
    outmods = []
    for x in range(len(groups)):
        #add group identifier to ssorted_pcls, numbered from zero
        ssorted_pcls[:, groups[x], 4] = x  
        #write peet models for each group
        out_name = join(out_dir, 'model_group_%02d.fid' % (x))
        outmods.append(out_name)
        outmod = PEETmodel()
        g = np.swapaxes(ssorted_pcls[:, groups[x], :3], 1, 0)
        for p in range(len(g)):
            if p != 0:
                    outmod.add_contour(0)
            for r in range(g.shape[1]):
                outmod.add_point(0, p, g[p,r])
        outmod.write_model(out_name)   
    #remaining particles set to len(groups) + 1
    ssorted_pcls[0, np.logical_not(np.any(groups, axis = 0)), 4] = len(groups) + 1 
    
    if model3d:
        split3d_models = []
        split3d_motls = []
        model3d = PEETmodel(model3d).get_all_points()
        motl3d = PEETMotiveList(motl3d)
        for x in range(len(groups)):
            #add group identifier to ssorted_pcls, numbered from zero
            ssorted_pcls[:, groups[x], 4] = x  
            #write peet models for each group
            out_name = join(out_dir, 'model_group_%02d.mod' % (x))
            out_motl_name = join(out_dir, 'motl_group_%02d.csv' % (x))
            split3d_models.append(out_name)
            split3d_motls.append(out_motl_name)
            outmod = PEETmodel()
            outmotl = PEETMotiveList()
            g = model3d[groups[x]]
            motl_ind = np.array(ssorted_pcls[0, groups[x], 3], dtype = int)
            for p in range(len(g)):
                if p != 0:
                        outmod.add_contour(0)
                outmod.add_point(0, p, g[p])
            outmod.write_model(out_name)   
            for ind in motl_ind:
                outmotl.add_pcle(motl3d[ind])
            outmotl.renumber()
            outmotl.write_PEET_motive_list(out_motl_name)
    
        return outmods, groups, remainder, ssorted_pcls, split3d_models, split3d_motls
    else:
        return outmods, groups, remainder, ssorted_pcls


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

def get_tiltrange(tiltlog):
    """
    Extract tilt range from tilt.com

    Parameters
    ----------
    tiltlog : str
        Path tot tilt.com.

    Returns
    -------
    trange : list of 2 floats
        Min and max tilt angles (or the inverse).

    """
    with open(tiltlog) as f:
        aa = f.readlines()
    aline = False
    alist = []
    for x in range(len(aa)):    
        if aa[x] == ' Projection angles:\n':
            aline = x + 1
            if not aa[x + 1].strip():
                aline += 1
        if not isinstance(aline, bool):
            if x >= aline:
                if aa[x] == '\n' and aa[x + 1] == '\n':
                    break
                else:
                    alist.append(aa[x])
    trange = [float(alist[0].split()[0]), float(alist[-1].split()[-1].strip())]
    return trange

def prepare_prm(prm, ite, tomo, tom_n, out_dir, base_name, new_prm_dir,
                        search_rad = False,
                        phimax_step = False,
                        psimax_step = False,
                        thetamax_step = False,
                        hicutoff = False, #expect list/tuple
                        lowcutoff = False, #expect list/tuple
                        refthr = False,
                        #if init, using tomograms in prm file
                        mod = False,
                        motl = False,
                        reuse_tiltrange = False
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
        tomo [str, list, bool or 'init'] path to tomo(s)
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
        mod = [list of str] List of mode file paths. Overwrites prm model files.
        motl = [list of str] List of motive list paths. Overwrites prm motl.
        reuse_tiltrange = [bool] Use tilt range from prm file rather than 
            extracting it from tilt.com
        
    Returns:
        str, new prm path
        float, apix
    """
    prmdir, prmname = os.path.split(prm)
    prm = PEETPRMFile(prm)
    
    motls, modfiles, init_tomos = get_mod_motl_and_tomo(join(prmdir, prmname), ite)
    cwd = os.getcwd()
    os.chdir(prmdir)
            
    reference = prm.prm_dict['reference']
    reference = check_refpath(reference)
    mask = prm.prm_dict['maskType']
    if not np.isin(mask, ['none', 'sphere', 'cylinder']):
        mask = check_refpath(mask)

    # #determine binning of tomo used for peet run, used for volume naming
    r_apix, r_size = get_apix_and_size(reference)
    # st_apix, st_size = get_apix_and_size(st)
    # #at what binning was the peet project actually run?
    # peet_bin = int(np.round(r_apix/st_apix, decimals = 0))

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
        elif isinstance(tom_n, np.ndarray):
            if np.sort(tom_n)[0] == 0:
                print(('prepare_prm: Zero found for tomogram number. '
                       + 'Assuming entered numbering is from zero, not 1.'))
                tom_n = np.array(tom_n)
            else:                
                tom_n = np.array(tom_n) - 1

    if not mod:
        mod = [modfiles[n] for n in tom_n]
    if not motl:
        csv = [motls[n] for n in tom_n]
    else:
        csv = motl
        
    #tilt range - stored as list of lists
    #assuming Flexo tomogram is being used 
    if reuse_tiltrange:
        trange = prm.prm_dict['tiltRange']
        trange = [trange[n] for n in tom_n]
    else:
        trange = get_tiltrange(join(out_dir, 'tilt.log'))
        trange = [trange for x in tom_n]
    
    # if isinstance(tomo, bool):
    #     #The point of this is to run peet on flexo tomograms, so the default:
    #     tomo = [join(out_dir, base_name + '_bin%s.rec' % int(peet_bin))
    #             for m in range(len(mod))]
    if isinstance(tomo, str):
        if tomo == 'init':
            tomo = [init_tomos[tom_n[t]] for t in range(len(tom_n))]
        else:
            tomo = [tomo]
    elif (isinstance(tomo, list) or isinstance(tomo, tuple)
            or isinstance(tomo, (np.ndarray, np.generic))):
        tomo = list(tomo)
        if len(tomo) != len(tom_n):       
            raise ValueError('prepare_prm: Number of tomogram paths does ' +
                             'not match the number of requested tomograms.')
            
    if len(tomo) != len(mod):
        raise Exception("Numbers of tomogram and models don't match %s %s" % (len(tomo), len(mod)))
            
    #check reference binning, bin reference/model if it's not the same as tomo
    tomo_apix, tomo_size = get_apix_and_size(tomo[0])
    rel_bin = np.round(tomo_apix/r_apix, decimals = 2)
    if rel_bin != 1.:
        print('Reference and tomogram voxel sizes are not the same (%s, %s). Adjusting reference and model files to match...' %
              (r_apix, tomo_apix))
        new_reference = join(out_dir, 'bin_' + split(reference)[-1])
        check_output('squeezevol -f %s %s %s' % (rel_bin, reference, new_reference), shell = True)
        new_apix, new_size = get_apix_and_size(new_reference)
        if np.any(new_size%2):
            new_size -= new_size%2
            check_output('trimvol -nx %s -ny %s -nz %s %s %s' %
                         (new_size[0], new_size[1], new_size[2], new_reference, new_reference), shell = True)
                        
        if rel_bin < 1:
            hicutoff = [0.4*rel_bin, np.round(max(1./r_size[0], 0.03), decimals = 3)]
        reference = new_reference
        
        
        for mn in range(len(mod)):
            mod_str = ('.').join(split(mod[mn])[1].split('.')[:-1])
            output_model = join(new_prm_dir, mod_str + '_bin%s.mod' % rel_bin)
            output_motl = join(new_prm_dir, mod_str + '_bin%s.csv' % rel_bin)            
            bin_model(mod[mn], output_model, rel_bin, motl = csv[mn],
                      out_motl = output_motl)
            mod[mn] = output_model
            csv[mn] = output_motl
        
    new_prm = prm.deepcopy()
    new_prm.prm_dict['fnModParticle'] = mod
    new_prm.prm_dict['initMOTL'] = csv
    new_prm.prm_dict['fnVolume'] = tomo
    new_prm.prm_dict['tiltRange'] = trange
    new_prm.prm_dict['fnOutput'] = base_name
    new_prm.prm_dict['reference'] = reference
    new_prm.prm_dict['maskType'] = mask
    
    #get rid of spherical sampling if it's in the prm
    new_prm.prm_dict['sampleSphere'] = 'none'
    new_prm.prm_dict['sampleInterval'] = 'NaN'
    
    
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
    
    if isinstance(search_rad, (bool, type(None))):
        #default False to make it easier to work with defaults in higher order
        #functions
        #search_rad = [0, 0, 0]
        #it makes no sense for the default to be 0. If someone forgets to
        #set the search rad (like me) then the FSC output will be wrong
        search_rad = [r_size[0]/4]
    elif isinstance(search_rad, int):
        search_rad = [search_rad]
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
    return new_prmpath, r_apix    

def get_fsc(vol1, vol2, out_dir, step=0.01):
    """
    Basic FSC.

    Parameters
    ----------
    vol1 : str
        Path to halfmap1.
    vol2 : str
        Path to halfmap1.
    out_dir : str
        Output directory.
    step : float, optional
        Step size. The default is 0.001.

    """
    if isinstance(vol1, str):
        vol1 = mrcfile.open(vol1).data
    if isinstance(vol2, str):
        vol2 = mrcfile.open(vol2).data
    
    FTvol1 = fft.fftshift(fft.fftn(vol1))
    FTvol2 = fft.fftshift(fft.fftn(vol2))
    
    cc = np.real(FTvol1*FTvol2.conj())/(np.abs(FTvol1)*np.abs(FTvol2))
    x = fft.fftshift(fft.fftfreq(vol1.shape[2], 1))
    y = fft.fftshift(fft.fftfreq(vol1.shape[1], 1))
    z = fft.fftshift(fft.fftfreq(vol1.shape[0], 1))
    xx, yy, zz = np.meshgrid(z, y, x, sparse = True, indexing = 'ij')
    R = np.sqrt(xx**2+yy**2+zz**2)

    # calculate the mean
    f = lambda r : cc[(R >= r-step/2) & (R < r+step/2)].mean()
    r  = np.linspace(0,x[-1], num=round((1/(2))//(step*2)))
    mean = np.vectorize(f)(r) #I want this in the same format as calcUnbiasedFSC
    
    m = np.logical_and(np.logical_not(np.isnan(mean)), r > 0)
    r = r[m]
    mean = mean[m]
    
    with open(join(out_dir, 'arrFSCC.txt'), 'w') as g:
        g.write(('\n').join(list(np.array(mean, dtype = str))))
        
    with open(join(out_dir, 'freqShells.txt'), 'w') as g:
        g.write(('\n').join(list(np.array(r, dtype = str))))
    
   # return mean, r, R, FTvol1, FTvol2, cc

def peet_halfmaps(peet_dir, prm1, prm2, machines, use_davens_fsc = True):
    """
    Run PEET with half datasets, 

    Parameters
    ----------
    peet_dir : str
        Path to peet directory, i.e. where halfmap directories will be.
    prm1 : str
        Path to half dataset prm 1.
    prm2 : str
        Path to half dataset prm 1.
    machines : list
        Machines for parallelisation, see run_split_peet.

    """
    
    #fsc dirs - list, directories containing freShells or w/e
    
    cwd = os.getcwd()
    fsc1d, prm_base1 = split(prm1)
    fsc2d, prm_base2 = split(prm2)
    prm_base1 = prm_base1[:-4]
    prm_base2 = prm_base2[:-4]

    os.chdir(fsc1d)
    run_generic_process(['prmParser', prm_base1 + '.prm'], 
                        join(fsc1d, 'parser.log'))
    #check_output('prmParser %s' % prm_base1, shell = True)
    if not isfile(prm_base1 + '-001.com'): 
        raise Exception('prmParser failed to generate com files. %s' % prm1)
    os.chdir(fsc2d)
    run_generic_process(['prmParser', prm_base2 + '.prm'], 
                        join(fsc2d, 'parser.log'))  
    if not isfile(prm_base2 + '-001.com'):
        raise Exception('prmParser failed to generate com files. %s' % prm2)
    
    run_split_peet(prm_base1, fsc1d, prm_base2, fsc2d, machines)
    #calcUnbiasedFSC
    os.chdir(peet_dir)
    
    
    if use_davens_fsc:
        vol1 = glob.glob(fsc1d + '/unMasked*.mrc')
        vol1 = sorted(vol1, key=lambda x: int(x.split('_')[-1].split('.')[0].strip('Ref')))[-1]
        vol2 = glob.glob(fsc2d + '/unMasked*.mrc')
        vol2 = sorted(vol2, key=lambda x: int(x.split('_')[-1].split('.')[0].strip('Ref')))[-1]        
        get_fsc(vol1, vol2, peet_dir)
    else:
        fsc_log = join(peet_dir, 'calcUnbiasedFSC.log')
        print('Running calcUnbiasedFSC...')
        run_generic_process(['calcUnbiasedFSC', prm1, prm2], out_log = fsc_log)

    os.chdir(cwd)

def optimal_lowpass(y):
    """
    Return 1/spatial frequency for a specified electron dose. 
    """
    y = max(y, 7.03) #truncate negative values
    a = ((0.4*y - 2.81)/0.245)**(1./ - 1.665) #from grant and grigorieff
    a = round(1./a, 2)
    return a

def optimal_lowpass_list(order = False,
              mdoc = False, flux = False, tilt_dose = False,
              pre_exposure = 0, tlt = False, orderfile_type = None):
    """
    Return optimal resolution to dosefilter a tilt series. Requires 
    1) SerialEM mdoc file 
    or
    2) order in which tilts were acquired and the flux (dose) per tilt (tilt_dose)
        in electrons/square Angstrom.
    or
    3) Order/dose file. Currently supported orderfile_type is 'old_relion' base
             on Empiar 10045 (requires tlt). 
             
    In cases where the dose was not calibrated (ExposureDose = 0) in the .mdoc,
    the tilt_dose or flux (electrons/Angstrom/sec) can be specified to the same
    effect.

    Parameters
    ----------
    order : list, ndarray, str, optional
        Stack ordinals sorted by increasing time of data collection. I.e. if 
        collection was started at 0 degrees, then tilted to -3 ... -60, the 
        order would be [21, 20, ..., 0, ..., 41].
        Can be order file, in which case the orderfile_type needs to be specified.
        The default is False.
    mdoc : str, optional
        Path to .mdoc file. The default is False.
    flux : int or float, optional
        flux (electrons/Angstrom/sec). The default is False.
    tilt_dose : int, float, list or ndarray, optional
        flux per tilt (electrons/Angstrom). Can be a single value or a list/ndarray
        of the same length as order. The default is False.
    pre_exposure : int, float, optional
        Pre-exposure. The default is 0.
    tlt : str, optional
        Path to tilt angles file. It's possible to use a "stock" mdoc file for
        multiple tilt series, some of which don't have fewer tilts. This shortens
        the lowfreq list to the same length as the number of tilt angles.'
    orderfile_type : str, optional
        Type of order file. Currently supported: 'old_relion' 
        (based on Empiar 10045)

    Returns
    -------
    lowfreqs : list
        list of 1/freqs.

    """
    
    def read_orderfile(file, orderfile_type):
        if orderfile_type == 'old_relion':
            if not tlt:
                raise Exception('Tilt angle file required with order file type "%s".' % orderfile_type )
            tmp = []
            with open(file) as f:
                for line in f.readlines():
                    tmpl = []
                    for sl in line.split():
                        try:
                            tmpl.append(int(sl))
                        except:
                            try:
                                tmpl.append(float(sl))
                            except:
                                raise
                    tmp.append(tmpl)
                    
            tmp = np.array(tmp)
            tilt_angles = [float(x.strip()) for x in open(tlt, 'r')] # these are refined angles. Just check if it starts with 60 or -60
            if tilt_angles[0] > tilt_angles[-1]:
                order = np.argsort(tmp[:, 0][::-1])
            else:
                order = np.argsort(tmp[:, 0])
            flux = np.hstack((0, np.diff(tmp[:, 1])))
            return order, flux
        else:
            raise Exception('Tilt order file "%s" not supported.' % orderfile_type)
    
    if mdoc:
        m = Mdoc_parser(mdoc)
        order = m.order
        dose_per_tilt = m.dose_per_tilt
        if np.all(m.dose_per_tilt == 0.):
            if flux:
                exp_times = m.exp_times
                dose_per_tilt = m.exp_times*exp_times
            elif not isinstance(tilt_dose, bool):
                if isinstance(tilt_dose, (int, float)):
                    dose_per_tilt = np.zeros(len(order)) + tilt_dose
                elif isinstance(tilt_dose, (list, np.ndarray)):
                    dose_per_tilt = np.array(dose_per_tilt)
            else:
                warnings.warn('Exposure dose in .mdoc is 0. No flux or tilt_dose was specified. Only global pre_exposure can be applied.')
                return np.zeros(len(order)) + pre_exposure
            
    elif isinstance(order, str):
        if not orderfile_type:
            raise Exception('orderfile_type required.')
        order, dose_per_tilt = read_orderfile(order, orderfile_type)
        if isinstance(dose_per_tilt, bool):
            if isinstance(tilt_dose, (int, float)):
                dose_per_tilt = np.zeros(len(order)) + tilt_dose
            elif isinstance(tilt_dose, (list, np.ndarray)):
                dose_per_tilt = np.array(dose_per_tilt)
            else:
                warnings.warn('No electron exposure information found.')
                             
    elif not isinstance(tilt_dose, (bool, type(None))) and not isinstance(order, (bool, type(None))):
        if isinstance(tilt_dose, (int, float)):
            dose_per_tilt = np.zeros(len(order)) + tilt_dose
        elif isinstance(tilt_dose, (list, np.ndarray)):
            dose_per_tilt = np.array(dose_per_tilt)         
    else:
        raise Exception('mdoc or tilt_dose and order is required.')

    oo = np.argsort(order)
    if tlt:
        #see docstring
        tilt_angles = [float(x.strip()) for x in open(tlt, 'r')]
        order_mask = np.isin(oo, np.arange(len(tilt_angles)))
        oo = oo[order_mask]
    cum_dose = np.cumsum(dose_per_tilt) + pre_exposure
    acc_dose = cum_dose[oo]
    lowfreqs = [optimal_lowpass(x) for x in acc_dose]
    return lowfreqs

def butterworth_filter(cutoff, box_size, apix, order = 4,
                       t = 'low', analog=True):
    """Author: Daven
    Generates butter curve. Uses scipy.signal.butter"""
    if type(cutoff) ==list:
        cutoff = [1./x for x in cutoff]
    else:
        cutoff = 1./cutoff  
    b, a = butter(order, cutoff, t, analog = analog)
    d = dist_from_centre_map(box_size, apix)
    freq = np.unique(d)
    w, h = freqs(b, a, freq)
    f = interp1d(w, abs(h))
    return f(d)

def dist_from_centre_map(box_size, apix): 
    #author: Daven Vasishtan
    #while it may seem possible to do this with sqrt of meshgrid**2, the
    #latter produces an asymmetric array 
    #the the difference in speed is negligible
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


def ctf(wl, ampC, Cs, defocus, ps, f):
    """ returns ctf curve
    defocus in angstroms
    Cs in mm"""
    a = (-np.sqrt(1 - ampC**2)*np.sin(2*np.pi/wl*(defocus*(f*wl)**2/
        2-Cs*(f*wl)**4/4) + ps) - ampC*np.cos(2*np.pi/wl*
        (defocus*(f*wl)**2/2 - Cs*(f*wl)**4/4) + ps))
    return a

def ctf_convolve_andor_butter(inp, apix, V = 300000.0, Cs = 27000000.0,
                              ampC = 0.07, ps = 0,
                              lowpass = 0, defocus = 0, butter_order = 4,
                              no_ctf_convolution = False,
                              padding = 10, phaseflip = True):

    """
    Convolve with CTF, phase flip and bandpass at the same time
    (saves ffting back and forth)
    Intended for a single image
    Input:
        inp [np array [xy]]
        apix [float] [angstrom]
        V [int] [volt]
        Cs [float] [angstrom]
        ampC [float] fraction of amplitude contrast
        defocus [float] [angstrom]
        lowpass [float] [spatial freq]
        
    """
    
    wl = (12.2639/(V+((0.00000097845)*(V**2)))**0.5)
    
    if padding:
        inp = np.pad(inp, ((padding, padding), (padding, padding)), 'linear_ramp',
                    end_values=(np.mean(inp), np.mean(inp)))
    (Nx, Ny) = inp.shape
    CTF = 1
    if defocus and not no_ctf_convolution:
        # Dx = np.float64(1)/np.float64(Nx*apix)
        # Dy = np.float64(1)/np.float64(Ny*apix)
        # x = np.arange(-Dx * Nx/2, Dx * Nx/2, Dx, dtype = 'float32') 
        # y = np.arange(-Dy * Ny/2, Dy * Ny/2, Dy, dtype = 'float32')
        x = fft.fftshift(fft.fftfreq(Nx, apix))
        y = fft.fftshift(fft.fftfreq(Ny, apix))
        xx, yy = np.meshgrid(x, y, sparse = True, indexing = 'ij')
        CTF = ctf(wl, ampC, Cs, defocus, ps, np.sqrt(xx**2 + yy**2))    
    window = 1
    if lowpass:
        window = butterworth_filter(lowpass, (Nx, Ny), apix, order = 4)
    FFTimg = fft.fftshift(fft.fft2(inp))
    if phaseflip:
        filtered = fft.ifft2(fft.ifftshift(((FFTimg * CTF)*np.sign(CTF)) * window)).real
    else:
        filtered = fft.ifft2(fft.ifftshift(((FFTimg * CTF)) * window)).real
    if defocus and not no_ctf_convolution and not phaseflip:
        filtered *= -1
    if padding:
        filtered = filtered[padding:-padding, padding:-padding]
    return filtered


def ctf_convolve_andor_dosefilter_wrapper(ali, apix, V = 300000,
        Cs = 27000000, ampC = 0.07, ps = 0, defocus = 0, butter_order = 4, 
        no_ctf_convolution = False, padding = 5,
        order = False, mdoc = False, flux = False, tilt_dose = False,
        pre_exposure = False, lowfreqs = False, phaseflip = False):
    """
    Can convolute with CTF and dosefilter (butter) at the same step
    OR
        just convolute with CTF
    OR
        just dosefilter

    Parameters
    ----------
    ali : ndarray
        Image or array of images (shape (X, Y) or (n, X, Y)).
    apix : float
        Pixel/voxel size in Angstroms.
    V : vloat, optional
        Acceleration voltage. The default is 300000.
    Cs : float, optional
        Spherical aberration (mm). The default is 27000000.
    ampC : float, optional
        Proportion of amplitude contrast. The default is 0.07.
    ps : float, optional
        Phase shift in radians? question mark? This is not tested at all. The default is 0.
    defocus : ndarray, optional
        Defoci in Angstroms. The default is 0.
    butter_order : int, optional
        Order of Butterworth filter. The default is 4. (Because I think it looks nice)
    no_ctf_convolution : bool, optional
        Disable ctf convoltuion. The default is False.
    padding : int, optional
        Padding (number of pixels) to be added before filtering. The default is 5.
    order : list or ndarray, optional
        Order in which tilts were collected. The default is False.
    mdoc : str, optional
        Path to .mdoc file. The default is False.
    flux : int or float, optional
        flux (electrons/Angstrom/sec). The default is False.
    tilt_dose : int, float, list or ndarray, optional
        flux per tilt (electrons/Angstrom). Can be a single value or a list/ndarray
        of the same length as order. The default is False.
    pre_exposure : int, float, optional
        Pre-exposure. The default is 0.  
    lowfreqs : ndarray, optional
        List of 1/resolution values for filtering, one for each image. This will
        bypass determining lowfreqs from mdoc/tilt_dose etc.
    phaseflip : bool
    
    Returns
    -------
    ali : ndarray
        Filtered image array.

    """

    ali = deepcopy(ali)
    
    if not np.any(defocus):
        defocus = [0 for x in range(len(ali))]
        
    if isinstance(lowfreqs, (bool, type(None))):
        if mdoc or (order and tilt_dose):
            #generate list of lowpass 1/frequencies for filtering  
            lowfreqs = optimal_lowpass_list(order = order,
                  mdoc = mdoc, flux = flux, tilt_dose = tilt_dose,
                  pre_exposure = pre_exposure)
        else:
        #skip dosefilter
            lowfreqs = np.zeros(len(ali))
    lowfreqs = np.array(lowfreqs)*2 #this should make the filter be close to zero at the critical 1/frequency
    for x in range(len(ali)):
        ali[x] = ctf_convolve_andor_butter(ali[x], apix, V = V, Cs = Cs,
                       ampC = ampC, ps = ps, lowpass = lowfreqs[x],
                       defocus = defocus[x], butter_order = butter_order,
                       no_ctf_convolution = no_ctf_convolution,
                       padding = padding, phaseflip = phaseflip)   

    return ali

def pad_pcl(pcl, box_size, sorted_pcl, stack_size):
    """
    Pads 2D particle coordinates that are outside the 
    stack coordinates with mean (of the particle).
    """
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

def extract_2d(stack, sorted_pcls, box_size,
                          offsets = False, normalise_first = False):
    """
    Extract boxes from a 2D stack
    Input:
        stack [str or ndarray] path of image stack or mrcfile array
        sorted_pcls [ndarray] see fid2ndarray
        box_size [list ints] 

    Returns extracted particles  [particle, tilt number, [X,Y]]
        """
    def get_2d_offset(pcl):
        return pcl - round(pcl)
    
    if isinstance(stack, str):
        mrc = deepcopy(mrcfile.open(stack).data)
    else:
        mrc = stack

    #this is confusing, but basically stack_size[2] should be mrc.shape[0]
    if normalise_first:
        mrc = np.array(mrc, dtype = float)
        for t in range(len(mrc)):
            mrc[t] = norm(mrc[t])
    stack_size = np.flip(mrc.shape)
    
    if np.squeeze(sorted_pcls).ndim == 2:
        sorted_pcls = np.squeeze(sorted_pcls)
        #i.e. in case there is only one particle
        #There could be only one tilt but it's not intended for that use...
        all_pcls = np.zeros((1, stack_size[2], box_size[1], box_size[0]))  
        all_offsets = np.zeros((1, stack_size[2], 2))

        #expand sorted_pcls to (n, 1, n) array
        s = np.zeros((sorted_pcls.shape[0], 1, sorted_pcls.shape[1]))
        s[:, 0, :] = sorted_pcls
        sorted_pcls = s
    else:
        all_pcls = np.zeros((
                sorted_pcls.shape[1], stack_size[2], box_size[0], box_size[1]))
        all_offsets = np.zeros((sorted_pcls.shape[1], stack_size[2], 2))

    if stack_size[2] != sorted_pcls.shape[0]:
        raise Exception ('Mismatch between particle Z coordinates (tilt numbers) and length of image file.'
                          + '\nThis is probably a result of an error in dealing with IMOD excludelist.'
                          + '\nLength of Z coordinates: %s\nLength of image file: %s' % (
                              sorted_pcls.shape[0], stack_size[2]))        

    for x in range(sorted_pcls.shape[1]):
        for y in range(sorted_pcls.shape[0]):
            x1 = int(round(max(0, sorted_pcls[y, x, 0] - box_size[0]//2)))
            x2 = int(round(min(
                        stack_size[0], sorted_pcls[y, x, 0] + box_size[0]//2)))        
            y1 = int(round(max(0, sorted_pcls[y, x, 1] - box_size[1]//2)))
            y2 = int(round(min(
                        stack_size[1], sorted_pcls[y, x, 1] + box_size[1]//2)))           
            if (np.array([x1,y1,x2,y2]) < 0).any():
                raise ValueError('View %s of particle %s is completely\
                                 outside the image stack.' % (y, x))
                                 
            #particles here are extracted by range(len(image_stack)) and not by
            #tilt index (ssorted_pcls[:, :, 2]). This is because reprojected 
            #data do not have excludelist views. 
            pcl = mrc[y, y1:y2, x1:x2]
            pcl = pad_pcl(pcl, box_size, sorted_pcls[y,x], stack_size)
            all_pcls[x, y] = pcl  

            if offsets:
                dx = get_2d_offset(sorted_pcls[y, x, 0])
                dy = get_2d_offset(sorted_pcls[y, x, 1])  
                # offsets are YX to match format of .xf
                all_offsets[x, y] = np.array([dy, dx])
    if sorted_pcls.shape[1] == 1:
        all_pcls = all_pcls.squeeze()
        
    #stupidly the shape (npcls, ntilts, x, y), compared to the (ntilts, npcls...)
    #but I only extract one pcl at a time so it should work
    if offsets:
        # offsets are YX to match format of .xf
        return all_pcls, all_offsets
    else:
        return all_pcls#, all_resid
    
def ncc(target, probe, max_dist, interp = 1, outfile = False,
        subpixel = 'full_spline', ccnorm = 'none'):
    """
    Cross correlation map between two images. Same size.

    Parameters
    ----------
    target : ndarray
        reference image.
    probe : ndarray
        query.
    max_dist : int
        Search range, map is first cropped to 2x max_dist.
    interp : int, optional
        Interpolation for sub-pixel accuracy. The default is 1.
    outfile : str, optional
        Output pat file. The default is False.
    subpixel : str, optional
        Interpolation methods:
            'zoom': uses ndimage.interpolation.zoom to upscale the CC map
            'cropped_spline' first crops the CC map (to 2*max_dist*interp)
                            then converts to spline
            'full_spline' converts to spline then accesses the central part
                            equivalent to 2*max_dist*interp
            The default is 'full_spline'.
    ccnorm : str, optional
        Normalisation method. 
            'none' - normalise by size
            'phase_corr' - normalised phase correlation
            'by_autocorr' - normalise by autocorrelations
            The default is 'none'.

    Returns
    -------
    ndarray
        size is max_dist*2*interp

    https://dsp.stackexchange.com/questions/31919/phase-correlation-vs-normalized-cross-correlation
    """
    
    
    if np.std(target) == 0 or np.std(probe) == 0:
        raise ValueError('ncc: Cannot normalise blank images')    
    # if max_dist > min(target.shape)//2 - 1:
    #     max_dist = min(target.shape)//2 - 1
#    if interp > 1:
#        target, probe = zoom(target, interp), zoom(probe, interp)
    #norm  
    target = (target - np.mean(target))/(np.std(target))
    probe = (probe - np.mean(probe))/(np.std(probe))
        
    if ccnorm == 'none':
        cc_fft = fftn(target) * np.conj(fftn(probe))
        ncc_fft = cc_fft/cc_fft.size
    elif ccnorm == 'by_autocorr':
        ft_target = fftn(target)
        conj_target = np.conj(ft_target)
        ft_probe = fftn(probe)
        conj_probe = np.conj(ft_probe)
        auto_t = np.absolute(ft_target*conj_target)
        auto_p = np.absolute(ft_probe*conj_probe)
        denom = np.sqrt(np.max(auto_t))*np.sqrt(np.max(auto_p))
        cc_fft = ft_target * conj_probe
        ncc_fft = cc_fft/denom
    elif ccnorm == 'phase_corr':
        cc_fft = fftn(target) * np.conj(fftn(probe))
        fft_abs = np.absolute(cc_fft)
        fft_abs[fft_abs == 0] = 1 #avoid divison by zero
        ncc_fft = cc_fft/fft_abs
    elif ccnorm == 'ncc':
        mask = np.array(target.shape) - max_dist
        targsq = target**2
        targ_sum_under_mask = convolve2d(target, np.ones(mask), mode='same', boundary='wrap')
        targsq_sum_under_mask = convolve2d(targsq, np.ones(mask), mode='same', boundary='wrap')
        ftarg = fft.fft2(target)
        fprobe = fft.fft2(probe).conj()
        cc = fft.fftshift(np.real(fft.ifft2(ftarg*fprobe)))
        cc_mean = targ_sum_under_mask/(mask[0]*mask[1])
        cc_std = (targsq_sum_under_mask-cc_mean**2)**0.5
        new_cc = (cc-cc_mean)/cc_std
        new_cc[cc_std==0] = new_cc[cc_std!=0].min()
        ncc = new_cc
        
    if ccnorm != 'ncc':
        ncc = ifftshift(ifftn(ncc_fft).real)
    
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

def get_peaks(cc_map, n_peaks = 100, return_blank = False):
    """Finds peaks in 2D array.
    Input arguments:
        cc_map [2d array]
        max_peaks [int] maximum number of peaks to be stored. also defines
            size of output array
        return_blank [bool] return empty array of the expected shape (
            this is intended to deal with blank query)
    Returns:
        out_peaks [n_peaks by 3 numpy array] 
                [:,0] peak X coord
                [:,1] peak Y coord
                [:,2] peak value
                [:,3] mask value, i.e. is this a valid shift
    """
    out_peaks = np.zeros((n_peaks, 4), dtype = float)
    #out_peaks[:,2] = 0
    if return_blank:
        return out_peaks

    peak_pos = peak_local_max(cc_map, 1)
    if len(peak_pos) == 0:
        if np.std(cc_map) == 0.:
            return out_peaks
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
    m = np.array(out_peaks[:, 2] != 0, dtype = int)
    #mask in numerical form
    out_peaks[..., 3] = m
    return out_peaks

def combine_fsc_halves(prm1, prm2, tom_n, out_dir, ite,
                        combine_all = False):
    """
    combines model files and motive lists split for fsc
    tom_n [int] numbered from one
    combine_all [bool] combine and return paths for all MOTLs and all mods
    """

    if not isdir(out_dir):
        os.makedirs(out_dir)
    
    motls1, modfiles1, init_tomos = get_mod_motl_and_tomo(prm1, ite)
    motls2, modfiles2, init_tomos = get_mod_motl_and_tomo(prm2, ite)
    
    prm = PEETPRMFile(prm1)
    base_name = prm.prm_dict['fnOutput']

    new_mods = []
    new_motls = []
    if combine_all:
        tom_numbers = np.arange(len(modfiles1))
    else:
        tom_numbers = np.arange(tom_n -1, tom_n)
    for x in tom_numbers:
        outmod = join(out_dir, base_name + '_Tom%s_combined.mod' % str(x + 1))
        outcsv = join(out_dir, base_name + '_Tom%s_combined.csv' % str(x + 1))
        new_mods.append(outmod)
        new_motls.append(outcsv)

        #read in csv halves and interleave
        csv1 = PEETMotiveList(motls1[x])
        csv2 = PEETMotiveList(motls2[x])
        
        new_arr = np.zeros((len(csv1) + len(csv2), 20))
        if len(csv1) != len(csv2):
            #in case the two MOTLs are not the same lenght
            #place the remainder of the longer MOTL at the end
            shorter = min(len(csv1), len(csv2))
            if len(csv1) < len(csv2):
                remainder = csv2[len(csv1):]
            else:
                remainder = csv1[len(csv2):]
            new_arr[:shorter*2][::2] = csv1[:shorter]
            new_arr[:shorter*2][1::2] = csv2[:shorter]
            new_arr[shorter*2:] = remainder
        else:
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
        
        new_arr = np.zeros((len(mod1) + len(mod2), 3))
        if len(mod1) != len(mod2):
            #in case the two models are not the same lenght
            #place the remainder of the longer model at the end
            shorter = min(len(mod1), len(mod2))
            if len(mod1) < len(mod2):
                remainder = mod2[len(mod1):]
            else:
                remainder = mod1[len(mod2):]
            new_arr[:shorter*2][::2] = mod1[:shorter]
            new_arr[:shorter*2][1::2] = mod2[:shorter]
            new_arr[shorter*2:] = remainder
        else:
            new_arr[::2] = mod1
            new_arr[1::2] = mod2   
            
        new_mod = PEETmodel()
        for y in range(len(new_arr)):    
            new_mod.add_point(0, 0, new_arr[y])
        new_mod.write_model(outmod)
    
    if combine_all:
        #return new_mods[tom_n - 1], new_motls[tom_n - 1]
        #not sure why was I returning the equivalent of combine_all = False here
        
        return new_mods, new_motls
    else:
        return [new_mods[0]], [new_motls[0]]
    
    
    
    
def get_resolution(fsc, fshells, cutoff, apix = False, fudge_ctf = True,
                   get_area = 'cutoff'):
    """
    Get resolution value at cutoff. The correct approach is to get the lowest
    resolution where FSC crosses the cutoff. But for estimating minor improvement
    it's worth looking at bumps at higher resolution. (bumps due to CTF from
    a single tomogram...)
    
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
             fshells = False, fsc = False, simpleFSC = False):  
    """
    Expecting output from PEET calcUnbiasedFSC or simpleFSC

    Parameters
    ----------
    peet_dirs : list
        List of directories containing FSC files..
    out_dir : str
        Where to write the plot.
    cutoff : float, optional
        Cutoff. The default is 0.143.
    apix : float, optional
        Pixel size. The default is False.
    fshells : list, optional
        Specify freqShells.txt directly. The default is False.
    fsc : list, optional
        Specify arrFSCC.txt directly. The default is False.
    simpleFSC : bool, optional
        Look for simpleFSC output instead of calcUnbiasedFSC. The default is False.

    Returns
    -------
    float
        Resolution or area under FSC.

    """
    apix = float(apix)
    
    fig, axs = plt.subplots(1, 1, figsize = (7, 7))
    axs.axhline(0, color = 'black', lw = 1)
    axs.axhline(cutoff, color = 'grey', lw = 1, linestyle='dashed')
    axs.set(xlabel = 'Fraction of sampling frequency')
    axs.set(ylabel = 'Fourier Shell Correlation')
    res = []
    get_area = 'cutoff' 
    #this is set to the cutoff of the first fsc
    #the reasoning is that the cutoff is potentially unreliable
    
    if isinstance(peet_dirs, str):
        peet_dirs = [peet_dirs]
        
    if simpleFSC:
        fsc = [join(peet_dirs[x], 'simpleFSC.csv')
                    for x in range(len(peet_dirs))]
    else:
        if isinstance(fshells, bool):
            fshells = [join(peet_dirs[x], 'freqShells.txt')
                        for x in range(len(peet_dirs))]
        if isinstance(fsc, bool):
            fsc = [join(peet_dirs[x], 'arrFSCC.txt')
                    for x in range(len(peet_dirs))]

    for x in range(len(peet_dirs)):
        if simpleFSC:
            with open(fsc[x], 'r') as f:
                t = np.array([ff.split(',') for ff in f.read().strip('\n').split()], dtype = float)  
                ar_fshells = np.linspace(0,0.5,t.shape[0] + 1)[1:]
                #ar_fshells = t[:, 0]
                ar_fsc = t[:, 1]
        else:
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
        axR.set(xlabel = 'Resolution [Å]')
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

# def peet_to_clonemodel(mod, csv, outcsv, add_offsets = True):
    
#     motl = PEETMotiveList(csv)
#     mod = PEETmodel(mod).get_all_points()
#     offsets = motl.get_all_offsets()
#     if add_offsets:
#         mod += offsets
        
#     mat = motl.angles_to_rot_matrix()
#     xyz = [np.degrees(euler_from_matrix(mat[m], 'szyx')) for m in range(len(mat))]
#     xyz = np.flip(xyz, axis = 1)
    
    
#     with open(outcsv, 'w') as f:
#         for p in range(len(mod)):
#             f.write('X, Y, Z pixel coordinates, X, Y, Z slicer angles, ccc\n')
#             f.write('%s,%s,%s,%s,%s,%s,0\n' % (
#                 mod[p, 0], mod[p, 1], mod[p, 2],
#                 xyz[p, 0], xyz[p, 1], xyz[p, 2],
#                 ))
    
#def subtract_surroundings(full_tomo, mask, ali, tlt):
    
def tomo_fourier_mask(ang, tomo_size, out_mask = False, thick = 4, sigma = 2, z_pad = 10, use_half_fourier = False,
                      angle_offset = 0.75, rotx = False, invert = False):
    
    """
    
    based on https://www.sciencedirect.com/science/article/pii/S1047847714000288
    
    validation 
    s = 20 + 2
    t = 1
    test = np.zeros((s,s))
    test[1:t+1,1:t+1] = 1
    sh = (s/2.) - 1 - (t)/2
    test = shift(test, (sh, sh), prefilter = True, order = 1)
    ff,ax = plt.subplots(1,3)
    rotated_test = rotate(test, 90, prefilter = True, order = 1)
    test = shift(test, (0.5, 0.5), prefilter = True, order = 1)
    rotated_test = shift(rotated_test, (0.5, 0.5), prefilter = True, order = 1)
    rotated_test = rotated_test[1:-1, 1:-1]
    test = test[1:-1, 1:-1]
    print(test.shape)
    ax[0].imshow(test)
    ax[1].imshow(rotated_test)
    ax[2].imshow(rotated_test - test)
    """
    
    def def_plane(fit, tomo_size):
        X, Y = np.meshgrid(
                np.arange(-tomo_size[0]//2, tomo_size[0]//2), np.arange(-tomo_size[1]//2, tomo_size[1]//2))
        Z = fit[0]*X + fit[1]*Y + fit[2]
        return X, Y, Z
    
    def deg2slope(ang):
        x = np.cos(np.radians(ang))
        y = np.sin(np.radians(ang))
        return y/x
    
    def squeezed_angle(ang, ratio):
        x = np.cos(np.radians(ang))
        y = np.sin(np.radians(ang))*ratio
        return np.round(np.degrees(np.arctan(y/x)), decimals = 3)
    
    b = 1
    ratio = float(tomo_size[2])/tomo_size[0]
    tomo_size = np.array(tomo_size) + b*2
    #adding 4 to size. Image needs to be shifted by +0.5,+0.5 to align with imod origin
    
    if angle_offset:
        #the angles of slices in imod tomogram FFT are slightly off at higher angles,
        #almost like they're rotated in one direction by a scalar * 1/cos
        ang = np.array(ang) + angle_offset * np.sin(np.radians(np.absolute(ang)))
        #ang = np.array(ang) + angle_offset
    
    
    if tomo_size[0]%2:
        offset = True
        tomo_size[0] += 1
    else:
        offset = False
        
    tomo_size[2] += z_pad*2

    #z padding to avoid filterin edge effects
    t2d = np.zeros((tomo_size[2], int(np.ceil((np.sqrt(2)*tomo_size[0])/2)*2) + 2), dtype = 'complex64')
    m2d = np.zeros((tomo_size[2], tomo_size[0]), dtype = 'complex64')

    if offset:
        mask = np.zeros((tomo_size[2] - b*2 - z_pad*2, tomo_size[1] - b*2, tomo_size[0] - 1 - b*2), dtype = 'complex64')
    else:
        mask = np.zeros((tomo_size[2] - b*2 - z_pad*2, tomo_size[1] - b*2, tomo_size[0] - b*2), dtype = 'complex64')
    diff = int((t2d.shape[1] - tomo_size[0])/2)
    
    for a in range(len(ang)):
        
        t2d[...] = 0
        t2d[1: 1 + thick] = 1.+0j
        t2d = shift(t2d, ((tomo_size[2]/2.) - 1 - thick/2., 0), prefilter = True, order = 1)
        #this places it in scipy origin of rotation. Shift + 0.5,+0.5 for imod origin
        t2d = rotate(t2d, squeezed_angle(ang[a], ratio), reshape = False, prefilter = True, order = 1)
        #the fourier space where slices are placed has even dimensions in X, Z and is then squeezed into tomo size
        #instead I squeeze the Y component of the tilt angle
        t2d = gaussian_filter(t2d, sigma) 
        m2d = np.max((t2d[:, diff + 1:-diff + 1], m2d), axis = 0)

    
    if z_pad:
        m2d = m2d[z_pad:-z_pad]
    if offset:
        m2d = shift(m2d, (0, -0.5), prefilter = True, order = 1)
        m2d = m2d[:, :-1] 
        
    m2d /= np.max(m2d) #max 1
    m2d = shift(m2d, (0.5,0.5), prefilter = True, order = 1)
    m2d = m2d[b:-b, b:-b]
    mask[...] = m2d[:, None, :]
    if invert:
        mask = (mask - 1)
        mask *= np.sign(mask)
    if rotx:
        mask = np.rot90(mask, k = -1)
        #origin is no longer correct after rotation...quick fix, leaves an empty row of pixels
        mask = shift(mask, (0,1,0), prefilter = True, order = 1)
        
        warnings.warn('rotx may be in the wrong direction, not verified.')
    if use_half_fourier:
        mask = mask[:, :, mask.shape[2]//2 - 1:]
    if out_mask:
        write_mrc(out_mask, mask, mode = 4, set_float = False)
    
    return mask


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
    rotx : bool, optional
        If True, write in rotated orientation (90 degrees around X axis).
        If False, write in _full.rec orientation (default initial tilt output).
        The input model should be in the corresponding orientation.
        The default is False.

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
    
def corrsearch(refpath, querypath, out_dir, trim = [10, 10, 10],
                  outname = 'out_corrsearch3d.txt',
                  plot = True,
                  figsize = (12,5),
                  iter_n = False, rotx = False):
    """
    Wrapper for corrsearch3d
    
    refpath [str]
    querypath [str]
    out_dir [str]
    trim [list of 3 ints] trim edges around X,Y,Z
    num_patches [list of 3 ints] number of patches in X,Y,Z
    outname [str] name of corrsearch3d txt file
    rotx [bool] has the tomogram been rotated around X (clip rotx)
    """
    
    out_file = join(realpath(out_dir), outname)
    _, tomo_size = get_apix_and_size(refpath)
    
    for x in range(len(trim)):
        trim[x] = min(trim[x], tomo_size[x]//10)

    num_patches = np.array(np.round(tomo_size/np.min(tomo_size))*3, dtype = int)
    
    patch_s = int(np.min(tomo_size)/2.5)
    cmd = ['corrsearch3d',
           '-ref', refpath, #ref
           '-align', querypath, #query
           '-o', out_file, #transform txt file
           '-size', '%s,%s,%s' % (patch_s, patch_s, patch_s), #size of patches
           '-number', '%s,%s,%s' % (num_patches[0], num_patches[1], num_patches[2]),#number of patches
           '-x', '%s,%s' % (trim[0], tomo_size[0] - trim[0]),
           '-y', '%s,%s' % (trim[1], tomo_size[1] - trim[1]),
           '-z', '%s,%s' % (trim[2], tomo_size[2] - trim[2]),
           '-l', '0.3,0.05'
          ]
    run_generic_process(cmd)
    
    disp = []
    with open(out_file) as f:
        for x in f.readlines():
            disp.append(x.split())
    disp = np.array(disp[1:], dtype = float)
    if not rotx:
        #change to xyz coords, dxdydz, ccc
        disp = disp[:, [0,2,1,3,5,4,6]]
        tomo_size = tomo_size[[0, 2, 1]]
    
    
    #get rid of 10% of worst CCC
    min_cc = np.percentile(disp[:,-1], 10)
    if plot:
        rejected = disp[disp[:, -1] <= min_cc]
    disp = disp[disp[:, -1] > min_cc]

    fit, _ = fit_pts_to_plane(disp[:, [0, 1, 5]])
    #X1, Y1, Z1 = def_plane(fit, (np.max(disp[:, [0, 1, 5]], axis = 0)))
    rX, rY = np.rad2deg(np.arctan(fit[:2]))
    OFFSET, global_xtilt = np.rad2deg(np.arctan(fit[:2]))[:, 0]
    
    SHIFT = np.median(disp[:, [3, 5]], axis = 0)
    
    
    if plot:
        def line_in_plane(X, Y, fit):
            return X*fit[0] + Y*fit[1] + fit[2]
        lx = line_in_plane(np.arange(tomo_size[0]), int(tomo_size[1]/2), fit)
        ly = line_in_plane(int(tomo_size[0]/2), np.arange(tomo_size[1]),fit)
                          
        f, ax = plt.subplots(1,2, figsize = figsize)
        f.suptitle('Relative tomogram rotation')
        ax[0].scatter(disp[:, 0], disp[:, 5], alpha = 0.3, label = 'dZ shift')
        ax[0].scatter(rejected[:, 0], rejected[:, 5], alpha = 0.3,
                      label = 'rejected', c = 'r')
        ax[0].plot(lx, linewidth = 3, label = 'best fit')
        ax[0].title.set_text('Tilt around Y axis, %.02f deg' % OFFSET)
        ax[0].set(xlabel = 'X coordinate [pixels]', ylabel = 'dZ [pixels]')
        
        ax[1].scatter(disp[:, 1], disp[:, 5], alpha = 0.3, label = 'dZ shift')
        ax[1].scatter(rejected[:, 1], rejected[:, 5], alpha = 0.3,
                      label = 'rejected', c = 'r')
        ax[1].plot(ly, linewidth = 3, label = 'best fit')
        ax[1].title.set_text('Tilt around X axis, %.02f deg' % global_xtilt)
        ax[1].set(xlabel = 'Y coordinate [pixels]', ylabel = 'dZ [pixels]')

        ax[1].legend()
        if iter_n:
            plot_name = 'tomo_rotation%02d.png' % iter_n
        else:
            plot_name = 'tomo_rotation.png'
        plt.savefig(join(realpath(out_dir), plot_name))
        plt.close()
    return SHIFT, -OFFSET, -global_xtilt
        

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


#running stuff

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
        print(pgid)
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
        
def check_ssh(hostname):
    process = Popen(['ssh', '-q', hostname, 'exit'])
    #com = process.communicate()
    poll = process.poll()
    #print (poll)
    if poll == 0:
        pass
    else:
        raise Exception('Could not ssh into %s, %s' % (hostname, poll))    
        
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
            write_to_log(out_log,  str(datetime.datetime.now()) + '\n')
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
        out_log = realpath('run_generic_process_error.log')
        write_to_log(out_log,  str(datetime.datetime.now()) + '\n')
        write_to_log(out_log, (' ').join(cmd) + '\n')
        if 'process' in locals():
            kill_process(process)
            com = process.communicate()
            write_to_log(out_log, com[0].decode() + '\n' + com[1].decode())
        raise

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
                #poll stil returns integer, not bytes in python3
                #have to check if there is still output in the PIPE because
                #of race condition with process.poll()
                #some processchunks errors return 0 status:
                #when done, check return status but also if chunks are done
                if (process.poll() != 0
                or (process.poll() == 0 and chunks_done1 < total_chunks1)
                or (process.poll() == 0 and chunks_done1 == 0)):                    
                    com = [m.decode() for m in process.communicate()]
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
                or (process2.poll() == 0 and chunks_done2 < total_chunks2)
                #somehow I've managed to get chunks_done2 > total_chunks2?
                or (process2.poll() == 0 and chunks_done2 == 0)):                    
                    com = [m.decode() for m in process2.communicate()]
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
