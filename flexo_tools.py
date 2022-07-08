# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:48:06 2022

@author: vojta
"""
import os
from copy import deepcopy
from os.path import isfile, join, realpath, split, isdir
import glob
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from subprocess import check_output
import mrcfile
from PEETPRMParser import PEETPRMFile 
from PEETModelParser import PEETmodel
from PEETMotiveList import PEETMotiveList
from mdoc_parser import Mdoc_parser
import warnings
from scipy import fftpack as fft
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from numpy.fft import fftfreq
from scipy.signal import butter, freqs
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import zoom, shift, rotate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.signal import convolve2d
from skimage.feature import peak_local_max
from transformations import euler_from_matrix

from definite_functions_for_flexo import run_generic_process, run_split_peet, ctf, get_resolution


#from definite_functions_for_flexo import read_defocus_file

def norm(x):
    """normalise to 0 mean and 1 standard deviation"""
    return (x - np.mean(x))/np.std(x)

def machines_from_imod_calib():
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

    Returns
    -------
    None.

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

    Returns
    -------
    None.

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
    """I followed the instructons for ctfphaseflip where tilts are listed in 
    reverse order."""
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
    reads deofocus file, converts to IMOD defocus 3 format.
    Optional arguments required if input is CTFFind format.
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
                   apix = False, tlt = False,
                   excludelist = False):
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
    Returns:
        [numpy.ndarray] frequency at cutoff, resolution at cutoff,
                        FSC at nyquist
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

def peet_halfmaps(peet_dir, prm1, prm2, machines):
    
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
    fsc_log = join(peet_dir, 'calcUnbiasedFSC.log')
    print('Running calcUnbiasedFSC...')
    run_generic_process(['calcUnbiasedFSC', prm1, prm2], out_log = fsc_log)

    os.chdir(cwd)

def optimal_lowpass(y):
    """Returns optimal 1/spatial frequency for a specified electron dose"""
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
        list of optimal resolution for low pass filter.

    """
    """
    Generate list of accummulated electron doses/tilt or the optimal
    filtering frequencies.
    zerotlt [int] numbered from 1
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

def ctf_convolve_andor_butter(inp, apix, V = 300000.0, Cs = 27000000.0,
                              ampC = 0.07, ps = 0,
                              lowpass = 0, defocus = 0, butter_order = 4,
                              no_ctf_convolution = False,
                              padding = 10, phaseflip = True):

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

def extract_2d(stack, sorted_pcls, box_size,
                          offsets = False, normalise_first = False):
    """Simplified version of 2D particle extraction from a 2D stack.
    Input:
        stack [str or numpy array] path of image stack or mrcfile array
        sorted_pcls [2D numpy array] [tilt number: xcoord, ycoord, z]
                    - coordinates of particles to be extracted
        box_size [list of even natural numbers] e.g. [40,38]

    Returns extracted particles [numpy array [particle, tilt number, [X,Y]]]
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
        subpixel = 'full_spline', ccnorm = 'none',
        permissive = True):
    """
    ccnorm can be 
    'none' - no normalisation
    'phase_corr' - normalised phase correlation
    'by_autocorr' - normalise by autocorrelations
    
    Outputs normalised cross-correlation map of two images.
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
    
def plot_fsc(peet_dirs, out_dir, cutoff = 0.143, apix = False,
             fshells = False, fsc = False, simpleFSC = False):  
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

