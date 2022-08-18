# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:41:20 2022

@author: vojta
"""

from os.path import join, abspath, realpath, isfile
import numpy as np
from PEETModelParser import PEETmodel
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.spatial import KDTree
from subprocess import check_output
from skimage.filters import threshold_yen
from scipy.ndimage import gaussian_filter
import mrcfile
import time
#from scipy.spatial.distance import cdist
#from scipy.optimize import minimize
#import sys

from flexo_tools import find_nearest, get_apix_and_size, get_peaks

class Extracted_particles:
    """
    Class for processing shifts calculated by Flexo _extract_and_cc.
    
    """
    def __init__(self, sorted_pcls, cc_peaks = False, out_dir = False,
                 base_name = False,
                 chunk_base = False, n_peaks = 5, excludelist = [],
                 groups = False, tilt_angles = False, model_3d = False,
                 apix = False, exclude_worst_pcl_fraction = 0.1,
                 exclude_lowest_cc_fraction = False, **kwargs):

        self.sorted_pcls = sorted_pcls #see flexo_tools.fid2ndarray for deets
        self.out_dir = out_dir #flexo output dir
        self.xcor_dir = kwargs.get('xcor_dir')
        self.extracted_pcls_dir = kwargs.get('extracted_pcls_dir')
        self.base_name = base_name
        self.chunk_base = chunk_base
        self.n_peaks = n_peaks
        self.excludelist = excludelist
        self.groups = groups
        self.tilt_angles = tilt_angles
        self.tilt_subset = False   
        self.shifts = False
        self.shift_mask = False
        self.cc_values = False
        self.model_3d = model_3d
        self.apix = apix
        self.all_cc_maps = False
        
        self.flg_excludelist_removed = False  
        self.flg_outliers_removed = False
        self.exclude_worst_pcl_fraction = exclude_worst_pcl_fraction
        self.exclude_lowest_cc_fraction = exclude_lowest_cc_fraction
        
        if not self.xcor_dir:
            self.xcor_dir = join(self.out_dir, 'xcor_peaks')
        if not self.extracted_pcls_dir:
            self.extracted_pcls_dir = join(self.out_dir, 'extracted_particles')
        
        self.read_3d_model()     
        self.update_indices()   
        self.remove_tilts_using_excludelist()
        self.remove_group_outliers()

    
    def update_indices(self):
        """
        Updates some attributes, useful after removing tilts/particles.

        """
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
                    or isinstance(self.excludelist, bool) or not excludelist):
                tmp_excludelist = np.array(self.excludelist) - 1
                exc_mask = np.isin(self.tilt_indices, tmp_excludelist,
                                   invert = True)  
                self.sorted_pcls = self.sorted_pcls[exc_mask]
                self.tilt_angles = self.tilt_angles[exc_mask]
                self.update_indices()
                self.flg_excludelist_removed = True
        
    def remove_group_outliers(self, groups = False):
        """
        Remove particles that were not extracted due to their non_overlapping_pcls
        group being too small.
 
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
        """
        Read 3D (PEET) particle model 

        """
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

        
    def read_cc_peaks(self, out_dir = False, chunk_base = False,
                      spec_path = False, name_ext = 'peaks',
                      use_shift_gradient = False):
        """
        Reads pickled numpy arrays of cc_peaks (output from Flexo _extract_and_cc).
        Files be specified as tuple of paths or through directory path and
        base name.
        
        self.shifts has shape [num_tilts, num_pcls, n_peaks, 3
                                 (xshift, yshift)]
        """
        
        self.shifts = np.zeros((self.num_tilts, self.num_pcls,
                                  self.n_peaks, 2))
        self.cc_values = np.zeros((self.num_tilts, self.num_pcls,
                                  self.n_peaks))
        self.shift_mask = np.zeros((self.num_tilts, self.num_pcls,
                                  self.n_peaks, 2), dtype = bool)
        
        if out_dir:
            self.out_dir = out_dir
        if chunk_base:
            self.chunk_base = chunk_base
            
        for x in range(self.num_pcls):
            if self.out_dir and self.chunk_base:
                tmp_path = realpath(
                        join(self.xcor_dir, self.chunk_base
                             + '-%0*d_%s.npy' % (len(str(self.num_pcls)),
                                                 self.pcl_ids[x], name_ext)))
            elif isinstance(spec_path, tuple):
                tmp_path = spec_path[x]
            else:
                raise ValueError('Input paths not specified')
            tmp_arr = np.load(tmp_path)
            self.shifts[:, x] = tmp_arr[:, :, :2]
            self.cc_values[:, x] = tmp_arr[:, :, 2]
            self.shift_mask[:, x] = np.repeat(tmp_arr[:, :, 3, None],
                                              2, axis = 2)
        self.shift_mask = np.array(self.shift_mask, dtype = bool)
        
        self.shift_mask = np.logical_and((self.cc_values > 0)[..., None], self.shift_mask)

        #get rid of the worst scoring particles entirely
        if self.exclude_worst_pcl_fraction:

            if use_shift_gradient:
                #clean by sigma of the first derivative of shifts
                #in other words get rid of particles that jump all over the place
                #this performs slightly worse than excluding particles by ccc
                dmag = np.sqrt(self.shifts[:, :, 0, 0]**2 + self.shifts[:, :, 0, 1]**2)
                fder = np.gradient(dmag, axis = 0)
                fder = np.ma.masked_array(fder, mask = np.logical_not(self.shift_mask[:, :, 0, 0]))
                self.nice_tilts(find_nearest(self.tilt_angles, 0))
                fder = fder[self.tilt_subset]
                s_fder = np.ma.std(fder, axis = 0)
                thr_s = np.sort(s_fder)[int(np.round(s_fder.shape[0]*self.exclude_worst_pcl_fraction))]
                med_mask = s_fder <= thr_s
                self.shift_mask[:, med_mask] = False
                
            else:
                peak_median = np.ma.median(self.cc_values[:, :, 0], axis = 0)
                thr_cc = np.sort(peak_median)[int(np.round(peak_median.shape[0]*self.exclude_worst_pcl_fraction))]
                med_mask = peak_median <= thr_cc
                self.shift_mask[:, med_mask] = False 
            
    def read_cc_maps(self, map_base = 'ccmap'):
        """
        Reads in cross-correlation maps (output from Flexo _extract_and_cc).

        """
        
        if not self.extracted_pcls_dir:
            raise ValueError('Input paths not specified')
        tmp_mapname = join(self.extracted_pcls_dir, map_base
                + '_%0*d.mrc' % (len(str(self.num_pcls)), self.pcl_ids[0]))
        
        map_apix, map_size = get_apix_and_size(tmp_mapname)
        self.all_cc_maps = np.zeros((self.num_pcls, map_size[2], map_size[0], map_size[1]))
        
        for x in range(self.num_pcls):
            if self.out_dir and self.chunk_base:
                tmp_mapname = join(self.extracted_pcls_dir, map_base
                + '_%0*d.mrc' % (len(str(self.num_pcls)), self.pcl_ids[x]))
                
            self.all_cc_maps[x] = mrcfile.open(tmp_mapname).data

    def cluster_pcls(self, z_bias = 3, neighbour_distance = 10,
                     min_neighbours = 10):
        """
        Get 3D paticle distribution for distance weighting.

        Parameters
        ----------
        z_bias : int or float, optional
            The Z coordinates are simply multiplied by this factor.
            Values > 1 result in finer separation in the Z direction. 
             The default is 3.
        neighbour_distance : int or float, optional
            Particles within this distance (in voxels) will be groupped together.
            The default is 10.
        min_neighbours : int, optional
            Minimum number of particles per local group. The default is 10.

        Returns
        -------
        dst : ndarray
            Interparticle distances.
        pos : TYPE
            Indices of particles considered to be part of the local group.

        """
        
        #get neighbouring particles        
        tree = KDTree(self.model_3d*[1, 1, z_bias])
        dst, pos = tree.query(self.model_3d*[1, 1, z_bias],
                              self.num_pcls)
        
        min_neighbours = min(min_neighbours, self.num_pcls)
        for x in range(len(dst)):
            farenuf = np.where(dst[x] >= min(neighbour_distance, np.max(dst[x])))[0][0]
            dst[x][max(min_neighbours + 1, farenuf):] = np.inf
            pos[x][max(min_neighbours + 1, farenuf):] = dst.shape[0] + 1
    
        mn_nbrs = np.mean(np.sum(pos != dst.shape[0] + 1, axis = 1) - 1)
        std_nbrs = np.std(np.sum(pos != dst.shape[0] + 1, axis = 1) - 1)
        print('Mean number of neighbours for shift weighting %.2f +/- %.2f' 
              % (mn_nbrs, std_nbrs))
        
        return dst, pos


    def pick_shifts_basic_weighting(self, neighbour_distance = 70,
                                        n_peaks = 5,
                                        cc_weight_exp = 5,
                                        plot_pcl_n = False,
                                        z_bias = 3,
                                        shift_std_cutoff = 3,
                                        subtract_global_shift = True,
                                        figsize = (12,12),
                                        smooth_neighbouring_tilts = False,
                                        min_neighbours = 2,
                                        use_nbr_median = False):
        """
        This method attempts to select a single shift for each particle in each
        tilt from a set of cross correlation peaks. Typically, multiple peaks 
        are extracted from each cross-correlation map in case the maximum ccc
        peak has unreasonably high shift.
        
        1) Shifts that are [shift_std_cutoff] sigma larger than the median shift
        (per tilt) are removed.
        2) For each particle, the median shift of the local group of particles
            is determined. The shifts, for this particle, are then scored based
            on their ccc and distance from the median group shift. 
            
        
        The output is a mask with at most one shift left for each particle in 
        each tilt.
        

        Parameters
        ----------
        neighbour_distance : int or float, optional
            Particles within this distance (in voxels) will be groupped together.
            The default is 70.
        n_peaks : ind, optional
            Number of peaks in each cross-correlation map that wil be analysed.
            The default is 5.
        cc_weight_exp : int, optional
            Exponent for scaling ccc. Shifts are scored by 1/ccc**cc_weight_exp. 
            The default is 5.
        plot_pcl_n : list, optional
            Write plots for particles with specified indices. The default is False.
        z_bias : int or float, optional
            Used for groupping particles in 3D.
            The Z coordinates are simply multiplied by this factor.
            Values > 1 result in finer separation in the Z direction.
            The default is 3.
        shift_std_cutoff : int or float, optional
            Shifts larger than [shift_std_cutoff]*global shift std are disregarded.
            The default is 3.
        subtract_global_shift : bool, optional
            Subtract global shift for each tilt. This is only used during shift
            selection, the value of selected shifts written out as fiducial model
            are unaffected. 
            The default is True.
        figsize : tuple of 2 ints, optional
            Size of particle plot in inches. The default is (12,12).
        smooth_neighbouring_tilts : bool, optional
            Apply smoothing ACROSS tilts. The default is False.
        min_neighbours : int, optional
            Minimum number of particles for local groups. The default is 2.
        use_nbr_median: bool, optional
            Use median shift of local groups (using max ccc shift of each particle)
            instead of weighted mean. Performs slightly worse.
            The default is False.

        """
        
        def weighted_mean(vals, distances, exp = 2, axis = 0, stdev = False):
            """
            Weight shifts by 3D particle distances.

            Parameters
            ----------
            vals : ndarray
                Particle shifts.
            distances : ndarray
                3D particle distances.
            exp : int or float, optional
                Exponent for 3D distance scaling. Larger values favour closer 
                particles. The default is 2.
            axis : int, optional
                Axis of input ndarray along which the mean is calculated.
                The default is 0.
            stdev : bool, optional
                Return stdev. The default is False.

            Returns
            -------
            ndarray
                Local group weighted mean shifts.

            """            

            weights = 1./distances**exp
    
            if isinstance(vals, np.ma.masked_array):
                #need nominator or denominator to have a mask
                nom = np.sum(vals*weights, axis = axis)
                nom_mask = np.sum(np.logical_not(vals.mask), axis = axis)
                nom = np.ma.masked_where(nom_mask < 2, nom)
    
                weighted_mean = nom/np.sum(weights, axis = axis)
            else:
                weighted_mean = (np.sum(vals*weights, axis = axis)/
                                 np.sum(weights, axis = axis))
            if stdev:
                weighted_stdev = np.sqrt(np.sum(
                    weights*(vals - np.expand_dims(weighted_mean, axis))**2,
                                                                axis = axis)
                    /(((np.float(vals.shape[axis]) - 1)/vals.shape[axis])
                                            *np.sum(weights, axis = axis))
                                                )
                if isinstance(vals, np.ma.masked_array):
                    weighted_stdev = np.ma.masked_array(weighted_stdev, mask = weighted_mean.mask)
                return weighted_mean, weighted_stdev
            else:
                return weighted_mean 
    
        def cc_distance_weight_neighbours(shifts, cc_values, pcl_index, dst, pos,
                                          n_peaks = 5, cc_weight_exp = 3,
                                          cc_weight = True, use_self = False):
            """
            Calculate weighted mean shift of the local group of particles
            (local in 3D). Multiple shifts [n_peaks], weighted by their ccc
            are used for each particle.
            
            Parameters
            ----------
            shifts : ndarray
                All shifts.
            cc_values : ndarray
                All cc values.
            pcl_index : int
                Index of particle to be analysed.
            dst : ndarray
                Array of particle-particle distances.
            pos : ndarray
                Array of local group indices.
            n_peaks : int, optional
                Number of shifts used for analysis. The default is 5.
            cc_weight_exp : int or float, optional
                ccc scaling. Larger values favour high ccc shifts.
                The default is 3.
            cc_weight : bool, optional
                If False, use only max ccc shift for each particle.
                The default is True.
            use_self : bool, optional
                Include subject particle in mean calculation. The distance
                (weight) is set to 1. Using this seems to produce worse results,
                likely due to the subject paricle being weighted too strongly.
                The default is False.

            Returns
            -------
            wmn : ndarray
                Weighted mean shifts.
            wstd : ndarray
                stdev.

            """
    
            #pick neighbours within distance limit. First index is pcl_index,
            #filler value is 1 + num_pcls 
            if use_self:
                starting_index = 0
            else:
                starting_index = 1
            nbr_indices = np.unique(pos[pcl_index][starting_index:])[:-1]
            nbr_shifts = shifts[:, nbr_indices]
            nbr_shifts = nbr_shifts[:, :, :n_peaks]
            nbr_cccs = cc_values[:, nbr_indices]
            nbr_cccs = nbr_cccs[:, :, :n_peaks]
            nbr_dist = dst[pcl_index][dst[pcl_index] != np.inf][starting_index:]
            if use_self:
                nbr_dist += 1

            
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
            too_few_values_mask = np.zeros(wshifts.shape, dtype = bool)
            for y in range(len(wshifts)):
                tmp_dst = tilt_distance[lo_dst_indices[y]:hi_dst_indices[y]]
                min_idx = max(0, y - hw)
                max_idx = min(len(wshifts), y + hw + 1)
                tmp_wshifts = wshifts[min_idx:max_idx]
                if isinstance(tmp_wshifts, np.ma.masked_array):
                    #do not return a value if wshift[y] is masked
                    too_few_values_mask[y] = wshifts.mask[y, 0]
                    #mask element if there are less than two values to calculate mean
                    if np.logical_not(tmp_wshifts.mask[..., 0]).sum() < 2:
                        too_few_values_mask[y] = True
                    tmp_dst = np.ma.masked_array(tmp_dst, mask = tmp_wshifts.mask[..., 0])
                tilt_weighted[y] = weighted_mean(tmp_wshifts, tmp_dst[:, None])
    
            tilt_weighted = np.ma.masked_array(tilt_weighted, mask = too_few_values_mask)
            return tilt_weighted
    
        def score_shifts(shifts, cc_values, wshifts, pcl_index, n_peaks = 10, 
                          cc_weight_exp = 5, dist_weight_exp = 2):
            """
            Score shifts based on their distance to local group weighted mean
            and ccc.
            
            Score of shifts from a reference shift (wshift):
                [euclidian distance]**exp1/[ratio of max CCC]***exp2

            Parameters
            ----------
            shifts : ndarray
                All shifts.
            cc_values : ndarray
                All cc values.
            wshifts : ndarray
                local group weighted mean shifts.
            pcl_index : int
                Subject particle index.
            n_peaks : int, optional
                Number of considered shifts. The default is 10.
            cc_weight_exp : int or float, optional
                Scaling of ccc values. The default is 5.
            dist_weight_exp : int or float, optional
                Scaling of shifts. Larger values prioritise smaller shifts.
                The default is 2.

            Returns
            -------
            ndarray
                Particle mask with a single True value for the selected shift.

            """
    
            def euc_dist(a, b):
                """
                Calculate element-wise euclidian distance between two arrays.
                Masked arrays "should" be preserved correctly.

                """
            
                mask = False
                if isinstance(a, np.ma.core.MaskedArray):
                    mask = a.mask[..., 0]
                    a = deepcopy(a.data)
                if isinstance(b, np.ma.core.MaskedArray):
                    mask = b.mask[..., 0]
                    b = deepcopy(b.data)
                s =  np.sqrt((b[..., 0] - a[..., 0])**2
                                + (b[..., 1]-a[..., 1])**2)
                if not isinstance(mask, bool):
                    s = np.ma.masked_array(s, mask = mask)
                return s
         
            distance = euc_dist(shifts[:, pcl_index, :n_peaks],
                                wshifts[:, None])
            cc_ratios = (cc_values[:, pcl_index, :n_peaks]  
                         /cc_values[:, pcl_index, 0, None])
    
            #minimisation
            sc = (distance**dist_weight_exp)/cc_ratios**cc_weight_exp     
    
            #using masked arrays to keep any mask that's applied to wshifts
            m = np.ma.masked_where(sc != np.min(sc, axis = 1)[:, None], np.ones(sc.shape))
            m = np.logical_not(m.mask)
    
            #tilts that were not weighted due to missing wshifts:
            #pick best ccc
            if isinstance(wshifts, np.ma.masked_array):
                m1 = np.ma.masked_where(cc_ratios != np.max(cc_ratios, axis = 1)[:, None],
                                      np.ones(sc.shape))
                m1 = np.logical_not(m1.mask)
                m[wshifts[..., 0].mask] = m1[wshifts[..., 0].mask]
                
            if np.any(np.sum(m, axis = 1) > 1):
                #I don't have a good solution in case if there is more than
                #one peak left for each map
                #this *shouldn't* happen very often..........................
                #I just pick the highest CCC peak.........
                bummer = np.where(np.sum(m, axis = 1) > 1)
                mshape = m[bummer[0]].shape
                tmpm = np.zeros(mshape, dtype = bool)
                if tmpm.ndim > 1:
                    tmpm[:, 0] = True
                else:
                    tmpm[0] = True
                m[bummer[0]] =  tmpm
                warnings.warn('Pcl %s: %s maps had more than 1 shift after weighting'
                              % (pcl_index, mshape[0]))
            return m
    
        def plot_weighted_pcl(self, pcl_index, tilt_mean, weighted_std, 
                              figsize = (12,12), out_dir = False):
            f, ax = plt.subplots(2, 1, figsize = figsize)
            xvals = np.array(self.tilt_angles, dtype = int)
            for axis in range(2):
                ax[axis].set_xlim(np.min(self.tilt_angles) - 1, np.max(self.tilt_angles) + 1)
                ax[axis].axhline(0, c = 'k')
                #weighted mean
                ax[axis].scatter(xvals, tilt_mean[:, axis],
                                  c = 'tab:blue', alpha = 0.6,
                                  label = 'weighted mean')  
                ax[axis].plot(xvals, tilt_mean[:, axis],
                  c = 'tab:blue', alpha = 0.6)
                
                ax[axis].fill_between(xvals,
                      tilt_mean[:, axis] + weighted_std[:, axis],
                      tilt_mean[:, axis] - weighted_std[:, axis], alpha = 0.5,
                      label = 'neighbour STD') #color = 'tab:blue', 
                #max ccc shifts
                ax[axis].scatter(xvals, self.shifts[:, pcl_index, 0, axis],
                                  c = 'tab:purple', alpha = 0.6, marker = '2',
                                  label = 'max CCC shift', s = plt.rcParams['lines.markersize']**3)
                ax[axis].plot(xvals, self.shifts[:, pcl_index, 0, axis],
                  c = 'tab:purple', alpha = 0.6)
                #best scoring shift
                masked_shift = self.shifts[:, pcl_index, :, axis][
                                        self.shift_mask[:, pcl_index, :, axis]]   
                no_vals_mask = np.sum(self.shift_mask[:, pcl_index, :, axis], axis = 1, dtype = bool)
                masked_xvals = xvals[no_vals_mask]
                
                ax[axis].scatter(masked_xvals, masked_shift, c = 'tab:orange',
                  label = 'max scoring shift')  
                ax[axis].plot(masked_xvals, masked_shift, c = 'tab:orange')
    
                ax[axis].legend()
    
                ax[axis].set_ylabel('shift [pixel]')
                ax[axis].set_xlabel('tilt angle [degrees]')
    
            ax[0].set_title('X axis shift')
            ax[1].set_title('Y axis shift')                
            if out_dir:
                plt.savefig(join(self.out_dir, 'shift_scoring_pcl%0*d.png' % (
                            len(str(self.num_pcls)), pcl_index)))
                plt.close()     

                
        def moving_average(series, sigma=3):
            b = gaussian(20, sigma)
            average = filters.convolve1d(series, b/b.sum())
            var = filters.convolve1d(np.power(series-average, 2), b/b.sum())
            return var, average
            #nehalemslab.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behaviour/
            
        def get_nbr_median(pos, pcl_index, tmp_shifts):
            nbr_indices = np.unique(pos[pcl_index])[:-1]
            nbr_shifts = tmp_shifts[:, :, 0]
            nbr_shifts = nbr_shifts[:, nbr_indices]
            return np.ma.median(nbr_shifts, axis = 1), np.ma.std(nbr_shifts, axis = 1)
    
        #end of def###############################################################
    
        if isinstance(self.shifts, bool):
            try:
                self.read_cc_peaks()
            except:
                raise ValueError(
                    'No CC data. Use extracted_particles.read_cc_peaks()')     
    
    
        self.shift_mask[:, :, n_peaks:] = 0  #[ntilts, npcls, npeaks, 2]
        self.cc_values = np.ma.masked_array(self.cc_values,
                                          mask = np.logical_not(self.shift_mask[..., 0]))
        self.shifts = np.ma.masked_array(self.shifts,
                                          mask = np.logical_not(self.shift_mask))
        
        #removes the worst scoring shifts. Not recommended.
        if self.exclude_lowest_cc_fraction:
            cc_thr = np.percentile(np.ma.compressed(self.cc_values[:, :, 0]), 
                                   self.exclude_lowest_cc_fraction*100)
            tmp_mask = self.cc_values.data > cc_thr
            self.shift_mask = np.logical_and(self.shift_mask, tmp_mask[..., None])
            
            self.shifts = np.ma.masked_array(self.shifts,
                                          mask = np.logical_not(self.shift_mask))
            self.cc_values = np.ma.masked_array(self.cc_values,
                                          mask = np.logical_not(self.shift_mask[..., 0]))
            

        dst, pos = self.cluster_pcls(z_bias = z_bias, 
                                     neighbour_distance = neighbour_distance,
                                     min_neighbours = min_neighbours)

        #the point is to detect warping. global shifts are removed for analysis
        glob_med = np.ma.median(self.shifts[:, :, 0], axis = 1)
        if subtract_global_shift:
            tmp_shifts = deepcopy(self.shifts) - glob_med[:, None, None]
        else:
            tmp_shifts = deepcopy(self.shifts)
            
        #coarse clean: remove shifts that are +/- 3x std from median
        self.nice_tilts(find_nearest(self.tilt_angles, 0))
        #using 0 deg tilt as the re
        if shift_std_cutoff:
            glob_std = np.ma.std(tmp_shifts[self.tilt_subset, :, 0], axis = (0,1))        
            
            if subtract_global_shift:
                ref_shift = 0
            else:
                ref_shift = glob_med[:, None, None]
                
            tmp_shifts = np.ma.masked_greater(tmp_shifts,
                        ref_shift + glob_std*shift_std_cutoff)
            tmp_shifts = np.ma.masked_less(tmp_shifts,
                        ref_shift - glob_std*shift_std_cutoff)           
            
                
        for pcl_index in range(self.num_pcls):
            
            #cc_distance_weight
            if use_nbr_median:
                cc_dst_mean, cc_dst_std = get_nbr_median(pos, pcl_index, tmp_shifts)
            else:
                cc_dst_mean, cc_dst_std = cc_distance_weight_neighbours(
                    tmp_shifts, self.cc_values,
                    pcl_index, dst, pos, n_peaks = n_peaks,
                    cc_weight_exp = cc_weight_exp, cc_weight = True)

            #VP 29/6/2022 - think smoothing across neighbouring tilts might not be useful, now disabled by default
            if not smooth_neighbouring_tilts:
                tilt_mean = cc_dst_mean.data
            else:
                tilt_mean = tilt_weighted_mean(cc_dst_mean)
                # tilt_mean = np.array((moving_average(cc_dst_mean[:, 0])[1],
                #               moving_average(cc_dst_mean[:, 1])[1])).T
                
            pcl_mask = score_shifts(tmp_shifts, self.cc_values, tilt_mean,
                                pcl_index,
                                n_peaks = n_peaks,
                                cc_weight_exp = cc_weight_exp)
            
            self.shift_mask[:, pcl_index, :n_peaks] = pcl_mask[..., None]
            
    
            #this is the incorrect std to use here, but good enough for eyeballing
            if not isinstance(plot_pcl_n, bool):
                if np.isin(pcl_index, plot_pcl_n):
                    plot_weighted_pcl(self, pcl_index, tilt_mean, cc_dst_std,
                              figsize = figsize, out_dir = True)  

    
    def shifts_form_median_maps(self, neighbour_distance = 10, 
                                min_neighbours = 10,
                                z_bias = 3,
                                interp = 10, limit = 10,
                                n_peaks = 2):
        """
        Extract shifts from median projections (of nearby particles) of cc maps.
        Normally, only ~ 5 shifts/cc values are extracted from each map. The 
        reasoning here is that taking the median of whole maps should produce
        more robust results. 
        
        THIS IS EXTREMELY SLOW. Obv it could be paralelised but the results 
        are not encouraging.

        """
                
        def average_cc_maps(pos, pcl_index):
            
            #the reasoning here is to get a median shift from the cc maps
            #rather than from the median of individual peak positions
            
            nbr_indices = np.unique(pos[pcl_index][1:])[:-1] #exclude self and filler
            med_cc_maps = np.median(self.all_cc_maps[nbr_indices], axis = 0)
            
            return med_cc_maps      
        
        def orthogonal_stripe_reference(med_stack):
            
            stripe_ref = np.zeros(med_stack.shape)
            thr = np.array([threshold_yen(med_stack[x]) for x in range(len(med_stack))])
            mm = np.ma.masked_greater(med_stack, thr[:, None, None])  
            medx = np.ma.median(mm, axis = 1)
            medy = np.ma.median(mm, axis = 2)
            stripe_ref = (stripe_ref + medx[:, None]) + medy[:, :, None]
            
            return stripe_ref
        
        if isinstance(self.all_cc_maps, bool):
            self.read_cc_maps()
            
        if interp*limit*2 != self.all_cc_maps.shape[-1]:
            raise Exception('interp and limit does not match up with cc map size')
            
        dst, pos = self.cluster_pcls(z_bias = z_bias, 
                             neighbour_distance = neighbour_distance,
                             min_neighbours = min_neighbours)
        
        med_cc_maps = np.median(self.all_cc_maps, axis = 0)
        #there are often strong (mostly vertical) stripes
        stripe_ref = orthogonal_stripe_reference(med_cc_maps)
        med_cc_maps -= stripe_ref
        
        glob_med = np.array([get_peaks(gaussian_filter(m, 2), 1)[0, :2] for m in med_cc_maps])
        glob_med = glob_med/interp - limit


        med_map_shifts = np.zeros((self.num_tilts, self.num_pcls, n_peaks, 2))
        med_map_ccvals = np.zeros((self.num_tilts, self.num_pcls, n_peaks))
        med_map_mask = np.zeros((self.num_tilts, self.num_pcls, n_peaks), dtype = bool)
        
        
        #nbr_med_maps = np.zeros((self.num_tilts, self.num_pcls, interp*limit*2, interp*limit*2))
        
        st = time.time()
        lts = []
        
        for pcl_index in range(self.num_pcls):
            lt = time.time()
            nbr_med = average_cc_maps(pos, pcl_index)
            nbr_med -= stripe_ref
            nbr_med_s = np.array([get_peaks(gaussian_filter(m, 2), n_peaks) for m in nbr_med])
            nbr_med_s[:, :, :2] = nbr_med_s[:, :, :2]/interp - limit
            
            #nbr_med_maps[:, pcl_index] = nbr_med
            med_map_shifts[:, pcl_index] = nbr_med_s[:, :, :2]
            
            med_map_ccvals[:, pcl_index] = nbr_med_s[:, :, 2]
            med_map_mask[:, pcl_index] = nbr_med_s[:, :, 3]
            
            loop_time = time.time() - lt
            lts.append(loop_time)
            
            
            print('Elapsed %.1f min/%.1f min' % ((time.time() - st)/60,
                ((time.time() - st) + np.mean(lts)*(self.num_pcls - pcl_index))/60.), end = '\r')
            
        self.shifts = med_map_shifts
        self.cc_values = med_map_ccvals
        self.shift_mask = np.stack((med_map_mask, med_map_mask), axis = -1)
        self.shift_mask = np.logical_and((self.cc_values > 0)[..., None], self.shift_mask)
        
        #get rid of the worst scoring particles entirely
        if self.exclude_worst_pcl_fraction:
            peak_median = np.ma.median(self.cc_values[:, :, 0], axis = 0)
            thr_cc = np.sort(peak_median)[int(np.round(peak_median.shape[0]*self.exclude_worst_pcl_fraction))]
            med_mask = peak_median <= thr_cc
            self.shift_mask[:, med_mask] = False             
            


    
    def write_fiducial_model(self, ali = False, use_local_medians = False):
        """
        Write selected shifts as an IMOD fiducial model.

        Parameters
        ----------
        ali : str, optional
            Path to tilt series. Used for imodtrans. The default is False.
        use_local_medians : bool, optional
            Use medians of local groups as shifts rather than the individual
            particles. The default is False.

        Returns
        -------
        str
            Path to output fiducial model.

        """

        def compress_masked_array(vals, axis=-1, fill=1000):
            """
            Reorder shifts so that unmasked values come first.
            """
            #https://stackoverflow.com/questions/46354509/transfer-unmasked-elements-from-maskedarray-into-regular-array
            cnt = vals.mask.sum(axis=axis)
            shp = vals.shape
            num = shp[axis]
            mask = (num - cnt[..., np.newaxis]) > np.arange(num)
            n = fill * np.ones(shp)
            n[mask] = vals.compressed()
            n = np.ma.masked_where(n == fill, n)
            return n        
        
        shifts = np.ma.masked_array(deepcopy(self.shifts), mask = np.logical_not(self.shift_mask))
        for m in range(2):
            shifts[..., m] = compress_masked_array(shifts[..., m])
    
        shifts = shifts[:, :, 0]
        
        
        if use_local_medians:
            dst, pos = self.cluster_pcls(z_bias = 3, 
                         neighbour_distance = 70,
                         min_neighbours = 10)
            for x in range(shifts.shape[1]):
                shifts[:, x] = np.ma.median(shifts[:, np.unique(pos[x][1:])[:-1]], axis = 1)
            
        
        shifted_pcls = np.ma.masked_array(deepcopy(self.sorted_pcls[:, :, :3]), 
                                          mask = np.repeat(shifts.mask[..., 0, None], 3, axis = 2))
        shifted_pcls[:, :, :2] = (shifted_pcls[:, :, :2] - shifts)
        
        outmod = PEETmodel() 
    
        for p in range(self.num_pcls):
            #I suspect an contour is already created with PEETmodel() instance,
            #no need to add it for the first pcl
            if p != 0:
                    outmod.add_contour(0)        
            for r in range(self.num_tilts):
                if np.any(shifted_pcls[r, p]):
                    #np.all would skip tilt 0
                    outmod.add_point(0, p, shifted_pcls[r,p])
    
        outmod_name = abspath(join(self.out_dir, self.base_name + '.fid'))
        outmod.write_model(outmod_name)
    
        if ali:
            #set image coordinate information from the given image file
            check_output('imodtrans -I %s %s %s' % 
                         (ali, outmod_name, outmod_name), shell = True)
        return outmod_name


    def nice_tilts(self, zero_tlt = False, min_size = 5):
        """
        Select a subset of tilts with the lowest tilt angle/highest signal.

        Parameters
        ----------
        zero_tlt : int or bool, optional
            Index of the tilt with the highest signal. If False, use cccs to 
            find the nicest tilt. The default is False.
        min_size : int, optional
            Minimum number of tilts. Uses this or 1/4 of the stack size, whatever
            is larger. The default is 5.

        """
        #define middle tilts based on middle tilt or median CCC
        if not isinstance(zero_tlt, bool):
            step = max(min_size, self.num_tilts//4)
            bot = int(zero_tlt - 1 - (step - 1)//2)
            #zero_tlt is numbered from 1, so take 1 off
            top = int(zero_tlt - 1 + (step - 1)//2 + 1)   
            self.tilt_subset = np.arange(bot, top)
        else:
            #pick tilts with the highest median CCCs
            tilt_medians = np.ma.median(self.cc_values[:, :, 0], axis = 1)
            tilt_median_order = np.argsort(tilt_medians)[::-1]
            num_hi_tilts = int(np.max((min_size, self.num_tilts/4)))
            gappy = np.sort(tilt_median_order[:num_hi_tilts])
            self.tilt_subset = np.arange(gappy[0], gappy[-1])
      

    def plot_median_cc_vs_tilt(self, out_dir = True, figsize = (12,8)):
        #x values are not continuous but easier to look at as line plot...
        #gaps in X (tilts) are filled in and not ommited.  Something to play 
        #with in the future...
        if out_dir and isinstance(out_dir, bool):
            out_dir = self.out_dir
        f, ax = plt.subplots(1, figsize = figsize)
        peak_median = np.ma.median(self.cc_values[:, :, 0], axis = 1)
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
        if self.exclude_lowest_cc_fraction:
            cc_thr = np.percentile(np.ma.compressed(self.cc_values[:, :, 0]), 
                                   self.exclude_lowest_cc_fraction*100)
            ax.axhline(cc_thr, label = 'min allowed ccc', c = 'r', linestyle = '--')
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
        peak_median = np.ma.median(self.cc_values[:, :, 0], axis = 0)
        
        order = np.argsort(peak_median)[::-1]
        peak_median = peak_median[order]
        q25, q75 = np.percentile(self.cc_values[:, :, 0],[25, 75], axis = 0)
        q25, q75 = q25[order], q75[order]
        pmin = np.min(self.cc_values[:, :, 0], axis = 0)[order]
        pmax = np.max(self.cc_values[:, :, 0], axis = 0)[order]
        xvals = np.array(self.pcl_ids, dtype = int)[order]
        ax.plot(list(range(len(xvals))), peak_median, color = 'midnightblue',
                label = 'median')
        if self.exclude_worst_pcl_fraction:
            thr_cc = np.sort(peak_median)[int(np.round(peak_median.shape[0]*self.exclude_worst_pcl_fraction))]
            ax.axhline(thr_cc, label = 'particles excluded by median ccc', c = 'r', linestyle = '--')
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
     
        
    def split_for_processchunks(self, pcls_per_core):
        """
        Split sorted_pcls for parallelisation.

        """
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
          
        #so these are particle indices and group IDs
        return c_tasks  
