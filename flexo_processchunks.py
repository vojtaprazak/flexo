#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:33:34 2019

@author: vojta
"""
#execfile('/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/test_new_toy_tomo/tmp_output/tmp_output_file.py')

#from definite_functions_for_flexo import *

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from os.path import join, isfile, isdir, split
from PEETPRMParser import PEETPRMFile 
from definite_functions_for_flexo import (run_processchunks, csv_sync,
    p_shifts_to_model_points, format_align, format_newst, format_tilt,
    imodscript, match_tomos, reconstruct_binned_tomo,
    run_generic_process, run_split_peet, plot_fsc, prepare_prm,
    extracted_particles
    #get_apix_and_size
    )


def just_flexo(
        #paths
        rec_dir, out_dir, base_name, defocus_file, tomo,
        ali, tlt, xf, localxf, reprojected_mod, st, orig_rec_dir,
        sub_ali_list, #path or list of paths
        plotback_ali_list, #path or list of paths
        groups, #path or array
        ssorted_pcls, #path or array
        #tomogram and running parameters
        tomo_size, apix, tomo_binning, output_binning, thickness, box,
        machines, pcls_per_core,
        global_xtilt, excludelist, axiszshift,
        separate_group, zero_tlt, SHIFT, OFFSET, 
        dose, n_patches, global_only, spec_tiny_size, globalXYZ, fidn, 
        #imaging params
        V, Cs, ampC, wl, ps,
        #flexo params
        butter_order = 4,
        dosesym = False,
        orderL = True,
        pre_exposure = 0,
        limit = 10,
        interp = 4,
        smooth = True,
        centre_bias = 0.01,
        thickness_scaling = 0.5,
        debug = 0,
        iters_done = 0,
        #dev
        allow_large_jumps = False,
        orthogonal_subtraction = False,
        shifts_exist = False,
        #peet_params
        tom_n = False,
        prm = False,
        prm2 = False,
        ite = 0,
        cutoff = 0.143,
        search_rad = False,
        phimax_step = False,
        psimax_step = False,
        thetamax_step = False,
        n_peaks = 100
        ):
    
    
    #turn display off
    plt.ioff()

    #chunk name needs to be different from "model_from_peet" chunks
    chunk_base = base_name + '_flexo'

    #this is a standin for adding "flexo installation path" to PATH    
    path = (',').join(['"' + x + '"' for x in sys.path if x != ''])

    #this should become optional to specify. meaning that particles would
    #not be saved
    test_dir = join(out_dir, 'extracted_particles')   
    if not isdir(test_dir):
        os.makedirs(test_dir)


    particles = extracted_particles(ssorted_pcls, out_dir = out_dir,
                                    base_name = base_name,
                                    chunk_base = chunk_base,
                                    n_peaks = 100,
                                    tilt_angles = tlt)

#VP EDIT: now within class extracted_particles
#    #inputs can be arrays or paths to pickled arrays
#    if isinstance(groups, str):
#        groups = np.load(groups)
#    if isinstance(ssorted_pcls, str):
#        ssorted_pcls = np.load(ssorted_pcls)
    
    #inputs can be single paths or lists of paths
    if isinstance(sub_ali_list, str):
        sub_ali_list = [sub_ali_list]
    if isinstance(plotback_ali_list, str):
        plotback_ali_list = [plotback_ali_list]

# VP EDIT: now within class extracted_particles   #######
#    #split pcls among available cores
#    #probably suggest values for different box sizes....
#    #rearrange ssorted_pcls by groups, only operate with indices
#    group_pcls = []
#    for x in range(len(particles.groups)):
#        group_pcls.extend(sorted_pcls[0, groups[x], 3:5])  
#    group_pcls = np.array(group_pcls, dtype = int)
##    group_pcls = np.hstack(group_pcls)
#    c_tasks = [group_pcls[x:x + pcls_per_core]
#                for x in np.arange(len(group_pcls) - 1, step = pcls_per_core)]

#VP EDIT: Now within class extracted_particles
#    #remove particles from ssorted_pcls that were excluded due to 
#    #non_overlapping_pcl groups
#    if not isinstance(groups, bool):
#        if len(groups.shape) == 1:
#            #in case there is only one group
#            gmask = groups
#        else:
#            gmask = np.sum(groups, axis = 0, dtype = bool)
#        ssorted_pcls = ssorted_pcls[:, gmask]
##    #remove tilts based on excludelist
#    if not isinstance(excludelist, bool):
#        excludelist = np.array(excludelist) - 1
#        exc_mask = np.isin(range(len(ssorted_pcls)), excludelist,
#                           invert = True)
#        ssorted_pcls = ssorted_pcls[exc_mask]
#       
#    tilt_angles = [float(x.strip('\n')) for x in open(tlt)][exc_mask]

    particles.remove_tilts_using_excludelist(excludelist)
    particles.remove_group_outliers(groups)
    c_tasks = particles.split_for_processchunks(pcls_per_core)

    def fstr(flist, c_task):
        """convert array elements into a printable list of strings """
        ind = np.array((np.unique(c_task[:,1])), dtype = int)
        return (',').join(['"' + flist[y] + '"' for y in ind])

    #reduce number of plots being written with debug = 2
    if debug < 3:
        plot_pcls = np.linspace(0, particles.num_pcls,
                                min(20, particles.num_pcls), dtype = int)
    else:
        plot_pcls = np.arange(particles.num_pcls)

    #things that can be list or bool have to be treated separately
    if isinstance(excludelist, bool):
        exc = '>excludelist = False'
    else:
        exc = '>excludelist = %s' % (',').join([str(y) for y in excludelist])

    if not shifts_exist:
        for x in range(len(c_tasks)):
            out_s = (
            '>sys.path.extend([%s])' % path,
            '>from definite_functions_for_flexo import p_extract_and_cc',
            '>sub_ali_list = [%s]' % fstr(sub_ali_list, c_tasks[x]),
            '>plotback_ali_list = [%s]' % fstr(plotback_ali_list, c_tasks[x]),
            '>reprojected_mod = "%s"' % reprojected_mod,
            '>defocus_file = "%s"' % defocus_file,
            '>tlt = "%s"' % tlt,
            '>box = [%s, %s]' % (box[0], box[1]),
            '>out_dir = "%s"' % out_dir,
            '>chunk_base = "%s"' % chunk_base,
            '>chunk_id = %s' % x,
            '>n_pcls = %s' % particles.num_pcls,
            '>pcl_indices = [%s]' % (',').join(
                    [str(y) for y in c_tasks[x][:,0]]),
            '>group_ids = [%s]' % (',').join([str(y) for y in c_tasks[x][:,1]]),
            exc,
            '>zero_tlt = %s' % zero_tlt,
            '>dosesym = %s' % dosesym,
            '>orderL = %s' % orderL,
            '>dose = %s' % dose,
            '>pre_exposure = %s' % pre_exposure,
            '>apix = %s' % apix,
            '>V = %s' % V,
            '>Cs = %s' % Cs,
            '>ampC = %s' % ampC,
            '>ps = %s' % ps,
            '>butter_order = %s' % butter_order,
            '>limit = %s' % limit,
            '>interp = %s' % interp,
            '>centre_bias = %s' % centre_bias,
            '>thickness_scaling = %s' % thickness_scaling,
            '>debug = %s' % debug,
            '>plot_pcls = [%s]' % (',').join([str(y) for y in plot_pcls]),
            '>test_dir = "%s"' % test_dir,
            '>allow_large_jumps = %s' % allow_large_jumps,
            '>xf = "%s"' % xf,
            '>orthogonal_subtraction = %s' % orthogonal_subtraction,
            '>n_peaks = %s' % n_peaks,
            '>p_extract_and_cc(',
            '>            sub_ali_list,',
            '>            plotback_ali_list,',
            '>            reprojected_mod,',
            '>            defocus_file,',
            '>            tlt,',
            '>            box,',
            '>            out_dir,',
            '>            chunk_base,',
            '>            chunk_id,',
            '>            n_pcls,',
            '>            pcl_indices,',
            '>            group_ids,',
            '>            excludelist,',
            '>            zero_tlt, ',   
            '>            dosesym,',
            '>            orderL,',
            '>            dose,',
            '>            pre_exposure,',
            '>            apix,',
            '>            V,',
            '>            Cs,',
            '>            ampC,',
            '>            ps,',
            '>            butter_order,',
            '>            limit,',
            '>            interp, ',
            '>            centre_bias,',
            '>            thickness_scaling,',
            '>            debug,',
            '>            plot_pcls,',
            '>            test_dir,',
            '>            allow_large_jumps,',
            '>            xf,',
            '>            orthogonal_subtraction,',
            '>            n_peaks',
            '>            )'
            )
                      
            with open(join(out_dir, chunk_base + '-%03d.com' % x), 'w') as f:
                for y in range(len(out_s)):
                    f.write(out_s[y] + '\n')
        #run extract_and_cc chunks
        run_processchunks(chunk_base, out_dir, machines)

#VP EDIT: now within class extracted_particles
#    cc_peaks = np.zeros((ssorted_pcls.shape[0], ssorted_pcls.shape[1],
#                         n_peaks, 3))
#    for x in range(ssorted_pcls.shape[1]):
#        pcl_index = int(ssorted_pcls[0, x, 3])
#        tmp_in = join(out_dir, 'cc_peaks', chunk_base
#                      + '-%03d_peaks.npy' % pcl_index)
#        cc_peaks[:, x] = np.load(tmp_in) 
        
    particles.read_cc_peaks()
    particles.plot_median_cc_vs_tilt()
    particles.plot_particle_med_cc()
    cos_model = particles.fit_cosine()
    
    
    
    
    
    
    
    
    plot_cc_per_tilt(cc_peaks, ssorted_pcls, out_dir = out_dir)
    cos_model = fit_thickness(peak_median, tilt_angles, return_model = True)
    
    
    dists = euc_dist(cc_peaks[:,:,:,:2])
    is_dists = intershift_euc_dist(cc_peaks[:,:,:,:2])
    #weighing looks like a bad idea   
    weighed_cc = weigh_cc_by_tilt(cc_peaks, ssorted_pcls[:, 0, 2])
    weighed_cc_peaks = deepcopy(cc_peaks)
    weighed_cc_peaks[:, :, :, 2] = weighed_cc
    
    plot_dists_vs_cc(weighed_cc, dists)
    plot_cc_per_tilt(weighed_cc_peaks, ssorted_pcls, out_dir = False)

    #instead plot x, y, peak val in 3D and look for clustering per particle/per group

def is_dist_over_cc_value(cc_peaks, npeaks = 1, plot = True):
    is_dist = intershift_euc_dist(cc_peaks[:,:,:,:2])
    mm = np.array(np.where(is_dist[:,:,:npeaks] == 0, is_dist[:,:,:npeaks], 1), dtype = bool)
    p_over_d = cc_peaks[:,:,:npeaks,2][mm]/is_dist[:,:,:npeaks][mm]
    if plot:
        plt.hist(p_over_d, 50)
        plt.show()
    return p_over_d

def weigh_cc_by_tilt(cc_peaks, tilt_angles):
    peak_median = np.median(cc_peaks[:, :, 0, 2], axis = 1)
    cos_model = fit_thickness(peak_median, tilt_angles, return_model = True)
    norm_peaks = cc_peaks[:, :, :, 2]/cos_model[:, None, None]
    return norm_peaks
    
def plot_dists_vs_cc(cc_vals, dists, cc_multiplier = 100):
    dists = dists[cc_vals != 0.]
    cc_vals = cc_vals[cc_vals != 0.]
    plt.figure()
    plt.scatter(cc_vals, dists)
    plt.figure()
    heatmap, xedges, yedges = np.histogram2d(
                               cc_vals*cc_multiplier,  dists, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis')
    plt.show()
    
def euc_dist(data):
    """
    Input:
        data [whatever number of dim by 2] assuming the last dimension are coords
    Output:
        distances [whatever nunber of dim]
    """
    return np.linalg.norm(0 - data, axis = data.ndim - 1)



def plot_cc_per_tilt(cc_peaks, ssorted_pcls, out_dir = False):
    #x values are not continuous but easier to look at as line plot...
    #gaps in X (tilts) are filled in and not ommited.  Something to play with
    #in the future...
    plt.figure()
    peak_median = np.median(cc_peaks[:,:,0,2], axis = 1)
    q25, q75 = np.percentile(cc_peaks[:,:,0,2], [25, 75], axis = 1)
    pmin = np.min(cc_peaks[:,:,0,2], axis = 1)
    pmax = np.max(cc_peaks[:,:,0,2], axis = 1)
    xvals = np.array(ssorted_pcls[:,0,2], dtype = int)
    plt.plot(xvals, peak_median, color = 'midnightblue')
    plt.fill_between(xvals, pmax, pmin, color = 'lightsteelblue')
    plt.fill_between(xvals, q25, q75, color = 'cornflowerblue')
    plt.xlabel('tilt number')
    plt.ylabel('cross correlation')
    plt.suptitle('Cross correlation peak median, quartiles and min/max.')
    if out_dir:
        plt.savefig(join(out_dir, 'tilt_cc_values.png'))
        plt.close()



def fit_thickness(data, tilt_angles, init_scale=1, init_const=0,
                  return_model = False):
    """Author: Daven Vasishtan
    """
    def f(p):
        model = p[0] * np.cos(np.radians(tilt_angles))+p[1]
        return np.sum((data-model)**2)
    new_params = minimize(f, [init_scale, init_const])
    if not return_model:
        return new_params
    else:
        return (new_params.x[0] * np.cos(np.radians(tilt_angles))
                + new_params.x[1])



        
    
    
    
    
    
    

    #combine csv parts
    comb_csv, sorted_shifts = csv_sync(out_dir, chunk_base,
                                       return_sorted = True)
    #add shifts to point coords and write fiducial model
    outmod_name = p_shifts_to_model_points(ssorted_pcls, sorted_shifts,
                                out_dir, excludelist, groups, ali, debug)

 
    
    #don't know what to do with the .xtlt file. under what circumstances
    #does tiltalign create it?
    xfile = False
  
    
    output_xf, localxf, output_tlt, zfac = format_align(out_dir,
                            base_name, ali, tlt, tomo_binning,
                            outmod_name, axiszshift,
                            xf, separate_group, fidn, n_patches,
                            global_only, globalXYZ, OFFSET, excludelist)
    #OFFSET has to be 0 in tlt.com if it's specified in align.com !
    imodscript('align.com', os.path.realpath(out_dir))
    OFFSET = 0
    output_ali = format_newst(base_name, out_dir, st, output_xf, tomo_binning)
    output_rec = format_tilt(base_name, out_dir, output_ali, output_tlt,
                             tomo_binning, thickness, global_xtilt, 0,
                             SHIFT, xfile, localxf, zfac, excludelist)
    
    if not isfile(output_xf):
        #i think this error message is no longer relevant
        raise ValueError("""align.com has failed.
        Try opening unwarped_particles.mod with an aligned stack.
        Go to special > bead helper > fix contours, save and try again.""")






    #make small tomogram and check for global positioning relative to original
    #tomogram (or tomogram from previous iteration)
    SHIFT, OFFSET, global_xtilt = match_tomos(tomo_binning, out_dir,
                            base_name, orig_rec_dir, spec_tiny_size)

    #final align.com
    output_xf, localxf, output_tlt, zfac = format_align(out_dir, base_name,
                            ali, tlt, tomo_binning, outmod_name,
                            axiszshift,
                            xf, separate_group, fidn, n_patches,
                            global_only, globalXYZ, OFFSET,
                            excludelist)
    #OFFSET has to be 0 in tlt.com if it's specified in align.com !
    imodscript('align.com', os.path.realpath(out_dir))
    OFFSET = 0
    #there need to be comfiles with default naming and default binning for
    #future iterations
    output_ali = format_newst(base_name, out_dir, st, output_xf, tomo_binning)
    #OFFSET has to be 0 in tlt.com if it's specified in align.com !
    output_rec = format_tilt(base_name, out_dir, output_ali, output_tlt,
                             tomo_binning, thickness,
                             global_xtilt, 0, SHIFT,
                             xfile, localxf, zfac, excludelist)   
    
#####prepare peet prm - needed to decide on final tomogram binning

    peet_dir = join(out_dir, 'peet')
    if not isdir(peet_dir):
        os.makedirs(peet_dir)
    cwd = os.getcwd()
    os.chdir(peet_dir)
    fsc1d = join(peet_dir, 'fsc1/')
    fsc2d = join(peet_dir, 'fsc2/') 
    if not os.path.isdir(fsc1d):
        os.makedirs(fsc1d)
    if not os.path.isdir(fsc2d):
        os.makedirs(fsc2d)            

    if not prm2:
        #first format parent .prm before splitting it for fsc
        peet_bin, new_prm, peet_apix = prepare_prm(
                prm, ite, tom_n, out_dir, base_name, st, peet_dir,
                search_rad = search_rad,
                phimax_step = phimax_step,
                psimax_step = psimax_step,
                thetamax_step = thetamax_step)
        
        r_new_prm = PEETPRMFile(new_prm)
        print 'Splitting current PEET run for FSC.'
        r_new_prm.split_by_classID(ite, fsc1d,
                        classes = [1], splitForFSC = True, writeprm = True)
        r_new_prm.split_by_classID(ite, fsc2d,
                        classes = [2], splitForFSC = True, writeprm = True)  
        prm1 = join(fsc1d, base_name + '_fromIter%s_cls1.prm' % ite)
        prm2 = join(fsc2d, base_name + '_fromIter%s_cls2.prm' % ite)        
    else:
        #if PEET was already split for FSC
        peet_bin, prm1 = prepare_prm(
                prm, ite, tom_n, out_dir, base_name, st, fsc1d)        
        peet_bin2, prm2 = prepare_prm(
                prm2, ite, tom_n, out_dir, base_name, st, fsc2d)
        if peet_bin != peet_bin2:
            raise ValueError('The two PEET half data-sets do not have the'
                             + ' same binning.')

      
#    unbin_apix, _ = get_apix_and_size(st)
#    peet_apix = peet_bin * unbin_apix
    
    #binning of the "default" tomo and ali needs to be preserved across 
    #iterations, including whats in tilt.com and align.com
    #all tomograms are first made with _binx extension, then the "default name"
    #softlink is made to a tilt series, a full tomo and tomo with the correct
    #binning
    
    default_tomo = join(out_dir, base_name + '.rec')
    default_full = join(out_dir, base_name + '_full.rec')
    default_ali = join(out_dir, base_name + '.ali')
    rel_bin = max(peet_bin, tomo_binning)/float(min(peet_bin, tomo_binning))
    
    #always make lowest binning tomo:
    print 'Reconstructing tomograms...'
    output_rec, out_tomo, output_ali = reconstruct_binned_tomo(
                            out_dir, base_name, min(peet_bin, tomo_binning),
                            st, output_xf, output_tlt, thickness,
                            global_xtilt, SHIFT, xfile, localxf, zfac,
                            excludelist, defocus_file, V, Cs, ampC,
                            min(peet_apix, apix),
                            deftol = 200,
                            interp_w = 4, #this shouldn't be 4 everywhere...
                            n_tilts = ssorted_pcls.shape[0],
                            machines = machines)
    
    def make_link(src, dst):
        if os.path.islink(dst):
            os.unlink(dst)
        os.symlink(src, dst)
    
    if peet_bin >= tomo_binning:
        make_link(output_rec, default_full)
        make_link(out_tomo, default_tomo)
        make_link(output_ali, default_ali)

    #if the relative binning is not a multiple of 2 then the other binning is 
    #done completely from scratch
    if rel_bin%2 != 0. and peet_bin != tomo_binning:
        output_rec, out_tomo, output_ali = reconstruct_binned_tomo(
                            out_dir, base_name, max(peet_bin, tomo_binning),
                            st, output_xf, output_tlt, thickness,
                            global_xtilt, SHIFT, xfile, localxf, zfac,
                            excludelist, defocus_file, V, Cs, ampC,
                            max(peet_apix, apix),
                            deftol = 200,
                            interp_w = 4, #this shouldn't be 4 everywhere...
                            n_tilts = ssorted_pcls.shape[0],
                            machines = machines)
        if tomo_binning > peet_bin:
            make_link(output_rec, default_full)
            make_link(out_tomo, default_tomo)
            make_link(output_ali, default_ali)      
            
    #if the relative binning is a multiple of 2 then just bin 
    elif rel_bin%2 == 0.:
        new_bin_tomo = join(out_dir, base_name + '_bin%s.rec' % 
                            max(tomo_binning, peet_bin))
        run_generic_process(['binvol', '-an', '5', '-bin', rel_bin,
                             out_tomo, new_bin_tomo])
#        check_output('binvol -an 5 -bin %s %s %s' % (
#                        rel_bin, out_tomo, new_bin_tomo), shell = True)
        if tomo_binning > peet_bin:
            #but the stack and full tomo only need to be binned if they are
            #the higher binning (otherwise they were already softlinked)
            run_generic_process(['newstack', '-an', '6', '-bin',
                                 rel_bin, output_ali, default_ali])
#            check_output('newstack -an 6 -bin %s %s %s' % (
#                        rel_bin, output_ali, default_ali), shell = True)
            run_generic_process(['binvol', '-an', '5', '-bin',
                                 rel_bin, output_rec, default_full])
#            check_output('binvol -an 5 -bin %s %s %s' % (
#                        rel_bin, output_rec, default_full), shell = True) 
            os.symlink(new_bin_tomo, default_tomo)

    
    #run PEET
    #getting the base this way should be fine since I define the prm names
    #earlier anyway        
    prm_base1 = os.path.split(prm1)[1][:-4]
    prm_base2 = os.path.split(prm2)[1][:-4]
    os.chdir(fsc1d)
    run_generic_process(['prmParser', prm_base1 + '.prm'], 
                        join(fsc1d, 'parser.log'))
    #check_output('prmParser %s' % prm_base1, shell = True)
    if not isfile(prm_base1 + '-001.com'): 
        raise Exception('prmParser failed to generate com files. %s' % prm1)
    os.chdir(fsc2d)
    run_generic_process(['prmParser', prm_base2 + '.prm'], 
                        join(fsc1d, 'parser.log'))  
    if not isfile(prm_base2 + '-001.com'):
        raise Exception('prmParser failed to generate com files. %s' % prm2)

    #check_output('prmParser %s' % prm_base2, shell = True)
    print 'Running PEET...'
    run_split_peet(prm_base1, fsc1d, prm_base2, fsc2d, machines)
    #calcUnbiasedFSC
    os.chdir(peet_dir)
    fsc_log = join(peet_dir, 'calcUnbiasedFSC.log')
    print 'Running calcUnbiasedFSC...'
    run_generic_process(['calcUnbiasedFSC', prm1, prm2], out_log = fsc_log)
    #iters_done = 1 during first iteration
    peet_dirs = [split(out_dir)[0] + '/iteration_%s/peet/' % (ii + 1)
                 for ii in range(iters_done)]
    print 'peet dirs %s' % peet_dirs
    res = plot_fsc(peet_dirs, peet_dir, cutoff, peet_apix)
    os.chdir(cwd)
    return res

