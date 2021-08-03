#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:55:35 2019

@author: vojta
"""
import matplotlib
#this is required.  Even writing plots to file
matplotlib.use('Agg')


from flexo_processchunks import just_flexo
from model_from_peet_processchunks import flexo_model_from_peet  
import argparse
import numpy as np
from os.path import realpath, join#
import time
parser = argparse.ArgumentParser(description=("TBD"))
parser.add_argument('flexo_comfile', help = "")
parser.add_argument('--just_flexo', default = False,
                    help = "/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/test_new_toy_tomo/tmp_output/tmp_output_file.py")
parser.add_argument('--shifts_exist', default = False, action='store_true')
parser.add_argument('--restart_from_iter', default = 1,
                    help = 'Numbered from 1')
args = parser.parse_args()

exec(compile(open(args.flexo_comfile).read(), args.flexo_comfile, 'exec'))

#order to be deprecated
try:
    float(order)
    print('variable "order" is no longer used')
except:
    pass

shifts_exist = args.shifts_exist

if np.any(np.array(box)/2 - 1 < limit):
    print('Maximum shift cannot be >= box_size/2.\
    Defaulting to maximum allowed shift.')
    limit = np.array(box[0])/2 - 1
    
try:
    curr_iter = int(args.restart_from_iter)
except:
    raise ValueError('Input for restart_from_iter has to be an integer')
starting_out_dir = realpath(out_dir)
orig_rec_dir = rec_dir #unchanged during iterations
orig_tomo = False
#if use_init_ali:
#    non_overlapping_pcls = False 
if curr_iter > 1:
    #this is done specifically to get orig_tomo after restarting
    exec(compile(open(join(out_dir, 'iteration_%s' % str(curr_iter - 1),
                  'tmp_output/tmp_output_file.py')).read(), join(out_dir, 'iteration_%s' % str(curr_iter - 1),
                  'tmp_output/tmp_output_file.py'), 'exec'))

for iters_done in range(curr_iter, iters + 1):
    
    if not globalXYZ:
        warnings.warn('Enabling tiltalign FixXYZCoordinates')
        globalXYZ = True
    
    if iters > 1:
        out_dir = join(starting_out_dir, 'iteration_%s' % iters_done)
        if iters_done > 1:
            rec_dir = join(starting_out_dir, 'iteration_%s' %
                           str(iters_done - 1))
            if iters_done == iters and global_only:
                print('Enabling local alignment for the last iteration.')
                global_only = False
    # if not globalXYZ:
        #VP 16/6/2021 this needs more thought, for now, disable
    #     #if local alignments have been used to generate original tomogram then 
    #     #globalXYZ will always be turned off
    #     #they will be off if only one iteration is run, or if it's the final iteration
    #     #with multiple iterations it should be turned on for PEET model to match
    #     if iters == 1:
    #         globalXYZ = False
    #     elif iters_done == iters:
    #         globalXYZ = False
    #     else:
    #         globalXYZ = True
    if curr_iter == iters_done and args.just_flexo:
        startTime = time.time()
        exec(compile(open(args.just_flexo).read(), args.just_flexo, 'exec'))
    else:
        startTime = time.time()
        ffile = flexo_model_from_peet(rec_dir, out_dir, base_name, model_file,
                                  model_file_binning, csv, average_volume, 
                                  defocus_file,
                                  box, imaging_params, box,
                                  machines, grey_dilation_level, 
                                  non_overlapping_pcls, iters,
                                  iters_done, zero_tlt,
                                  average_volume_binning,
                                  lamella_model, use_init_ali,
                                  orthogonal_subtraction,
                                  d_orth, n_orth,
                                  noisy_ref = noisy_ref)

#        flexo_model_from_peet will finish by running processchunks, sync file 
#        will execute just_flexo

        exec(compile(open(ffile).read(), ffile, 'exec'))
    
    
    res, model_file, csv, prm, prm2 = just_flexo(
            #paths
            rec_dir, out_dir, base_name, defocus_file, tomo, 
            ali, tlt, xf, localxf, reprojected_mod, st, orig_rec_dir,
            sub_ali_list,
            plotback_ali_list,
            groups,
            ssorted_pcls, 
            #tomogram and running parameters
            tomo_size, apix, tomo_binning, output_binning, thickness, box,
            machines, pcls_per_core,
            global_xtilt, excludelist, axiszshift, 
            separate_group, zero_tlt, SHIFT, OFFSET,
            dose, n_patches, global_only, spec_tiny_size, 
            globalXYZ, fidn,
            #imaging params
            V, Cs, ampC, wl, ps,
            #flexo params
            butter_order = butter_order,
            dosesym = dosesym,
            orderL = orderL,
            pre_exposure = pre_exposure,
            limit = limit,
            interp = interp,
            smooth = smooth,
            centre_bias = centre_bias,
            thickness_scaling = thickness_scaling,
            debug = debug,
            iters_done = iters_done,
            allow_large_jumps = allow_large_jumps,
            orthogonal_subtraction = orthogonal_subtraction,
            shifts_exist = shifts_exist,
            tom_n = tom_n,
            prm = prm,
            prm2 = prm2,
            ite = ite,
            cutoff = cutoff,
            search_rad = search_rad,
            phimax_step = phimax_step,
            psimax_step = psimax_step,
            thetamax_step = thetamax_step,
            no_ctf_convolution = no_ctf_convolution,
            RotOption = RotOption,
            TiltOption = TiltOption,
            MagOption = MagOption,
            poly = poly)
    
    print('Iteration execution time: %s s.' % int(np.round(
                                (time.time() - startTime ), decimals = 0)))    
    #iters_done += 1
    print('Iterations completed %s' % iters_done)
    shifts_exist = False
    
    if iters_done == iters:
        print('Done.')
    elif iters_done > 1:
#        nfreq = np.round(res[:,2], decimals = 3)
        if isinstance(res, bool):
            print('Continuing without FSC check.')
        else:
            r = res[:,1]
            if np.any(np.floor(r) == np.floor(apix*2)) or get_area:
                #in case resolution = resolution at Nyquist, use area under FSC
                areas = np.round(res[:,2], decimals = 3)
                if not get_area:
                    print(('Estimated resolution too close to Nyquist. ' + 
                           'Using area under FSC instead.'))
                print(('Areas under FSC of initial PEET, iterations 1-%s: %s' % 
                   (iters_done, (',').join([str(x) for x in areas]))))
                if areas[-1] <= areas[-2]:
                    print(('No apparent improvement since last iteration.'
                           + '  Exiting...'))
                    break
            else:
                print(('PEET resolution estimates for iterations 1-%s: %s' % 
                       (iters_done, (',').join([str(x) for x in r]))))
                if r[-1] >= r[-2]:
                    print(('No apparent improvement since last iteration.'
                           + '  Exiting...'))
                    break

        
    
    
"""
format_nonoverlapping_alis has machines as input but doesn't use it...
/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/nec_test17/iteration_1/example_particles particle 6 DOESNT EXIST AND I DON'T KNOW WHY.  it might have been removed due to non-overlapping group filter? verifY!!!!!!!!!!1
double check that the convention for centres of peaks etc is the same across (e.g. g2d)
write print statements to log...     
plot of ccs/tilt - in the model I could include electron damage
minimising tiltalign residuals together with maximising distance from the currend model could be a good way to score things quickly without averaging
why is there an offset between the saved bin_[3d model].mod and sorted_pcls?
peet should first run on the original data...
double check that zero_tlt is numbered from 0 in flexo_proceschunks
several functions create np.meshgrid...this could be a single function
   importantly, dist_from_centre_map is symmetrical around the centre, meshgrid isnt
there shouldn't be a crazy outlier cluster of particles
                
cc_valus have 0 entries is a problem... np.ma.masked_less_equal() doesn't work amazingly well because most functions ignore mask....
    commented out self.cc_values masking in read_cc_peaks
    get_shift_magnitude needs to deal with 0 entries somehow
    
the initial tilt_subset should include tilts either side of zero tlt to make sure a potential big jump at this point doesn't get removed


after running a clasifier, maybe check if the shift_std of the low_std group of particles went up - this would indicate that the classifier failed on dist_matrix data

probability contour values are not scaled properly after gmm classification

gpr min/max scale probably needs to be tweaked based on averaging results... but it should reflect the density of points somehow
    ....there could be an automatic optimiser using averaging results for scoring........
"""              
            
            
            
 
    
### what files are used for what and defined where:
#tomo, (defined in verify_inputs): join(rec_dir, base_name + '.rec'):
#    verify_inputs: tomo_apix, tomo_binning (with stack_apix)
#full_tomo, (defined in verify_inputs)
#st fom newst.com (read in verify_inputs) 
#    verify_inputs: stack_apix
#xf (read in verify_inputs) from newst.com 
#ali (read in verify_inputs) from tilt.com
#tlt (read in verify_inputs) from tilt.com
#localxf (read in verify_inputs) from tilt.com
#zfac (read in verify_inputs) from tilt.com
#xfile (read in verify_inputs) from tilt.com (xaxistilt)    
#defocus file (from input file or defined in verify_inputs): 
#   realpath(join(rec_dir, str(base_name) + '.defocus'))
#   i think this is then overrwriten later
#model_file - first through input file, then renamed to 
#   base_name + _peet.mod at each iter.
#lamella_mask_path (defined in model_from_peet_processchunks)    
#    join(out_dir, 'lamella_mask.mrc')
#reprojected_tomogram_path (defined in model_from_peet_processchunks)  
#   = join(out_dir, base_name + '_reprojected.ali')
#full_tomo (defined in model_from_peet_processchunks)  
#   = join(rec_dir, base_name + '_full.rec')
#out_mask (defined in model_from_peet_processchunks)  
#   = join(out_dir, 'binary_mask.mrc')
#masked_tomo (defined in model_from_peet_processchunks)  
#   = join(out_dir, 'masked_%s.mrc') % base_name
#mask_path   (defined in model_from_peet_processchunks by NEW_mask_from_plotback)  
#tmp_output_folder (defined in model_from_peet_processchunks)  
#    = join(out_dir, 'tmp_output')
#tmp_output_file (defined in model_from_peet_processchunks)  
#    = join(tmp_output_folder, 'tmp_output_file.py')
    
    
    
##example imput file
#    ######## Input files ########
## Original tomogram reconstruction directory
#rec_dir = '/raid/fsj/grunewald/vojta/ribosome_flexo/11/point_tomo/shifted'
## Output directory for Fleo
#out_dir = '/raid/fsj/grunewald/vojta/ribosome_flexo/11/point_tomo/iters_flexo_shifted4_forcing_model_on_origin'
## Boundary model style model marking the desired sub-volume
#lamella_model = False#'/raid/fsj/grunewald/vojta/nec/I12a/tighter_tomopitch.mod'
## Specify non-default defocus file (normally base_name.defocus).  Use False for default file.
#defocus_file = False
#
## Model file marking position of particles to be used for alignment.  Typically output of PEET alignment.
#model_file = '/raid/fsj/grunewald/vojta/ribosome_flexo/11/point_tomo/3d_tomo.mod'
## Binning of PEET model file relative to current volume. E.g. if PEET was run with volume binned 2x and the volume for Flexo alignment is binned 4x, model_file_binning = 0.5
#model_file_binning = 1
#
## Volume to be used as reference for particle alignment.  White on black. Typically chimera segmented.
#average_volume = '/raid/fsj/grunewald/vojta/ribosome_flexo/11/point_tomo/sfs3.mrc'
## Binning of reference volume relative current volume.
#average_volume_binning = 1 #possibly just detect...
#
## CSV file corresponding to the specified model file
#csv = '/raid/fsj/grunewald/vojta/ribosome_flexo/11/point_tomo/csv_offset.csv'
#
## PEET parameter file(s) for FSC determination.  If one is specified, it will be split. 
#prm = '/raid/fsj/grunewald/vojta/nec/I12a/peet_capsids/run1/bin6/run3_global_new_tomo/pentons_i4/run1/remdup1_6_cc15/c_fromIter7_remdup1.0.prm'
#prm2 = False
#tom_n = 1 #from one
#ite = 0
# PEET search parameters: Translation search radius, phi/psi/theta max and step
#search_rad = False # [int or tuple of 3 ints] use False for defaults
#phimax_step = False # [tuple of 2 ints] use False for defaults
#psimax_step = False # [tuple of 2 ints] use False for defaults
#thetamax_step = False # [tuple of 2 ints] use False for defaults
#get_area = True
#
######### Output parameters ########
#
## Base name for output files
#base_name = 'p'
## Desired binning of final output volume
#output_binning = 1
#
######### Data acquisition parameters ########
## Tilt series acquisition parameters.  Accelerating voltage [kV], spherical aberration [mm], Fraction of absorbtion contrast 
#imaging_params = 300,2.7,0.07
##Phase shift
#ps = 0
## First tilt of the series (i.e. dose = 0) numbered from 1. Use False to find tilt closest to 0 degree microscope tilt.  
#zero_tlt = False 
## Was data collected using dose-symmetric (Hagen) scheme?
#dosesym = False
## When inspecting tilt series in 3dmod.  Was data collected from the first tilt to the left or right?
#orderL = True #if True, tilts are filtered from zerotlt to 0, then zerotlt to max
## Pre-exposure
#pre_exposure = 0 #100e for ~20 A cutoff, 200e ~ 30 A
## Dose per tilt (E-/A^2)
#dose = 0
#
######### Alignment parameters ########
## Box size for particle extraction.
#box = [26,26]
## Maximum shift allowed
#limit = 10
## Interpolation for sub-pixel accuracy
#interp = 4
##  Dilation used for mask generation
#grey_dilation_level = 0
## Orthogonal subtraction
#orthogonal_subtraction = False
#d_orth = 60
#n_orth = 6
## Avoid particle overlap during cross correlation
#non_overlapping_pcls = True
## Use original projectons for cross correlation.
#use_init_ali = False
## Additional binning for tomogram positioning. Uses tomograms binned to [spec_tiny_size]*[current binning].  Combined binning of up to 20x works.
#spec_tiny_size = 2 
## Higher values bias more towards smaller shifts
#centre_bias = 0.01
## Thickness scaling.  Favour smaller shifts as relative thickness increases. Larger values bias towards smaller shifts
#thickness_scaling = 0.5
## Allow sudden jumps in contours
#allow_large_jumps = False


##tiltalign parameters
## Number of patches for IMOD local alignment
#n_patches = 2,2
## Number of fiducials on each surface required for each patch
#fidn = 14,8
## Disable local alignment
#global_only = True
## Use global XYZ coordinates for local alignment. Only set to False if local alignments were used to generate input tomogram 
#globalXYZ = False #do not turn this on, it's possible it should ALWAYS be off...but who knows
#
#RotOption = 1 #3 for groupping
#TiltOption = 2 #5 for groupping
#MagOption =  1 #3 for groupping


######### DEV/to be deprecated ########
## Apply smoothing to fiducial contours
#smooth = True #smooth shifts, currently doesn't do anything
#unreasonably_harsh_filter = False
#no_ctf_convolution = False
#poly = False #use polynomial fit to shifts instead of detected
#noisy_ref = False #add reprojected tomo to plotback and use as reference
#
######### Running parameters ########
## Machines for parallel processing.  Use machines = False to use localhost.  Use ['machine1']*2 + ['machine2']*2 to use 2 cores on machine1 and machine2
#machines = ['citra']*15 + ['darklord']*3 + ['rasputin2']*10
## Number of particles sent to each core for cross correlation
#pcls_per_core = 20
## Number of iterations to run
#iters = 2
## Order of Butterworth filter
#butter_order = 4
## Apply dose filter
#dosefilter = False
## Debug 0 - pretty much nothing, #1 print output (still unfinished, need to setup output error file) #2 make figure (very slow!!!)
#debug = 0 



