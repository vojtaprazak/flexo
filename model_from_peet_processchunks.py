#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:57:10 2019

@author: vojta
"""


#from definite_functions_for_flexo import *
from shutil import copyfile
import os
import numpy as np
from os.path import join, isfile, split, realpath
from definite_functions_for_flexo import (verify_inputs, run_generic_process,
                                          lame_mask, rotate_model,
                                          make_and_reproject_plotback,
                                          NEW_mask_from_plotback,
                                          reproject_volume,
                                          format_nonoverlapping_alis,
                                          replace_pcles,
                                          mult_mask,
                                          rec_orth_tomos,
                                          run_processchunks)
from MapParser_f32_new import MapParser
from PEETMotiveList import PEETMotiveList
#tomo10
#execfile('/raid/fsj/grunewald/vojta/actin_flexo/tomo10_flexo/input_for_flexo_model_form_peet.py')

#synthetic data set
#execfile('/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/test/input_for_flexo_model_form_peet.py')
#model shifted by 0,-1.6,-1.6 
#model_file = '/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/test/bin2_3d_model_shift0_1-6_1-6.mod'

def flexo_model_from_peet(rec_dir, out_dir, base_name, model_file,
                          model_file_binning, csv, average_volume, defocus_file,
                          box, imaging_params,
                          machines, grey_dilation_level = 5,
                          non_overlapping_pcls = False, iters = 1,
                          iters_done = 1, zero_tlt = 0,
                          average_volume_binning = 1, lamella_model = False,
                          use_init_ali = False, orthogonal_subtraction = False,
                          d_orth = 60, n_orth = 6,
                          threshold = 0.01):
    
    """
    Inputs:
        rec_dir [str] path
        out_dir [str] path
        base_name [str] path
        model_file [str] path to peet .mod
        model_file_binning [int] binning relative to tomogram (e.g. 0.5 if model is bigger)
        csv [str] path to peet .csv
        average_volume [str] path to average volume in .mrc format, white on black
        defocus_file [str] path to defocus file (both imod and ctffind supported???)
        box [list or tuple of 2 ints] don't try rectangle...
        imaging_params [tuple or list of 3 ints/floats] eg 300,2.7,0.07

       # num_cores [integer] how many cores should be used for parallel processes
        non_overlapping_pcls [bool] use masktomrec to make a set of non-overlaping models/alis
        grey_dilation_level [int] used for mask generation for masktomrec
        reproject_tomogram [bool] used when mse_mtr == False.  Use reprojected tomogram instead of the real aligned stack
        
        
        zero_tlt [int] ordinal of the lowest dose image (from 1 ). If == 0, zero_tlt will be set to the ordinal of 0 degree tilt from .rawtlt
        threshold [float] fraction of "leftover" particles that don't fit into
            any group
        ...
    """
    V, Cs, ampC = imaging_params
    V = V * 1E3
    Cs = Cs * 1E7
    wl = (12.2639/(V+((0.00000097845)*(V**2)))**0.5)


    print 'Checking input files...'
    #VP 19/10/2020 out_tomo should probably not be defined and pas
    #out_tomo = os.path.abspath(join(out_dir, base_name + '.rec'))  
    
    (st, tomo, full_tomo, tomo_size, ali, tlt, apix, tomo_binning,
     defocus_file, thickness, global_xtilt, excludelist, axiszshift,
     separate_group, zero_tlt, xf, SHIFT, OFFSET, localxf, zfac, xfile,
     ) = verify_inputs(rec_dir, 
                        base_name,
                        out_dir,
                        defocus_file,
                        input_binning = False,
                        zero_tlt = zero_tlt)


    
    if iters > 1:
        #force model coordinates on input tomogram.  This is done to compensate
        #for origin offset during iterations
        nmodel_file = join(out_dir, base_name + '_peet.mod')
#        check_output('imodtrans -I %s %s %s'
#                     % (tomo, model_file, nmodel_file), shell = True)
        run_generic_process(['imodtrans', '-I', tomo, model_file, nmodel_file])
        if isfile(nmodel_file):
            model_file = nmodel_file
#        VP 19/10/2020 defocus_file is already defined this way in verify_inputs
#        if iters_done > 1 and excludelist:    
#            defocus_file = join(rec_dir, base_name + '.defocus')
        print 'Starting iteration %s of %s.' % (iters_done, iters)
#        VP 19/10/2020 st path is read from newst.com and shouldn't be defined here
#        st = realpath(join(rec_dir, base_name + '.st'))
        new_defocus_file = join(realpath(out_dir), split(defocus_file)[1])
        
#        VP 19/10/2020 this shouldn't be needed!!
#        new_st = realpath(join(out_dir, base_name + '.st'))
#        if not isfile(new_st):
#            os.symlink(st, new_st)
        if not isfile(new_defocus_file):
            os.symlink(defocus_file, new_defocus_file)




#    if iters == 1:
#        #not sure how this will work with iterations..
#        #might not need to preserve these at all...?
#        orig_tlt = tlt
#        orig_ali = ali
#        starting_xf = xf

    #the tilt command for reprojection needs to match input for 
    #tilt.com. Check if variables exist, then append to base cmd
    var_str = np.array([
            ' -XAXISTILT %s' % global_xtilt,
            ' -XTILTFILE %s' % xfile,
            ' -LOCALFILE %s' % localxf,
            ' -OFFSET %s' % OFFSET,
            ' -SHIFT %s %s' % (SHIFT[0], SHIFT[1]),
            ' -ZFACTORFILE %s' % zfac
            ])
    var_mask = np.zeros(len(var_str), dtype = 'bool')
    add_var =  (global_xtilt, xfile, localxf, OFFSET,
              [SHIFT[0], SHIFT[1]], zfac)
    for y in range(len(add_var)):
        if add_var[y]:
            var_mask[y] = True
    if np.any(var_mask):
        var_str = var_str[var_mask]
    
    print 'var_str = %s' % var_str
    
    if lamella_model:
        print 'Generating lamella mask.'
        lamella_mask_path = join(out_dir, 'lamella_mask.mrc')
        lame_mask(lamella_model, tomo_size, lamella_mask_path)
    else:
        lamella_mask_path = False    
        
    reprojected_tomogram_path = join(
            out_dir, base_name + '_reprojected.ali')
    full_tomo = join(rec_dir, base_name + '_full.rec')
    out_mask = join(out_dir, 'binary_mask.mrc')
    masked_tomo = join(out_dir, 'masked_%s.mrc') % base_name

    #sorted_pcls, rawtiltmod is likely to become obsolete
    #removed VP 7/2/20
#    sorted_pcls, rawtiltmod = backtransform_model(model_file, base_name, 
#                    rec_dir, out_dir, tomo, ali, model_file_binning) 

    #read peet model    
    print 'Processing particle models.'
    rsorted_pcls, reprojected_mod = rotate_model(tomo, full_tomo,
            ali, base_name, model_file, csv, rec_dir, out_dir, tlt, tomo_size,
            model_file_binning, False, var_str)     
#    excludelist_mask = np.isin(
#                range(rsorted_pcls.shape[0]), excludelist, invert = True)    


    print 'Generating reference tilt series.'
    if not non_overlapping_pcls:  
    #fast, simple approach:
    #Makes plotback using all pcls and reproject it into a TS. 
        #make and reproject plotback
        plotback_path, plotback_ali_path = make_and_reproject_plotback(
                    base_name, out_dir, average_volume, tomo_size, csv,
                    model_file, apix, model_file_binning, tomo_binning,
                    tlt, ali, average_volume_binning, False, False, var_str,
                    lamella_mask_path)              
                        



        reprojected_tomogram_path = join(
                out_dir, base_name + '_reprojected.ali')
        full_tomo = join(rec_dir, base_name + '_full.rec')
        out_mask = join(out_dir, 'binary_mask.mrc')
        masked_tomo = join(out_dir, 'masked_%s.mrc') % base_name


        if use_init_ali:
            reprojected_tomogram_path = join(
                out_dir, base_name + '_initial.ali')         
            
            copyfile(ali, reprojected_tomogram_path)
        else:
            #generate mask from plotback
            mask_path = NEW_mask_from_plotback(plotback_path,
                            out_mask, grey_dilation_level,
                            lamella_model, lamella_mask_path)
            #remove lamella_model from inputs, and multiply with existing
            #lamella_mask instead?
            
            
            print 'should include the option to run mtr using only lamella mask...'
            #apply mask and reproject
#            check_output('clip multiply %s %s %s' % 
#                (mask_path, full_tomo, masked_tomo), shell = True)
            run_generic_process(['clip', 'multiply',
                                 mask_path, full_tomo, masked_tomo])
            reproject_volume(masked_tomo, ali, tlt, tomo_size[2],
                         reprojected_tomogram_path, var_str) 

        #unify outputs with non_overlapping_pcls
        ########
        ssorted_path = join(out_dir, base_name + '_ssorted.npy')
        group_path = join(out_dir, base_name + '_groups.npy')
        sub_ali_list = [reprojected_tomogram_path]
        plotback_ali_list = [plotback_ali_path]

        groups = np.ones(rsorted_pcls.shape[1], dtype = bool)
        ssorted_pcls = np.dstack((rsorted_pcls,
                                  np.zeros((rsorted_pcls.shape[0],
                                            rsorted_pcls.shape[1], 2))))  
        ssorted_pcls[:,:,3] = range(rsorted_pcls.shape[1])

        np.save(group_path, groups)
        np.save(ssorted_path, ssorted_pcls)
        ########
      
    else:
        #makes non-overlapping plotback and reprojects TS out of it.
        #Also makes the equivalent tilt series where overlapping particles are 
        #subtracted.  
        
        (group_path, sub_ali_list, plotback_ali_list, ssorted_path, 
         plotback_list, out_mask_list, smooth_mask_list
         ) = format_nonoverlapping_alis(
                out_dir, base_name, average_volume, full_tomo,
                ali, tlt, model_file, csv, apix, rsorted_pcls,
                tomo_size, var_str, machines, model_file_binning,
                grey_dilation_level,
                average_volume_binning,
                lamella_mask_path,
                threshold = threshold)
        
        if orthogonal_subtraction:
            
            g0 = np.load(group_path)
            average_map = MapParser.readMRC(average_volume) 
            csv_file = PEETMotiveList(csv)

            for group_ids in range(len(g0)):
                print ('Commencing orthogonal subtraction %s of %s.' 
                       % (group_ids + 1, len(g0)))
                groups = g0[group_ids].squeeze()
                #generate plotback
                replace_pcles(average_map, tomo_size, csv_file, model_file,
                              plotback_list[group_ids], apix, groups, 
                              model_file_binning, 
                              average_volume_binning, True)

                reproject_volume(plotback_list[group_ids], ali, tlt,
                                 tomo_size[2], plotback_ali_list[group_ids],
                                 var_str)    

                #generate mask
                NEW_mask_from_plotback(plotback_list[group_ids],
                    out_mask_list[group_ids], grey_dilation_level, False,
                    False, smooth_mask_list[group_ids])

                if lamella_mask_path:
                    #combined masks with an existing lamella mask
                    mult_mask(lamella_mask_path,
                              smooth_mask_list[group_ids])

                rec_orth_tomos(
                                base_name,
                                out_dir,
                                tlt,
                                ali,
                                tomo_binning,
                                thickness,
                                smooth_mask_list[group_ids],
                                machines,
                                nvlp_id = group_ids,
                                global_xtilt = global_xtilt,
                                OFFSET = OFFSET,
                                SHIFT = SHIFT,
                                xfile = xfile,
                                localxf = localxf,
                                zfac = zfac,
                                excludelist = excludelist,
                                fakesirt = 0,
                                n_orth = n_orth,
                                d_orth = d_orth,
                                tmpfs = True)
        else:    
            run_processchunks(base_name, out_dir, machines)

#to be removed: check that variables are consistent in just_flexo
#        #remove excludelist entriest from plotback
#        #the full-length files are needed for recnstruction in just_flexo
#        #while the truncated ones are used for cc etc
#        full_defocus_file = defocus_file
#        full_xf = xf
#        if excludelist:# and iters_done == 1:
#
#            #fix these separately before looping
#            _, tlt, defocus_file, xf = deal_with_excludelist(
#                    False, tlt, defocus_file, xf, excludelist,
#                    base_name, rec_dir, out_dir)             
#            for a in range(len(plotback_ali_list)):
#                #remove entries from rsorted_pcls
#                novlp_sorted_pcls[a] = novlp_sorted_pcls[a][excludelist_mask]
#
#                #remove excludelist entriest from plotback
#                plotback_ali_list[a], _, _, _ = deal_with_excludelist(
#                        plotback_ali_list[a], False, False, False,
#                        excludelist, base_name, rec_dir, out_dir) 
#                #and from reprojected tomo.  
#                sub_ali_list[a], _, _, _ = deal_with_excludelist(
#                        sub_ali_list[a], False, False, False,
#                        excludelist, base_name, rec_dir, out_dir) 
#
#    full_ali = ali
#    if excludelist:# and iters_done == 1:
#        ali, _, _, _ = deal_with_excludelist(ali, False, False, False,
#                                excludelist, base_name, rec_dir, out_dir) 
#        rsorted_pcls = rsorted_pcls[excludelist_mask]     
#        #sorted_pcls = sorted_pcls[excludelist_mask]     

    tmp_output_folder = join(out_dir, 'tmp_output')
    tmp_output_file = join(tmp_output_folder, 'tmp_output_file.py')
    if not os.path.isdir(tmp_output_folder):
        os.makedirs(tmp_output_folder)
    np.save(join(tmp_output_folder, 'rsorted_pcls.npy'), rsorted_pcls)


#    if non_overlapping_pcls:
#        np.save(join(tmp_output_folder, 'groups.npy'), groups)
#        for x in range(len(novlp_sorted_pcls)):
#            np.save(join(tmp_output_folder, 'novlp_pcls_%02d.npy' % x), \
#                    novlp_sorted_pcls[x])




    with open(tmp_output_file, 'w') as f:
        f.write("rec_dir = '%s'\n" % rec_dir)
        f.write("out_dir = '%s'\n" % out_dir)
        f.write("base_name = '%s'\n" % base_name)
        f.write("defocus_file = '%s'\n" % defocus_file)
        #f.write("out_tomo = '%s'\n" % out_tomo)
        f.write("tomo = '%s'\n" % tomo)
        f.write("ali = '%s'\n" % ali)
        f.write("tlt = '%s'\n" % tlt)
        f.write("xf = '%s'\n" % xf)
        if localxf:
            f.write("localxf = '%s'\n" % localxf)
        else:
            f.write("localxf = False\n")
        f.write("reprojected_mod = '%s'\n" % reprojected_mod)
        f.write("st = '%s'\n" % st)
        f.write("sub_ali_list = [%s]\n" % 
                (',').join(['"' + str(y) + '"' for y in sub_ali_list]))
        f.write("plotback_ali_list = [%s]\n" % 
            (',').join(['"' + str(y) + '"' for y in plotback_ali_list]))
        f.write("groups = '%s'\n" % group_path)       
        f.write("ssorted_pcls = '%s'\n" % ssorted_path)            
        #orig_rec_dir specified outside model_from_peet
        f.write('tomo_size = %s\n' % [x for x in tomo_size])
        f.write('apix = %s\n' % apix)
        f.write('tomo_binning = %s\n' % tomo_binning)
        #output_binning specified outside model_from_peet
        f.write('thickness = %s\n' % thickness)
        f.write('box = %s\n' % box)
        f.write('machines = [%s]\n' % (',').join(
                ["'" + x + "'" for x in machines]))
        #pcls_per_core specified outside model_from_peet
        f.write('global_xtilt = %s\n' % global_xtilt)
        f.write('excludelist = %s\n' % excludelist)
        f.write('axiszshift = %s\n' % axiszshift)        
        f.write("separate_group = '%s'\n" % separate_group)
        f.write('zero_tlt = %s\n' % zero_tlt)
        f.write('SHIFT = %s\n' % [x for x in SHIFT])
        f.write('OFFSET = %s\n' % OFFSET)
        #dose specified outside model_from_peet
        #n_patches
        #global_only
        #spec_tiny_size
        #globalXYZ
        #fidn
        f.write('V = %s\n' % V)
        f.write('Cs = %s\n' % Cs)
        f.write('ampC = %s\n' % ampC)
        f.write('wl = %s\n' % wl)
        #ps

    print 'Flexo parameter file created: %s' % tmp_output_file
    

    print 'done'
    
    return tmp_output_file
