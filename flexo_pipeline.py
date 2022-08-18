# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:58:27 2022

@author: vojta
"""
import os
import sys
import time
from shutil import copyfile
from copy import deepcopy
from os.path import join, realpath, split, isfile, isdir
from subprocess import check_output
import numpy as np
from flexo_tools import (get_apix_and_size, find_nearest, make_lamella_mask,
                         get_mod_motl_and_tomo, bin_model,
                         make_non_overlapping_pcl_models,
                         fid2ndarray, optimal_lowpass_list)
from flexo_tools import replace_pcles, reproject_volume, mask_from_plotback, match_tomos, get2d_mask_from_plotback, tomo_subtraction
from flexo_tools import (machines_from_imod_calib, write_mrc, prepare_prm,
                         peet_halfmaps, ctf_convolve_andor_dosefilter_wrapper,
                         extract_2d, ncc, optimal_lowpass, combine_fsc_halves, plot_fsc,
                         get_peaks)
#from PEETModelParser import PEETmodel
from PEETPRMParser import PEETPRMFile 
#from scipy.spatial import KDTree
from IMOD_comfile import IMOD_comfile
import json
import multiprocessing
import socket
import matplotlib.pyplot as plt
import mrcfile
import warnings 


from extracted_particles import Extracted_particles
from definite_functions_for_flexo import (run_generic_process, run_processchunks,
        check_ssh, imodscript, get_binned_size)
#from PEETModelParser import PEETmodel

class Flexo:
    
    def __init__(self, rec_dir = False, out_dir = False, json_attr = False,
                 #imaging params
                 ps = 0, pre_exposure = 0,
                 #running params
                 mask_tomogram = True, noisy_ref = False, phaseflip = True,
                 use_existing_fid = False, add_fiducial_model = [],
                 machines = 1, limit = 10, interp = 10, 
                 allow_large_jumps = False,
                 non_overlapping_pcls = True,
                 use_init_ali = False, spec_tiny_size = 2,
                 pcls_per_core = 40, butter_order = 4, dosefilter = False,
                 debug = 2,
                 n_peaks = 5, #5 seems to work quite well
                 #iteration
                 num_iterations = 2, iters_done = 0,
                 #tomogram
                 tomo_binning = False, XTiltOption = 3, XTiltDefaultGrouping = 10,
                 LocalXTiltOption = 3, LocalXTiltDefaultGrouping = 6,
                 SurfacesToAnalyze = 1,
                 fidn = (40, 0), global_only = False,
                 #PEET
                 ite = 999, cutoff = 0.143, phimax_step = [0,1],
                 psimax_step = [0,1], thetamax_step = [0,1], get_area = True,
                 #dev
                 xcor_normalisation = 'none',
                 padding = 10,
                 keep_peet_binning = True,
                 use_refined_halfmap_models = True,
                 noisy_ref_std = 5,
                 apply_2d_mask = True,
                 exclude_worst_pcl_fraction = 0.1,
                 exclude_lowest_cc_fraction = False,
                 min_neighbours = 7,
                 use_local_median_shifts = False,
                 use_median_cc_maps = False,
                 shift_std_cutoff = 3,
                 use_nbr_median = False,
                 use_davens_fsc = False,
                 **kwargs):
        
        """
        Final checklist:
            switch to simple fsc
        """
        
        print('DEV NOTE use_existing_fid is False for toy data')
        #plt.ioff()        
        
        if json_attr:
            with open(json_attr, 'r') as f:
                json_dict = json.load(f)
            for key in json_dict:
                if isinstance(json_dict[key], str) and json_dict[key].endswith('.npy'):
                        setattr(self, key, np.load(json_dict[key]))   
                else:
                    setattr(self, key, json_dict[key])             
            
        else:
            #imaging params
            self.V = kwargs.get('V')
            self.Cs = kwargs.get('Cs')
            self.ampC = kwargs.get('ampC')
            self.ps = ps
            #self.wl = 
            self.apix = kwargs.get('apix')
            self.st_apix = kwargs.get('st_apix')
            self.order = kwargs.get('order')
            self.orderfile_type = kwargs.get('orderfile_type')
            self.mdoc = kwargs.get('mdoc')
            self.flux = kwargs.get('flux')
            self.tilt_dose = kwargs.get('tilt_dose')
            self.pre_exposure = pre_exposure
            self.lowfreqs = kwargs.get('lowfreqs') #determiend from mdoc or order+tilt_dose
            self.phaseflip = phaseflip #False if tilt series wasn't phaseflipped
            
            #paths and names
            self.rec_dir = rec_dir 
            self.orig_out_dir = out_dir #keep the same throughout iterations
            self.out_dir = join(out_dir, 'iteration_1') 
            self.update_dirs()
            # self.out_json = join(self.out_dir, 'flexo.json')
            # self.npy_dir = join(self.out_dir, 'npy')
            # self.pcle_dir = join(self.out_dir, 'extracted_particles')
            # self.xcor_peak_dir = join(self.out_dir, 'xcor_peaks')
            # self.match_tomos_dir = join(self.out_dir, 'match_tomos')
            # self.init_peet_dir = join(self.out_dir, 'init_peet')
            # self.peet_dir = join(self.out_dir, 'peet')
            
            #models
            self.model_file = kwargs.get('model_file') #initial 3d model file
            self.reprojected_mod = kwargs.get('reprojected_mod') #3d model projected to 2d fiducial model
            self.fid_list = [] #split reprojected_mod
            self.full_model_file = kwargs.get('full_model_file') #generated during model reprojection, fits unrotated full.rec
            self.split3d_models = [] #3d pcl model after non_overlapping_pcles, in rotated orientation
            self.out_fid = [] #new fiducial model with calculated shifts
            #motive lists
            self.motl = kwargs.get('motl') #initial motive list, matching model_File
            self.split_motls = [] #split motl, matching split3d_models
            #intermediate image or volume files
            self.query2d = kwargs.get('reprojected_tomo') #whatever becomes the query for cc
            self.plotback2d = kwargs.get('plotback2d') #whatever becomes the reference for cc

            
            #masking
            self.lamella_mask_path = kwargs.get('lamella_mask_path')
            self.lamella_model = kwargs.get('lamella_model')
            self.mask_tomogram = mask_tomogram #generate from plotback and mask initial tomogram before reprojection
            self.apply_2d_mask = apply_2d_mask #applied all the time, but should have no effect when noisy_ref  = False, or use_init_ali = False


            self.base_name = kwargs.get('base_name')
            self.imod_base_name = None #degermined from aligncom
            #self.namingstyle = namingstyle #leaving this to IMOD_comfile
            self.tomo = kwargs.get('tomo')
            self.full_tomo = kwargs.get('full_tomo')
            self.out_tomo = kwargs.get('out_tomo')
            self.peet_tomo = kwargs.get('peet_tomo')
            self.st = kwargs.get('st')
            self.ali = kwargs.get('ali')
            self.tlt = kwargs.get('tlt')
            self.defocus_file = kwargs.get('defocus_file')
            self.xf = kwargs.get('xf')
            #self.localxf = kwargs.get('localxf')

            self.excluded_particles = []

            #sorting particles, np arrays
            self.groups = kwargs.get('groups')
            self.ssorted_pcls = kwargs.get('ssorted_pcls')
            #0=xcoords, 1=ycoords, 2=tilt number, 3=particle index, 4=group id (from 0), 5=tilt angle, 6=defocus)]
            
            
            #running parameters
            self.path = (',').join(['"' + x + '"' for x in sys.path])
            self.machines = machines #socket.gethostname() - make int or list. by default runs on 1 local core. If there isn't a cpu.adoc then 1 local core used
            self.pcls_per_core = pcls_per_core
            self.box_size = kwargs.get('box_size') #non square boxed don't make sense here
            self.limit = limit
            self.interp = interp
            self.n_peaks = n_peaks
            
            self.particle_interaction_distance = kwargs.get('particle_interaction_distance')
            
            self.use_existing_fid = use_existing_fid
            self.add_fiducial_model = add_fiducial_model
            self.non_overlapping_pcls = non_overlapping_pcls
            self.noisy_ref = noisy_ref
            
            self.allow_large_jumps =  allow_large_jumps
            self.dilation_size = 0
            
            self.use_init_ali = use_init_ali
            self.spec_tiny_size = spec_tiny_size
            #self.orthogonal_subtraction = orthogonal_subtraction
            #self.d_orth = d_orth
            #self.n_orth = n_orth
            self.butter_order = butter_order
            #self.dose = kwargs.get('dose') #electrons/angstrom/tilt 
            #self.dosefilter = dosefilter
            self.debug = debug
            self.plot_pcls = kwargs.get('plot_pcls') #write out plots for these particles 
            
            
            #iteration control, restarting
            self.num_iterations = num_iterations
            self.curr_iter = kwargs.get('curr_iter') #numbered from 1?
            self.fsc_dirs = [] 
            self.res = []   #list of ndarrays: (sampling frequency at cutoff, resolution, area under FSC, last bin for area calculation)
            
            
            #when restarting, each needs to be separately set to False if the stage is to be re-run
            self.flg_inputs_verified = kwargs.get('flg_inputs_verified')
            self.flg_image_data_exist = kwargs.get('flg_image_data_exist')
            self.flg_shifts_exist = kwargs.get('flg_shifts_exist')
            #self.flg_aligncom_formatted = kwargs.get('flg_aligncom_formatted')
            self.flg_tomos_exist = kwargs.get('flg_tomos_exist')
            self.flg_peet_ran = kwargs.get('flg_peet_ran')
            


            #tomogram parameters
            self.tomo_binning = kwargs.get('tomo_binning')
            self.peet_binning = kwargs.get('peet_binning')
            self.peet_apix = kwargs.get('peet_apix')
            self.thickness = kwargs.get('thickness')
            self.tomo_size = kwargs.get('tomo_size')
            self.st_size = kwargs.get('st_size')
            #self.global_xtilt = kwargs.get('global_xtilt')
            self.excludelist = []
            #self.axiszshift = kwargs.get('axiszshift')
            #self.separate_groups = kwargs.get('separate_groups')
            #self.zero_tlt = zero_tlt # no longer used
            #self.tilt_angles = kwargs.get('tilt_angles') #these are no longer used for anything
            #self.SHIFT = kwargs.get('SHIFT')
            #self.OFFSET = 0 #this should not be global. all offsets are immediately transfered to .tlt
            #self.separate_groups = kwargs.get('separate_groups')
            self.n_patches = kwargs.get('n_patches')
            self.fidn = fidn
            self.MinSizeOrOverlapXandY = kwargs.get('MinSizeOrOverlapXandY') 
            self.global_only = global_only
            
            self.RotOption = kwargs.get('RotOption') #taken from aligncom by defualt
            self.TiltOption = kwargs.get('TiltOption')
            self.MagOption = kwargs.get('MagOption')
            self.XTiltOption = XTiltOption
            self.XTiltDefaultGrouping = XTiltDefaultGrouping
            self.NoSeparateTiltGroups = kwargs.get('NoSeparateTiltGroups')
            self.LocalXTiltOption = LocalXTiltOption
            self.LocalXTiltDefaultGrouping = LocalXTiltDefaultGrouping
            self.SurfacesToAnalyze = SurfacesToAnalyze
            self.FixXYZCoordinates = kwargs.get('FixXYZCoordinates')
            
            #PEET or 3D model related
            self.model_file_binning = kwargs.get('model_file_binning') #absolute binning level, i.e. relative to stack. detected from .prm by default
            self.average_volume = kwargs.get('average_volume')
            #self.average_volume_binning = kwargs.get('average_volume_binning') #now detected from volume
            self.prm = kwargs.get('prm')
            self.prm_tomogram_number = kwargs.get('prm_tomogram_number') #numbered from one (for PEET consistency)
            self.prm1 = kwargs.get('prm1') 
            self.prm2 = kwargs.get('prm2') #for gold standard
            self.ite = ite #peet iteration, default last 
            self.cutoff = cutoff
            self.search_rad = kwargs.get('search_rad')
            print('search_rad not the same as limit')
            self.phimax_step = phimax_step
            self.psimax_step = psimax_step
            self.thetamax_step = thetamax_step
            self.get_area = get_area
            
            #DEV
            self.smooth = kwargs.get('smooth')
            self.unreasonably_harsh_filter = kwargs.get('unreasonably_harsh_filter')
            self.no_ctf_convolution = kwargs.get('no_ctf_convolution') #this is for running toy data where no ctf convolution should be applied
            self.poly = kwargs.get('poly')
            self.poly_order = kwargs.get('poly_order')
            self.smooth_ends = kwargs.get('smooth_ends')
            
            print('DEV NOTE xfproduct scaleshifts needs to take into account (at least relative) binning. test this separately')
            
            #self.no_cc_norm = True
            #self.orderL = True
            #self.dosesym = False
            self.ignore_resolution_check = True
            self.ignore_partial = kwargs.get('ignore_partial')
            self.xcor_normalisation = xcor_normalisation #     'none' 'phase_corr' 'by_autocorr' 
            self.padding = padding
            self.cleanup_list = []
            self.pre_bin = kwargs.get('pre_bin')
            self.pre_filter = kwargs.get('pre_filter') # 4 values for mtffilter -hi cutoff,falloff, -lo cutoff,falloff, list
            self.run_default_peet_binning = kwargs.get('run_default_peet_binning') 
            self.keep_peet_binning = keep_peet_binning #run peet at whatever binning the specified peet was run
            self.noisy_ref_std = noisy_ref_std
            self.exclude_worst_pcl_fraction = exclude_worst_pcl_fraction #exclude particles with the worst median ccc. 0/False disables this
            self.exclude_lowest_cc_fraction = exclude_lowest_cc_fraction #Removes the worst scoring shifts. NOT RECOMMENDED
            #self.use_refined_halfmap_models = use_refined_halfmap_models #i.e. combined halfmap models are used for plotback generation
            self.masktomrec_iters = kwargs.get('masktomrec_iters')
            self.min_neighbours = min_neighbours
            self.use_local_median_shifts = use_local_median_shifts #written shifts are replaced by median shifts of local particles
            self.use_median_cc_maps = use_median_cc_maps #take median of cc maps of local particles, get shifts from these (SLOWWW)
            self.shift_std_cutoff = shift_std_cutoff
            
            #to be deprecated:
            self.use_nbr_median = use_nbr_median #if True, ignore weighted mean function, which is wrong anyway. Using weighted mean seems to produce very slightly better results.
            self.use_davens_fsc  = use_davens_fsc#faster, looks worse

        self.update_comfiles()
    
        self.particles = None #extracted_particles object

        """
        List of tasks:
            1) generate default names based on com files                verify_inputs,  initialise_default_paths
            2) check that supplied model is the correct binning         verify_inputs, bin_model
    
            I'm not sure how to deal with excludelist. Currently feeding complete comfile contents to reproject_model.
            This results in particles in ecluded views to be replaced with zeros - which are then removed
            Reprojecting will result in the truncated number of views.  - the query and plotback might not necessarily match,so the extraction will need to deal with it
    
            It might be enough to deal with it in fid2ndarray: the reprojected fid should reflect the tilt excludelist(s) which is what matters
    
    
            3) reproject 3d model                                       reproject_model
            4) generate image files. 
    
    
        
        WRITE INTO TMPFS!!! check_tmpfs
        
        
        if the resolution gets worse, restrict verything in tiltalign and repeat
        
        xfproduct is extremely slow  ??
    
        """
        
        """
        ssorted_pcls
                    [number of tilts:model points per tilt:7
                0=xcoords, 1=ycoords, 2=tilt number, 3=particle index, 4=group id (from 0), 5=tilt angle, 6=defocus)]
                group id == len(groups) indicates unallocated pcls   
        """

    def to_json(self):

        if not isdir(self.npy_dir):
            os.makedirs(self.npy_dir)
        out_inst = deepcopy(self)
        for key in out_inst.__dict__.keys():
            if isinstance(out_inst.__dict__[key], IMOD_comfile):
                out_inst.__dict__[key] = None
            if isinstance(out_inst.__dict__[key], Extracted_particles):
                out_inst.__dict__[key] = None
            if isinstance(out_inst.__dict__[key], np.integer):
                out_inst.__dict__[key] = int(out_inst.__dict__[key])
            if isinstance(out_inst.__dict__[key], np.floating):
                out_inst.__dict__[key] = float(out_inst.__dict__[key])
            if isinstance(out_inst.__dict__[key], (list, tuple)):
                #this shouldn't happen, but need to check for ndarray within lists
                #checking just the first should be enough
                #could catch but better raise an exception
                if len(out_inst.__dict__[key]) > 0:
                    if isinstance(out_inst.__dict__[key][0], np.ndarray):
                        raise Exception('%s is a list of ndarrays, this will break json.dump (%s)' % (key, out_inst.__dict__[key]))
            if isinstance(out_inst.__dict__[key], np.ndarray):
                pickle = join(self.npy_dir, key + '.npy')
                np.save(pickle, self.__dict__[key])
                out_inst.__dict__[key] = pickle
            
        with open(join(self.out_dir, self.out_json), 'w') as f:
            json.dump(out_inst.__dict__, f, indent = 4)


    def update_dirs(self):
        self.out_json = join(self.out_dir, 'flexo.json')
        self.npy_dir = join(self.out_dir, 'npy')
        self.pcle_dir = join(self.out_dir, 'extracted_particles')
        self.xcor_peak_dir = join(self.out_dir, 'xcor_peaks')
        self.match_tomos_dir = join(self.out_dir, 'match_tomos')
        self.init_peet_dir = join(self.out_dir, 'init_peet')
        self.peet_dir = join(self.out_dir, 'peet')

    def update_comfiles(self):
        self.tiltcom = IMOD_comfile(self.rec_dir, 'tilt.com')
        self.newstcom = IMOD_comfile(self.rec_dir, 'newst.com')
        self.aligncom = IMOD_comfile(self.rec_dir, 'align.com')
        self.ctfcom = IMOD_comfile(self.rec_dir, 'ctfcorrection.com')
        
    def verify_inputs(self):
        """Verify input files.
        base_name is read from imod com files if not specified
        

        """  
        
        # check multiprocessing cores ##############
        if not self.machines:
            self.machines = socket.gethostname().split('.')[0]
        elif isinstance(self.machines, int):                
            #first try to get machines from imod_calib_dir and distribute equally between them
            #otherwise use local machine
            machines, cpu_counts = machines_from_imod_calib() #returns empty list if cpu.adoc not in IMOD_CALIB_DIR 
            if machines:
                split_c = int(np.floor(self.machines/len(cpu_counts)))
                tmp_m = [[machines[m]]*min(split_c, cpu_counts[m]) for m in range(len(machines))]
                tmp_m = np.ravel(tmp_m).tolist()
                self.machines = tmp_m
            else:
                machines = socket.gethostname().split('.')[0]
                max_c = multiprocessing.cpu_count()
                self.machines = [machines]*min(self.machines, max_c)
        elif isinstance(self.machines, str):
            self.machines = [self.machines]
        for machine in np.unique(self.machines):
            check_ssh(machine)
        
        #strictly required
        if not isdir(self.rec_dir):
            raise Exception('Specified reconstruction folder does not exist.  %s'
                            % self.rec_dir)
        if realpath(self.rec_dir) == realpath(self.out_dir):
            raise Exception("Output directory cannot be the same as input directory.")
        if not isdir(self.out_dir):
            os.makedirs(self.out_dir)
            

            
        # #either prm or model+motl can be inputs, but prioritise the latter
        # if self.prm and not self.prm_tomogram_number:
        #     print("PEET tomogram number not specified, assuming it's the first")
        #     self.prm_tomogram_number = 1
            
        # print('DEV NOTE: if two FSCs are supplied, first run combine_fsc_halves')
        # if self.prm:
        #     motls, modfiles, tomos = get_mod_motl_and_tomo(self.prm, self.ite) #try to get last iteration                
        #     prm_tomo = tomos[self.prm_tomogram_number - 1]
        #     prm_tomo_apix, prm_tomo_size = get_apix_and_size(prm_tomo)
        
        # # if self.prm2:
        # #     self.prm1 = self.prm
        

        # input files ##############
        if not self.imod_base_name:
            self.imod_base_name = self.tiltcom.get_imod_base_name()
            if self.imod_base_name == '':
                raise Exception('Failed to parse IMOD base name, maybe the tiltcom files are the same?')
        #if not self.base_name:
        self.base_name = self.imod_base_name
        print('DEV NOTE base_name set to be the same as imod_base_name, IMOD_comfile parsing/formatting breaks otherwise...')
            #print('IMOD base name not specified. Using base name of input stack (from tilt.com).')
            
        self.st = self.newstcom.dict['InputFile']
        self.xf = self.newstcom.dict['TransformFile']
        self.ali = self.tiltcom.dict['InputProjections']
        self.full_tomo = self.tiltcom.dict['OutputFile']
        self.tlt = self.tiltcom.dict['TILTFILE']
        self.thickness = self.tiltcom.dict['THICKNESS']
        self.reprojected_mod = join(self.out_dir, self.base_name + '_reprojected.fid')
        if not self.tomo:
            if self.full_tomo.endswith('_full.rec'):
                ext = '_full.rec'
                self.tomo = self.full_tomo[:-len(ext)] + '.rec'
            elif self.full_tomo.endswith('_full_rec.mrc'):
                ext = '_full_rec.mrc'
                self.tomo = self.full_tomo[:-len(ext)] + '_rec.mrc'
        
        if not self.average_volume or not isfile(self.average_volume):
            raise Exception("Average volume not found %s." % self.average_volume)
        if not self.defocus_file:
            self.defocus_file = self.ctfcom.dict['DefocusFile']
        if not isfile(self.defocus_file):
            self.no_ctf_convolution = True

        for mf in [self.xf, self.st, self.ali, self.tlt]:
            if not isfile(mf):
                raise Exception('File not found %s' % mf)

        if not isfile(self.full_tomo):
            if self.pre_bin:
                pass
            else:
                raise Exception('File not found %s' % self.full_tomo)
                
           
        # check binning ##############
        stack_apix, stack_size = get_apix_and_size(self.st)
        tomo_apix, tomo_size = get_apix_and_size(self.tomo)
        #full_tomo_apix, full_tomo_size = get_apix_and_size(self.full_tomo)    
        ali_apix, ali_size = get_apix_and_size(self.ali)
        
        # if tomo_apix != full_tomo_apix:
        #     raise ValueError(('The original unrotated tomogram (%s)' +
        #                      'pixel size does not match the rotated tomogram (%s).')
        #                      % (split(self.full_tomo)[1], split(self.tomo)[1]))
            
        self.apix = float(tomo_apix)
        self.tomo_size = tomo_size.tolist()
        self.st_size = stack_size.tolist()
        self.st_apix = float(stack_apix)
        self.tomo_binning = float(np.round(tomo_apix/stack_apix, decimals = 0))
        if self.pre_bin:
            if isinstance(self.pre_bin, bool):
                self.pre_bin = self.tomo_binning
                print('Remaking initial tomograms without binning...')
            else:
                self.tomo_binning = self.pre_bin
            
        if not self.box_size:
            average_apix, average_size = get_apix_and_size(self.average_volume)
            map_binning = average_apix/stack_apix
            rel_bin = self.tomo_binning/map_binning
            average_size = average_size[0]/rel_bin
            if average_size%2:
                average_size += 1
            weebit = int(np.ceil(average_size/20))*2
            self.box_size = [int(average_size + weebit), int(average_size + weebit)]
        elif isinstance(self.box_size, int):
            self.box_size = [self.box_size, self.box_size]
            
        if not self.particle_interaction_distance:
            self.particle_interaction_distance = float(min(self.box_size[0]*2, np.sort(self.tomo_size)[1]//16)) #ballpark....
            
        # prm, models ##############
        # one of three options: 1) specify prm only, 2) specify prm1 and prm2, 3) specify model and motl
        # prioritise halfmaps, then single prm over model and motl
        if not self.prm_tomogram_number:
            self.prm_tomogram_number = 1      
            
        if not self.prm:
            if not self.prm1 and not self.prm2:
                if not self.model_file and self.motl:
                    raise Exception ('One of the following is required:\n1) "combined" PEET parameter file, 2) 2 halfmap PEET parameter files or model file and motive list')

        if self.prm1 and self.prm2:
            modfiles, motls = combine_fsc_halves(self.prm1, self.prm2,
                            self.prm_tomogram_number, self.peet_dir, self.ite)
            _, _, tomos = get_mod_motl_and_tomo(self.prm1, self.ite) #try to get last iteration            
            prm_tomo = tomos[self.prm_tomogram_number - 1]
            prm_tomo_apix, prm_tomo_size = get_apix_and_size(prm_tomo)
            self.model_file = modfiles[self.prm_tomogram_number - 1]
            self.motl = motls[self.prm_tomogram_number - 1]

        elif self.prm and not (self.prm1 and self.prm2):
            motls, modfiles, tomos = get_mod_motl_and_tomo(self.prm, self.ite) #try to get last iteration
            prm_tomo = tomos[self.prm_tomogram_number - 1]
            prm_tomo_apix, prm_tomo_size = get_apix_and_size(prm_tomo)
            self.model_file = modfiles[self.prm_tomogram_number - 1]
            self.motl = motls[self.prm_tomogram_number - 1]

        #Write a model file with the same binning as input tomogram. 
        if not self.model_file_binning:
            if self.prm or (self.prm1 and self.prm2):
                self.model_file_binning = float(np.round(prm_tomo_apix/stack_apix, decimals = 0)) #work with "absolute binning"
            else:
                self.model_file_binning = self.tomo_binning
                
        if self.model_file_binning != self.tomo_binning:
            to_bin = self.tomo_binning/self.model_file_binning
            mod_str = ('.').join(split(self.model_file)[1].split('.')[:-1])
            output_model = join(self.out_dir, mod_str + '_bin%s.mod' % to_bin)
            output_motl = join(self.out_dir, mod_str + '_bin%s.csv' % to_bin) #write a new motl without offsets to avoid adding them several times by mistake
            print('Binning model file %s' % output_model)
            bin_model(self.model_file, output_model, to_bin, motl = self.motl,
                      out_motl = output_motl)
            self.model_file = output_model
            self.motl = output_motl
            
        if self.prm or (self.prm1 and self.prm2) and self.keep_peet_binning:
            self.peet_binning = float(np.round(prm_tomo_apix/stack_apix, decimals = 0))
        else:
            self.peet_binning = self.tomo_binning            
        
        tilt_angles = [float(x.strip()) for x in open(self.tlt, 'r')]
        if len(tilt_angles) != ali_size[2]:     
            raise ValueError((
                """The number of tilt angles in tilt file (%s)
                does not match the number of images in aligned stack (%s).
                This could mean that there is "ExcludeSections" entry in
                newst.com. This is not currently supported, excludelist
                should be in align.com or tilt.com.""")
                             % (len(tilt_angles), ali_size[2]))   

        #this may cause more problem than it solves........moved to generate_image_data
        # #need to make sure that the orig stack has excludelist entries removed
        # #this is technically only required for 2/3 current ways of generating
        # #the image data (use_init_ali and noisy_ref)       
        # if self.newstcom.namingstyle == 0:
        #     ext = '_orig.ali'
        # elif self.newstcom.namingstyle == 1:
        #     ext = '_orig_ali.mrc'
        # new_ali = join(self.out_dir, self.base_name + ext)                    
        # if ali_size[2] != stack_size[2] - len(self.excludelist):
        #     check_output('newstack -fromone -exclude %s %s %s' %
        #                  ((',').join([str(int(x)) for x in self.excludelist]),
        #                   self.ali, new_ali), shell = True)
        # else:
        #     if isfile(new_ali):
        #         os.unlink(new_ali)
        #     os.symlink(self.ali, new_ali)
        # self.ali = new_ali
                    
        # excludelist ##############
        self.excludelist = np.unique(np.hstack((self.tiltcom.excludelist,
                                                self.aligncom.excludelist,
                                                ))).tolist()
        if self.excludelist == ['']:
            #this happens if a logfile has EXCLUDELIST keyword but no values. It also breaks running tilt so have to raise
            raise Exception('Invalid excludelist entries in one of input com files.')
            
        if np.isin(0, self.excludelist):
            raise Exception ('0 entry found in excludelist. IMOD numbering is from 1 by default.')
        #excludelists are kept intact in IMOD comfiles. This means that reprojected
        #stacks will not have excludelist tilts.
        if self.newstcom.excludelist:
            raise Exception ('Newstack ExcludeSections option is currently not supported.')
        
        # imaging params ##############
        self.V = self.ctfcom.dict['Voltage']*1E3
        if self.V == 200*1E3:
            warnings.warn('Voltage specified in ctfcorrection.com is 200kV. Change this in IMOD gui/ctfcorrection.com if this is not correct.')
        self.Cs = self.ctfcom.dict['SphericalAberration']*1E7
        self.ampC = self.ctfcom.dict['AmplitudeContrast']       
                
        # exposure ##############
        
        excludelist_mask =  np.isin(np.arange(stack_size[2]), np.array(self.excludelist) - 1, invert = True)
        if isinstance(self.lowfreqs, (bool, type(None))):
            ismdoc = not isinstance(self.mdoc, (bool, type(None)))
            isorder = (all((not isinstance(self.order, (bool, type(None))),
                          not isinstance(self.tilt_dose, (bool, type(None)))
                          )) or isinstance(self.order, str)) # try to read the file in case it  contains dose
            if ismdoc or isorder:
                self.lowfreqs = optimal_lowpass_list(order = self.order,
                          mdoc = self.mdoc, flux = self.flux, tilt_dose = self.tilt_dose,
                          pre_exposure = self.pre_exposure, tlt = self.tlt,
                          orderfile_type = self.orderfile_type)
                if len(self.lowfreqs) != stack_size[2]:
                    raise Exception('Number of entries in mdoc or order file does (%s) not match the number of tilts in the tilt series (%s)' %
                                    (len(self.lowfreqs), stack_size[2]))
                self.lowfreqs = np.array(self.lowfreqs)[excludelist_mask].tolist()
            else:
                lowfreqs = np.zeros(stack_size[2]) + optimal_lowpass(self.pre_exposure)
                self.lowfreqs = lowfreqs[excludelist_mask]
                
        if len(self.lowfreqs) != stack_size[2]:
            if len(self.lowfreqs) != stack_size[2] - len(self.excludelist):
                raise Exception ('The length of specified tilt series order (or lowfreqs) does not match'
                                 + ' the number of tilts (even considering tilt.com excludelist).'
                                 + '\nLength of order: %s tilt series: %s\nExcludelist entries (numbered from 1) %s' % (
                                     len(self.lowfreqs), stack_size[2], self.excludelist)
                                 )
            
                

        # if isinstance(self.zero_tlt, bool): #zero_tlt no longer used
        #     self.zero_tlt = find_nearest(self.tilt_angles, 0)
        #     #zero_tlt needs to be adjusted for excludelist. It's not used
        #     #in flexo_model_from..., only in just_flexo where it needs to
        #     #correspond match alis, etc where excludelist tilts were removed 
        #     if not isinstance(self.excludelist, bool): 
        #         #just count the number of excludelist entirest that are smaller
        #         #than zero_tlt, then subtract from zero_tlt
        #         #excludelist is numbered from 1
        #         relevant_excludelist = [x for x in self.excludelist if x <= self.zero_tlt + 1]
        #         self.zero_tlt -= len(relevant_excludelist) 
        #         self.zero_tlt += 1 #numbered from 1!
        
        if self.noisy_ref and self.xcor_normalisation == 'phase_corr':
            warnings.warn('phase_corr is incompatible with noisy_ref. normalisation set to "none"')

        self.flg_inputs_verified = True                
        self.to_json()

    def update_comdict(self, orig_dict, mod_dict):
        for key in mod_dict.keys():
            orig_dict[key] = mod_dict[key]
        return orig_dict

    def format_align(self, output_binning, OFFSET = 0, out_dir = False,
                     fid_per_patch = 40):
        """
        Generates an align.com. 
        Important: either keep using the original angle offset and .rawtlt, or
        zero out angle offset and use .tlt.

        Parameters
        ----------
        OFFSET : float, optional
            Angle offset. The default is 0: The original offsets is kept.
        out_dir : str, optional
            Path to output dir. Using self.out_dir by default
        fid_per_patch : int, optional
            Used if self.n_patches is not specified and there is no local patch 
            entry in align.com to get a reasonable number of local patches


        """
        if not self.global_only:
            if 'NumberOfLocalPatchesXandY' not in self.aligncom.dict.keys() and not self.n_patches:
                tmp = min(3, self.st_size[2]/fid_per_patch) #more than 3 rarely does anything good
                tmp = (np.array(self.tomo_size)/np.max(self.tomo_size))[:2] * tmp #for rectangular chips
                n_patches = np.array(np.round(tmp), dtype = int).tolist()
                
            elif self.n_patches:
                n_patches = self.n_patches
            else:
                n_patches = False
        
            if 'TargetPatchSizeXandY' in self.aligncom.dict.keys():
                print("""IMOD tiltalign option TargetPatchSizeXandY is not used in Flexo,
                      NumberOfLocalPatchesXandY is used instead. This is calculated
                      automatically if not explcitly specified through n_patches.
                      """)
                del self.aligncom.dict['TargetPatchSizeXandY']
        else:
            n_patches = False
            
        if not out_dir:
            out_dir = self.out_dir
        out_aligncom = deepcopy(self.aligncom)
        
        base_output = join(out_dir, self.base_name)
        resid = str(base_output) + '.resid'
        resmod = str(base_output) + '.resmod' 
        
        if (self.aligncom.dict['MagReferenceView'] == 1 or 
            self.aligncom.dict['MagReferenceView'] == self.st_size[2]): 
            if not isinstance(self.ssorted_pcls, (bool, type(None))):
                mag_ref = find_nearest(self.ssorted_pcls[:, 0, 5], 0) + 1
            else:
                mag_ref = self.st_size[2]//2
        else:
            mag_ref = self.aligncom.dict['MagReferenceView']
    
        
        a_dict = {'ModelFile': self.out_fid,
                'ImageFile': self.ali,
                'RotationAngle': 0, #this needs to be 0 as alignment is done on .ali not .preali
                'ImagesAreBinned': int(output_binning),
                'MinFidsTotalAndEachSurface': self.fidn,
                'XTiltOption': self.XTiltOption,
                'XTiltDefaultGrouping': self.XTiltDefaultGrouping,
                'LocalXTiltOption': self.LocalXTiltOption,
                'LocalXTiltDefaultGrouping': self.LocalXTiltDefaultGrouping,
                'SurfacesToAnalyze': self.SurfacesToAnalyze,
                'MagReferenceView': int(mag_ref)
                
                }
        if 'OutputZFactorFile' in self.aligncom.dict.keys():
            a_dict['OutputZFactorFile'] = str(base_output) + '.zfac'
        if 'RobustFitting' not in self.aligncom.dict.keys(): #always use robust fitting since we always haven noisy fid
            a_dict['RobustFitting'] = None
        if 'WarnOnRobustFailure' not in self.aligncom.dict.keys(): #always use robust fitting since we always haven noisy fid
            a_dict['WarnOnRobustFailure'] = None

            
        
        if not isinstance(n_patches, bool):
            a_dict['NumberOfLocalPatchesXandY'] = n_patches
        if self.global_only:
            a_dict['LocalAlignments'] = 0
        else:
            a_dict['LocalAlignments'] = 1
        if self.NoSeparateTiltGroups is not None:
            a_dict['NoSeparateTiltGroups'] = self.NoSeparateTiltGroups
        if self.MinSizeOrOverlapXandY is not None:
            a_dict['MinSizeOrOverlapXandY'] = self.MinSizeOrOverlapXandY
        if OFFSET:
            a_dict['AngleOffset'] = OFFSET
        if self.RotOption:
            a_dict['RotOption'] = self.RotOption
        if self.TiltOption:
            a_dict['TiltOption'] = self.TiltOption
        if self.MagOption:
            a_dict['MagOption'] = self.MagOption
        if self.FixXYZCoordinates is not None:
            if self.FixXYZCoordinates:
                self.FixXYZCoordinates = 1
            else:
                self.FixXYZCoordinates = 0
            a_dict['FixXYZCoordinates'] = self.FixXYZCoordinates
            
        b_dict = {'InputFile1': self.newstcom.dict['TransformFile'], #use old xf
            'InputFile2': str(base_output) + '.tltxf',
            'OutputFile': str(base_output) + '.xf',
            'ScaleShifts': [1, int(output_binning)]
            }
        b_footer = '$if (-e %s) patch2imod -s 10 %s %s' % (resid, resid, resmod)
        
        
        out_aligncom.point2out_dir(out_dir = out_dir, base_name = self.base_name)
        out_aligncom.dict = self.update_comdict(out_aligncom.dict, a_dict)
        out_aligncom.b_dict = self.update_comdict(out_aligncom.b_dict, b_dict)
        out_aligncom.b_footer = [b_footer]
        
        #copy .rawtlt over
        if not isfile(out_aligncom.dict['TiltFile']):
            copyfile(self.aligncom.dict['TiltFile'], out_aligncom.dict['TiltFile'])
        
        out_aligncom.write_comfile(out_dir)
    
    def format_newst(self, output_binning, change_comname = False,
                     use_rec_dir = False, out_dir = False, output_ali = False):
        """
        Modifies and writes newst.com. Output paths will point to out_dir,
        input path (stack file) will be kept pointing to the rec_dir by default.

        Parameters
        ----------
        output_binning : int
            Binning relative to raw stack.
        change_comname : str, optional
            Change name of comfile?. The default is False (name is taken from input comfile).
        use_rec_dir : bool, optional
            If True, input files are kept the same (absolute paths to the reconstruction directory).
            The default is False.
        out_dir : str, optional
            Path to output directory. Uses Flexo.out_dir by default. The default is False.
        output_ali: str, optional
            Force output file name. Use absolute path.

        """
        
        
        if self.newstcom.namingstyle == 0:
            ext = '.ali'
        elif self.newstcom.namingstyle == 1:
            ext = '_ali.mrc'
            
        if not out_dir:
            out_dir = self.out_dir
        if not output_ali:            
            output_ali = join(out_dir, self.base_name + ext)

        a_dict = {'InputFile': self.newstcom.dict['InputFile'],
                  'OutputFile': output_ali,
                  #'TransformFile': join(out_dir, self.base_name + '.xf'),
            }
        if output_binning < 1:
            raise KeyError('output binning cannot be less than 1')

        binned_size = get_binned_size(self.st_size, output_binning)[0][:2]
        #rotation could happen, order size based on existing tilt.com
        if 'SizeToOutputInXandY' in self.newstcom.dict.keys():
            orig_size = self.newstcom.dict['SizeToOutputInXandY']
            #else:
            #    _, orig_size = get_apix_and_size(self.ali)
            if orig_size[0] < orig_size[1]:
                binned_size = np.sort(binned_size[:2])
            else:
                binned_size = np.sort(binned_size[:2])[::-1]
            a_dict['SizeToOutputInXandY'] = binned_size.tolist()
        a_dict['BinByFactor'] = int(output_binning)
       

        out_newstcom = deepcopy(self.newstcom)
        if not use_rec_dir:
            out_newstcom.point2out_dir(out_dir = out_dir, base_name = self.base_name)
        out_newstcom.dict = self.update_comdict(out_newstcom.dict, a_dict)
        out_newstcom.write_comfile(out_dir, change_name = change_comname)
        
    def format_tilt(self, output_binning, output_rec = False, input_ali = False,
                    OFFSET = False,
                    SHIFT = False, change_comname = False, use_rec_dir = False,
                    out_dir = False):
        """
        Modifies and writes tilt.com. Input and output paths point to 
        out_dir by default.

        Parameters
        ----------
        output_binning : int
            Binning relative to raw stack.
        output_rec : str, optional
            Non-default name of output volume. The default is False.
        input_ali : str, optional
            Force input image file name, use absolute path.
        OFFSET : float, optional
            Tilt angle offset (refer to IMOD tilt doc). The default is False (0.0).
        SHIFT : list of two floats, optional
            Volume shift (refer to IMOD tilt doc). The default is False (0.0, 0.0).
        change_comname : str, optional
            Nondefault com script name. The default is False (tilt.com).
        use_rec_dir : bool, optional
            Inputs will refer to the reconstruction directory, outputs to the
            out_dir. The default is False.
        out_dir : str, optional
            Path to output directory. Uses Flexo.out_dir by default. The default is False.

        Returns
        -------
        None

        """
        out_tiltcom = deepcopy(self.tiltcom)
        if not out_dir:
            out_dir = self.out_dir
            
        base_output = join(out_dir, self.base_name)
        if self.tiltcom.namingstyle == 0:
            rec_ext = '.rec'
        elif self.tiltcom.namingstyle == 1:
            rec_ext = '_rec.mrc'
            
        if not output_rec:
            output_rec = str(base_output) +  '_full' + rec_ext
        
        a_dict = {'OutputFile': output_rec,
                  'IMAGEBINNED': int(output_binning)
                  }
        if OFFSET:
            a_dict['OFFSET'] = OFFSET
        if SHIFT:
            a_dict['SHIFT'] = SHIFT
        if input_ali:
            a_dict['InputProjections'] = input_ali
        if self.global_only:
            if 'LOCALFILE' in out_tiltcom.dict:
                del out_tiltcom.dict['LOCALFILE']

        
        if not use_rec_dir:
            out_tiltcom.point2out_dir(out_dir = out_dir, base_name = self.base_name)
        out_tiltcom.dict = self.update_comdict(out_tiltcom.dict, a_dict)
        out_tiltcom.write_comfile(out_dir, change_name = change_comname)
        
    def format_ctfcom(self, out_dir = False, input_ali = False, output_ali = False):
        
        if not out_dir:
            out_dir = self.out_dir
        
        out_defocus = join(out_dir, self.base_name + '.defocus')
        if not isfile(out_defocus):
            copyfile(self.ctfcom.dict['DefocusFile'], out_defocus)

        a_dict = {'PixelSize': self.apix/10.}
        if input_ali:
            a_dict['InputStack'] = input_ali
        if output_ali:
            a_dict['OutputFileName'] = output_ali
        
        out_ctfcom = deepcopy(self.ctfcom)
        out_ctfcom.point2out_dir(out_dir = out_dir, base_name = self.base_name)
        out_ctfcom.dict = self.update_comdict(out_ctfcom.dict, a_dict)
        out_ctfcom.write_comfile(out_dir)
   
    def reproject_model(self, model_file = False, out_fid = False):
        """
        Reprojects 3D model into 2D fiducial model, uses self.model_file by
        default. The 3D model should fit the original rotated tomogram.

        Parameters
        ----------
        mdoel_file : str, optional
            Path to model file. The default is False (uses self.model_file).

        """
        
        if not model_file:
            model_file = self.model_file
        if not out_fid:
            out_fid = self.reprojected_mod
        
        #transform model to fit full tomo
        tmp_mod = join(self.out_dir, 'tmp.mod')
        self.full_model_file =  join(self.out_dir, split(model_file)[1][:-4] + '_full.mod')
        reproject_mod_log = join(self.out_dir, 'tilt_reproject_model.log')
        
        self.cleanup_list.extend((tmp_mod, reproject_mod_log))
        
        check_output('imodtrans -I %s %s %s' % (
                self.tomo, model_file, tmp_mod), shell = True)
        check_output('imodtrans -i %s %s %s' % (
                self.full_tomo, tmp_mod, self.full_model_file), shell = True)
    
    
        # reproject_cmd_list = ['-InputProjections', self.ali,
        #                       '-OutputFile', out_fid,
        #                       '-ProjectModel', self.full_model_file]
        # tiltcom_cmd_list = self.tiltcom.get_command_list(
        #     append_to_exclude_keys = ['InputProjections', 'OutputFile']) #these must not have -
        # tiltcom_cmd_list.extend(reproject_cmd_list)
        rep_com = deepcopy(self.tiltcom)
        rep_com.dict['InputProjections'] = self.ali
        rep_com.dict['OutputFile'] = out_fid
        rep_com.dict['ProjectModel'] = self.full_model_file
        if 'EXCLUDELIST' in rep_com.dict.keys():
            del rep_com.dict['EXCLUDELIST'] #this is a long story but trust me it's better to exclude the model points after the fact
        if 'EXCLUDELIST2' in rep_com.dict.keys():
            del rep_com.dict['EXCLUDELIST2']
        rep_com.write_comfile(self.out_dir, change_name = 'reproject_model.com')
        imodscript('reproject_model.com', self.out_dir) 
        
        # run_generic_process(tiltcom_cmd_list, out_log = reproject_mod_log)

    def make_and_reproject_plotback(self, model_file, plotback3d, plotback2d, motl,
                                    mask2d = False):
        """
        Generates a plotback and reprojects it into a 2D tilt series.

        Parameters
        ----------
        model_file : str
            Path to model file. Expecting "unrotated" orientation (not _full.rec)
        plotback3d : str
            Path to output plotback volume.
        plotback2d : str
            Path to output reprojected plotback.
        motl : str
            Path to PEET motive list.
        mask2d : str, Optional.
            Path to output 2d mask.

        Returns
        -------
        None.

        """

        #make plotback
        replace_pcles(self.average_volume, self.tomo_size, motl,
                      model_file, plotback3d, self.apix,
                      rotx = False)
        #apply mask
        if self.lamella_mask_path:
            tmp_plotback3d = plotback3d + '~'
            os.rename(plotback3d, tmp_plotback3d)
            run_generic_process(['clip', 'multiply', tmp_plotback3d,
                                 self.lamella_mask_path, plotback3d])
            os.remove(tmp_plotback3d)
            
        reproject_volume(plotback2d, tomo = plotback3d, tiltcom = self.tiltcom)
        
        p, s, o = get_apix_and_size(self.full_tomo, origin = True)
        check_output('alterheader -d %s,%s,%s -o %s,%s,%s %s' %
            (self.apix, self.apix, self.apix, o[0], o[1], o[2], plotback3d), shell = True)
        p, s, o = get_apix_and_size(self.ali, origin = True)
        check_output('alterheader -d %s,%s,%s -o %s,%s,%s %s' %
            (self.apix, self.apix, self.st_apix,
             o[0], o[1], o[2], plotback2d), shell = True)
        
        if mask2d:
            get2d_mask_from_plotback(plotback2d, mask2d, dilation = self.limit*2)
            
    def reproject_tomo(self, reprojected_tomo, plotback3d = False,
                       out_mask = False, masked_tomo = False,
                       masktomrec = False):
        """
        Reprojects full tomo, masks areas outside particles if 3D plotback 
        is specified.

        Parameters
        ----------
        reprojected_tomo : str
            Path to output file.
        plotback3d : str, optional
            Path to 3D plotback, which will be used to generate particle mask.
            The default is False.
        out_mask : str, optional
            Path to output mask.
        masked_tomo : str, optional
            Path to temporary masked volume. This is required for parallelisation.

        Returns
        -------
        None.

        """
        tomo2reproject = self.full_tomo
        if self.mask_tomogram and plotback3d:
            #make and apply particle mask
            if not masked_tomo:
                masked_tomo = join(self.out_dir, split(self.full_tomo)[1])
            if not out_mask:
                out_mask = join(self.out_dir, self.base_name + '_particle_mask.mrc')        
            mask_from_plotback(plotback3d, out_mask, size = self.dilation_size,
                           lamella_mask_path = self.lamella_mask_path)
            
            run_generic_process(['clip', 'multiply', tomo2reproject,
                                 out_mask, masked_tomo])   
            tomo2reproject = masked_tomo
         
        reproject_volume(reprojected_tomo, tomo = tomo2reproject, tiltcom = self.tiltcom)  
        
        p, s, o = get_apix_and_size(tomo2reproject, origin = True)
        check_output('alterheader -d %s,%s,%s -o %s,%s,%s %s' %
            (self.apix, self.apix, self.st_apix,
             o[0], o[1], o[2], reprojected_tomo), shell = True)
        print('DEV NOTE: I wanted to test supersampling here (and plotback)')    
                
    def generate_image_data(self):
        if self.lamella_model and not self.lamella_mask_path:
            
            self.lamella_mask_path = join(self.out_dir, self.base_name + '_lamella_mask.mrc')
            make_lamella_mask(self.lamella_model, tomo_size = [self.tomo_size[0], self.tomo_size[2], self.tomo_size[1]],
                              out_mask = self.lamella_mask_path, rotx = False)
            print('DEV NOTE: making lamella mask is unfinished. Could check for coordinates outside volume and rotate if needed.')
        
        #first, reproejcted 3d model to 2d fiducial model
        self.reproject_model()
        #ssorted_pcls is used to keep track of particle params. shape: [number of tilts:model points per tilt:7
        #   0=xcoords, 1=ycoords, 2=tilt number, 3=particle index, 4=group id (from 0), 5=tilt angle, 6=defocus)]
        self.ssorted_pcls = fid2ndarray(self.reprojected_mod,
                defocus_file = self.defocus_file, ali = self.ali,
                excludelist = self.excludelist,
                base_name = self.base_name, out_dir = self.out_dir,
                apix = self.apix, tlt = self.tlt)
        
        if self.non_overlapping_pcls:
            (self.fid_list, self.groups, remainder, self.ssorted_pcls,
             self.split3d_models, self.split_motls
             ) = make_non_overlapping_pcl_models(self.ssorted_pcls,
            self.box_size, self.out_dir, model3d = self.model_file,
            motl3d = self.motl)
                                                 
            self.excluded_particles.extend(remainder)
        else:
            self.fid_list = [self.reprojected_mod]
            self.groups = np.ones(self.ssorted_pcls.shape[1], dtype = bool)
            self.split3d_models = [self.model_file]
            self.split_motls = [self.motl]
        
        plotback3d = [join(self.out_dir, self.base_name + '_plotback3D_%02d.mrc' % n) 
                     for n in range(len(self.fid_list))]
        
        if self.noisy_ref: # in which case the tomo only needs to be reprojected once, so all the files have the same name
            tmp_r = np.zeros(len(self.fid_list), dtype = int)
            self.mask_tomogram = False # for noisy_ref, the tomo must be unmasked
            
        else:
            tmp_r = range(len(self.fid_list))
        mask3d = [join(self.out_dir, self.base_name + '_pcle_mask_%02d.mrc' % n) 
                     for n in tmp_r]
        masked_tomo = [join(self.out_dir, self.base_name + '_full_masked_%02d.mrc' % n) 
                     for n in tmp_r]
        reprojected_tomo = [join(self.out_dir, self.base_name + '_reprojected_%02d.mrc' % n) 
                     for n in tmp_r]
    
        self.plotback2d = [join(self.out_dir, self.base_name + '_plotback2D_%02d.mrc' % n) 
                     for n in range(len(self.fid_list))]
        
        if self.apply_2d_mask:
            self.mask2d = [join(self.out_dir, self.base_name + '_mask2D_%02d.mrc' % n) 
                     for n in range(len(self.fid_list))]
        else:
            self.mask2d = [False for n in range(len(self.fid_list))]
        
        if self.masktomrec_iters:
            self.noisy_ref = False
            self.use_init_ali = False
        #in these cases the original projections are query
        if self.noisy_ref or self.use_init_ali:
            #need to make sure that the orig stack has excludelist entries removed
            ali_apix, ali_size = get_apix_and_size(self.ali)
            if self.newstcom.namingstyle == 0:
                ext = '_query.ali'
            elif self.newstcom.namingstyle == 1:
                ext = '_query_ali.mrc'
            new_ali = join(self.out_dir, self.base_name + ext)
            if len(self.excludelist) > 0:
                check_output('newstack -fromone -exclude %s %s %s' %
                              ((',').join([str(int(x)) for x in self.excludelist]),
                              self.ali, new_ali), shell = True)
            else:
                if isfile(new_ali):
                    os.unlink(new_ali)
                os.symlink(self.ali, new_ali)
        
            self.query2d = [new_ali]*len(plotback3d)
       
        else:
            self.query2d = reprojected_tomo
            
        self.cleanup_list.extend(plotback3d)
        self.cleanup_list.extend(mask3d)
        self.cleanup_list.extend(masked_tomo)
        self.cleanup_list.extend(reprojected_tomo)
        
        self.to_json()
        
        #plotback: write files for processchunks, 2d and 3d
        for chunk_n in range(len(plotback3d)):
            chunk_path = join(self.out_dir, 'plotback-%03d.com' % chunk_n)
            out_s = (
            '>sys.path = [%s]' % self.path,
            '>from flexo_pipeline import Flexo',
            '>f = Flexo(json_attr = "%s")' % self.out_json,
            '>f.make_and_reproject_plotback("%s", "%s", "%s", "%s", "%s")' % (
                self.split3d_models[chunk_n], plotback3d[chunk_n],
                self.plotback2d[chunk_n], self.split_motls[chunk_n],
                self.mask2d[chunk_n])
            )
            with open(chunk_path, 'w') as f:
                for line in out_s:
                    f.write(line + '\n')
                
        run_processchunks('plotback', self.out_dir, self.machines)

                                                                          
        #now reproject tomograms
        #use_init_ali: use original projections as query, so only plotback is being reprojected
        #noisy_ref: original projections are query, but need reprojected tomo to add to the plotback
        if self.noisy_ref or not self.use_init_ali:
            if self.noisy_ref:
                chunk_path = join(self.out_dir, 'tomorep-%03d.com' % 0)
                out_s = (
                '>sys.path = [%s]' % self.path,
                '>from flexo_pipeline import Flexo',
                '>f = Flexo(json_attr = "%s")' % self.out_json,
                '>f.reproject_tomo("%s", "%s")' % (
                    reprojected_tomo[0], plotback3d[0])
                )
                with open(chunk_path, 'w') as f:
                    for line in out_s:
                        f.write(line + '\n')  
                    
                run_processchunks('tomorep', self.out_dir, self.machines)                
                
                #add reprojected tomo to plotback --> 2d reference
                for chunk_n in range(len(plotback3d)):
    
                    run_generic_process(['newstack', '-mea', '0,1',
                                         reprojected_tomo[chunk_n],
                                          reprojected_tomo[chunk_n]]) #it would be better to go through an intermediate but cba
                    run_generic_process(['newstack', '-mea', '0,%s' % self.noisy_ref_std,
                                         self.plotback2d[chunk_n],
                                         self.plotback2d[chunk_n]])
                    tmp_sum = self.plotback2d[chunk_n] + '~'
                    if isfile(tmp_sum):
                        os.remove(tmp_sum)
                    self.cleanup_list.append(tmp_sum)
                    run_generic_process(['clip', 'add', reprojected_tomo[chunk_n],
                                         self.plotback2d[chunk_n], tmp_sum])
                    os.rename(tmp_sum, self.plotback2d[chunk_n])
            elif self.masktomrec_iters:
                for chunk_n in range(len(plotback3d)):
                    sub_dir = join(self.out_dir, 'masktomrec%02d' % chunk_n)
                    final_ts = join(sub_dir, 'ts%02d.mrc' % self.masktomrec_iters)
                    chunk_path = join(self.out_dir, 'tomorep-%03d.com' % chunk_n)
                    out_s = (
                    '>sys.path = [%s]' % self.path,
                    '>from flexo_pipeline import *',
                    '>f = Flexo(json_attr = "%s")' % self.out_json,
                    '>tomo_subtraction(f.tiltcom, "%s",plotback3d = "%s", iterations = %s, dilation_size = %s)' % (
                        sub_dir, plotback3d[chunk_n], 
                        self.masktomrec_iters, int(self.limit*1.5)),
                    '$mv %s %s' % (final_ts, reprojected_tomo[chunk_n])
                    )

                    
                    with open(chunk_path, 'w') as f:
                        for line in out_s:
                            f.write(line + '\n')  
                    
                run_processchunks('tomorep', self.out_dir, self.machines)
                
            else:
                for chunk_n in range(len(plotback3d)):
                    chunk_path = join(self.out_dir, 'tomorep-%03d.com' % chunk_n)
                    out_s = (
                    '>sys.path = [%s]' % self.path,
                    '>from flexo_pipeline import Flexo',
                    '>f = Flexo(json_attr = "%s")' % self.out_json,
                    '>f.reproject_tomo("%s", "%s", "%s", "%s")' % (
                        reprojected_tomo[chunk_n], plotback3d[chunk_n],
                        mask3d[chunk_n], masked_tomo[chunk_n]), 
                    )
                    with open(chunk_path, 'w') as f:
                        for line in out_s:
                            f.write(line + '\n')  
                    
                run_processchunks('tomorep', self.out_dir, self.machines)
                    
                    
                    
        self.flg_image_data_exist = True
        self.to_json()

    def _extract_and_cc(self, pcl_indices, write_pcles = True): #this should ideally be split in two   
        """pcl_indices : list or 1D ndarray
        """
        def extract_filter_write(n):
            #aid = ali_f_id[x]
            if self.apply_2d_mask:
                normalise_first = True
            else:
                normalise_first = False
                
            query = extract_2d(arr_ali[ali_arr_ids[n]],
                               self.ssorted_pcls[:, pcl_indices[n], :3],
                               self.box_size, normalise_first = normalise_first)
            ref = extract_2d(arr_plotback[plotback_arr_ids[n]],
                             self.ssorted_pcls[:, pcl_indices[n], :3]
                             , self.box_size, normalise_first = normalise_first)
            if self.apply_2d_mask:
                mask = extract_2d(arr_mask[plotback_arr_ids[n]],
                             self.ssorted_pcls[:, pcl_indices[n], :3], self.box_size)
            #there is support for running averages of neighbouring tilts, could be worth playing with
            #use offsets = True in extract_2d_simplified to get offsets

            partial_list = np.zeros(len(query), dtype = float)
            if self.ignore_partial:
                #skip CC if > 20% of a query is flat
                for l in range(len(query)):
                    #can just check one line of each dim
                    dc = np.max(np.unique(np.diagonal(query[l]), return_counts = True)[1])
                    partial_list[l] = dc/query[0].shape[0]
                    # partial_list[l] = np.max(np.unique(query[l], 
                    #                                    return_counts = True)[1])
                    # partial_list /= query[0].size

            #write out particle stacks
            if write_pcles:
                testpcl = join(self.pcle_dir, 'query%0*d.mrc'
                               % (num_digits, pcl_indices[n]))
                testref = join(self.pcle_dir, 'ref%0*d.mrc'
                               % (num_digits, pcl_indices[n]))
                write_mrc(testpcl, query)
                write_mrc(testref, ref)

            ref = ctf_convolve_andor_dosefilter_wrapper(ref, self.apix,
                V = self.V, Cs = self.Cs, ampC = self.ampC, ps = self.ps,
                defocus = defoci[:, n], butter_order  = self.butter_order,
                no_ctf_convolution = self.no_ctf_convolution,
                padding = padding, lowfreqs = self.lowfreqs, phaseflip = self.phaseflip)
            
            query = ctf_convolve_andor_dosefilter_wrapper(query, self.apix,
                V = self.V, Cs = self.Cs, ampC = self.ampC, ps = self.ps,
                defocus = 0, butter_order  = self.butter_order,
                no_ctf_convolution = self.no_ctf_convolution,
                padding = padding, lowfreqs = self.lowfreqs)
            
            if self.apply_2d_mask:
                ref = ref*mask
                query = query*mask

            #write out filtered particle stacks
            if write_pcles:
                testpcl = join(self.pcle_dir, 'fquery%0*d.mrc'
                               % (num_digits, pcl_indices[n]))
                testref = join(self.pcle_dir, 'fref%0*d.mrc'
                               % (num_digits, pcl_indices[n]))
                #if not os.path.isfile(testpcl):
                write_mrc(testpcl, query)
                write_mrc(testref, ref)
            return ref, query, partial_list

        def cc(ref, query, partial_list, n):
            ccmaps = []
            for y in range(len(ref)):
                #I guess I'm doing this because it's easier than figuring out what the cc_peaks would look like????
                if self.ignore_partial and partial_list[y] > 0.2:
                    print('ignoring partial', n)
                    ccmap = np.zeros((self.limit*self.interp*2, self.limit*self.interp*2))
                    cc_peaks[y] = get_peaks(ccmap, self.n_peaks, return_blank = True)
                    
                elif np.std(query[y]) == 0.:
                    #separate logic checkto save a few ms...
                    ccmap = np.zeros((self.limit*self.interp*2, self.limit*self.interp*2))
                    cc_peaks[y] = get_peaks(ccmap, self.n_peaks, return_blank = True)                
                else:
                    ccmap = ncc(ref[y], query[y], self.limit, self.interp, 
                                ccnorm = self.xcor_normalisation)
                    cc_peaks[y] = get_peaks(ccmap, self.n_peaks)

                if write_pcles:
                    ccmaps.append(ccmap)
            #convert peak coords to shifts from 0
            cc_peaks[:,:,:2] = np.divide(cc_peaks[:,:,:2],
                                        float(self.interp))-float(self.limit)
            tmp_out = join(self.xcor_peak_dir, 'xcor-%0*d_peaks.npy' %
                           (num_digits, pcl_indices[n]))
            np.save(tmp_out, cc_peaks)
            
            if write_pcles:
                ccmap_out = join(self.pcle_dir, 'ccmap_%0*d.mrc'
                           % (num_digits, pcl_indices[n]))        
                write_mrc(ccmap_out, np.array(ccmaps))
            
        #end def ##################################

        padding = self.padding
        pcl_groups = self.ssorted_pcls[0, pcl_indices, 4] #group ids of this subset of particles
        unique_groups = np.array(np.unique(pcl_groups), dtype = int)
        num_digits = len(str(self.ssorted_pcls.shape[1]))
        if write_pcles:
            if not isdir(self.pcle_dir):
                os.makedirs(self.pcle_dir)
        if not isdir(self.xcor_peak_dir):
            os.makedirs(self.xcor_peak_dir)
            
        cc_peaks = np.zeros((self.ssorted_pcls.shape[0], self.n_peaks, 4)) #[number of tilts, number of peaks, 4] (x coord, y coord, peak value, mask)

        #need to generate an array of identifiers for the files in memory
        #because their number does not need to be the same as the number of 
        #non-overlapping groups. This should deal with non-consecutive, e.g.
        #[0,0,3,3]
        ali_arr_ids = np.zeros(len(pcl_groups), dtype = int)
        for x in range(len(unique_groups)):
            ali_arr_ids[pcl_groups == unique_groups[x]] = x
        plotback_arr_ids = deepcopy(ali_arr_ids)
        
        if isinstance(self.query2d, str):
            raise Exception('self.query2d. Expecting list, got str.')
        if np.unique(self.query2d).size == 1:
            #if the initial ali is passed on multiple times
            arr_ali = [deepcopy(mrcfile.open(self.query2d[0], permissive = True).data)]
            ali_arr_ids = np.zeros(len(pcl_groups), dtype = int)
        else:
            arr_ali = [deepcopy(mrcfile.open(self.query2d[gid], permissive = True).data)
                    for gid in unique_groups]
        arr_plotback = [deepcopy(mrcfile.open(self.plotback2d[gid], permissive = True).data)
                        for gid in unique_groups]
        if self.apply_2d_mask:
            arr_mask = [deepcopy(mrcfile.open(self.mask2d[gid], permissive = True).data)
                        for gid in unique_groups]
        else:
            arr_mask = False
    
        if self.no_ctf_convolution:
            defoci = np.zeros((self.ssorted_pcls.shape[0], len(pcl_indices)))
        else:
            defoci = self.ssorted_pcls[:, pcl_indices, 6]
        
        for n in range(len(pcl_indices)):
            ref, query, partial_list = extract_filter_write(n)
            cc(ref, query, partial_list, n)

    def extract_and_process_particles(self, shifts_exist = False, loop = False,
                                      ignore_partial = None):
        """
        IGNORE PARTIAL NEEDS TO BE OFF FOR SYNTHETIC DATA

        Parameters
        ----------
        shifts_exist : TYPE, optional
            DESCRIPTION. The default is False.
        loop : TYPE, optional
            DESCRIPTION. The default is False.
        ignore_partial : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        if isinstance(ignore_partial, type(None)) and isinstance(self.ignore_partial, type(None)):
            self.ignore_partial = True
        elif ignore_partial:
            self.ignore_partial = True
        else:
            self.ignore_partial = False
        
        self.to_json()
        
        self.particles = Extracted_particles(self.ssorted_pcls, 
                                        apix = self.apix, 
                                        out_dir = self.out_dir,
                                        excludelist = [],#self.excludelist, #EEEEEEEEEEHMmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
                                        base_name = self.base_name,
                                        chunk_base = 'xcor',
                                        n_peaks = self.n_peaks,
                                        groups = self.groups,
                                        tilt_angles = self.ssorted_pcls[:, 0, 5], #this way I shouldn't have to worry about excludelist
                                        model_3d = self.model_file,
                                        exclude_worst_pcl_fraction = self.exclude_worst_pcl_fraction,
                                        exclude_lowest_cc_fraction = self.exclude_lowest_cc_fraction)
        
        #reduce number of plots being written with debug = 2        
        if isinstance(self.plot_pcls, (bool, type(None))):
            if self.debug < 3:
                self.plot_pcls = np.linspace(0, self.particles.num_pcls,
                                        min(20, self.particles.num_pcls), dtype = int).tolist()
            else:
                self.plot_pcls = np.arange(self.particles.num_pcls).tolist()
        
        if not shifts_exist or not self.flg_shifts_exist:
            c_tasks = self.particles.split_for_processchunks(self.pcls_per_core)
            for x in range(len(c_tasks)):
                #self._extract_and_cc(np.array(c_tasks[x][:, 0], dtype = int))
                chunk_path = join(self.out_dir, 'xcor-%03d.com' % x)
                out_s = (            
                    '>sys.path = [%s]' % self.path,
                    '>from flexo_pipeline import Flexo',
                    '>f = Flexo(json_attr = "%s")' % self.out_json,
                    '>f._extract_and_cc([%s])' % (',').join(np.array(c_tasks[x][:, 0], dtype = str))
                    )
                with open(chunk_path, 'w') as f:
                    for line in out_s:
                        f.write(line + '\n')
            if loop: #for debugging....   
                print('looping')
                for x in range(len(c_tasks)):
                    self._extract_and_cc(np.array(c_tasks[x][:, 0], dtype = int))
            else:
                run_processchunks('xcor', self.out_dir, self.machines)        
        
        self.particles.read_cc_peaks()
        
        if self.use_median_cc_maps:
            self.particles.shifts_form_median_maps(
                neighbour_distance = self.particle_interaction_distance, 
                min_neighbours = self.min_neighbours,
                z_bias = 3,
                interp = self.interp, limit = self.limit,
                n_peaks = self.n_peaks)
        self.particles.plot_median_cc_vs_tilt()
        self.particles.plot_particle_med_cc()
        #self.particles.plot_shift_magnitude_v_cc(n_peaks = self.n_peaks)
        self.particles.plot_global_shifts()            

        self.particles.pick_shifts_basic_weighting(
            neighbour_distance = self.particle_interaction_distance,
            n_peaks = self.n_peaks,
            cc_weight_exp = 5,
            plot_pcl_n = self.plot_pcls,
            min_neighbours = self.min_neighbours,
            shift_std_cutoff = self.shift_std_cutoff,
            use_nbr_median = self.use_nbr_median)    
        self.out_fid = self.particles.write_fiducial_model(self.ali, use_local_medians = self.use_local_median_shifts)
        
        self.flg_shifts_exist = True
        self.to_json()
    
    def combine_fiducial_models(self):
        
        if self.use_existing_fid:
            #by default, existing fid is ignored if it's a patch tracked model
            with open(join(self.rec_dir, self. imod_base_name + '.edf')) as edf:
                for line in edf.readlines():
                    if line.startswith('ScreenState.FidModel.bn.track-patches.done=true'):
                        self.use_existing_fid = False
                    else:
                        i3dmod = join(self.rec_dir, self. imod_base_name + '.3dmod')
        
            self.add_fiducial_model.append(i3dmod)        
            self.use_existing_fid = False
        
        if self.add_fiducial_model:
            all_fid = []
            for m in self.use_existing_fid:
                out_fid = join(self.out_dir, 'reprojected_' + split(m)[-1])
                all_fid.append(out_fid)
                self.reproject_model(self, model_file = m, out_fid = out_fid)
        
            combined_fid = join(self.out_dir, self.base_name + '_combined.fid')
            check_output('imodjoin %s %s' % ((' ').join(all_fid), combined_fid), shell = True)
            self.out_fid = combined_fid
        self.to_json()

    
    def reconstruct_tomo(self, binning, out_ext = ''):

        pwd = os.getcwd()
        os.chdir(self.out_dir) #this is needed for splittilit
        
        out_base = join(self.out_dir, self.base_name)
        if self.tiltcom.namingstyle == 1:
            output_ali = out_base + out_ext + '_ali.mrc'
            ctf_ali = out_base + out_ext + '_ctfcorr_ali.mrc'
            output_rec = out_base + out_ext + '_full_rec.mrc'
            rotated_rec = out_base + out_ext + '_rec.mrc'
        else: 
            output_ali = out_base + out_ext + '.ali'
            ctf_ali = out_base + out_ext + '_ctfcorr.ali'
            output_rec = out_base + out_ext + '_full.rec'
            rotated_rec = out_base + out_ext + '.rec'
            
        self.format_newst(binning, output_ali = output_ali)
        self.format_ctfcom(input_ali = output_ali, output_ali = ctf_ali)
        self.format_tilt(binning, output_rec = output_rec,
                         input_ali = output_ali)

 
        imodscript('newst.com', self.out_dir)
        if not isfile(output_ali):
            raise Exception('newst.com has failed.')
        if len(self.machines)  > 1:
            if self.no_ctf_convolution:
                warnings.warn('Ctfcorrection disabled!')
            else:
                check_output('splitcorrection -m %s ctfcorrection.com' % 
                             max(1, int(np.floor(self.st_size[2]/len(self.machines)))),
                             shell = True)
                run_processchunks('ctfcorrection', self.out_dir, self.machines)
                os.rename(ctf_ali, output_ali)
            #reconstruct
            check_output('splittilt -n %s tilt.com' % len(self.machines), shell = True)
            run_processchunks('tilt', self.out_dir, self.machines)
            check_output("clip rotx %s %s" % (output_rec, rotated_rec), shell = True)
        else:
            if self.no_ctf_convolution:
                warnings.warn('Ctfcorrection disabled!')
            else:            
                imodscript('ctfcorrection.com', self.out_dir) 
                os.rename(ctf_ali, output_ali)
            imodscript('tilt.com', self.out_dir) 
            if not isfile(output_rec):
                raise Exception('tilt.com has failed.')
            check_output("clip rotx %s %s" % (output_rec, rotated_rec), shell = True)

        os.chdir(pwd)
        return rotated_rec

    def make_all_tomos(self):
        
        self.format_align(self.tomo_binning, out_dir = self.out_dir)
        #self.aligncom.dict['AngleOffset'] = 0 #not when using .rawtlt!!!
        imodscript('align.com', self.out_dir)
        
        if self.peet_binning != self.tomo_binning:
            self.peet_tomo = self.reconstruct_tomo(self.peet_binning, out_ext = '_bin%s' % int(self.peet_binning))
        #one could bin files and update comscripts here to save a reasonable amount of runtime
        self.out_tomo = self.reconstruct_tomo(self.tomo_binning)
        if self.peet_binning == self.tomo_binning:
            self.peet_tomo = self.out_tomo
        
        self.flg_tomos_exist = True
        self.to_json()
        
    def run_peet(self):  
        if not self.prm and not (self.prm1 and self.prm2):
            print('PEET prm file not specified, PEET will not run.')
        else:
            if self.curr_iter == 1:
                if self.fsc_dirs: #meaning this run is being restarted, in which case skip init_peet
                    self.fsc_dirs = [self.init_peet_dir]
                else:
                    self.peet_apix = self.prep_peet(self.init_peet_dir, 'init')
                    peet_halfmaps(self.init_peet_dir, self.prm1, self.prm2, self.machines, use_davens_fsc = self.use_davens_fsc)
                    self.fsc_dirs.append(self.init_peet_dir)
            
            self.peet_apix = self.prep_peet(self.peet_dir, self.peet_tomo)
            
            self.flg_peet_ran = True # this needs to be here in case peet crashes and a restart is attempted
            self.to_json()
            
            peet_halfmaps(self.peet_dir, self.prm1, self.prm2, self.machines, use_davens_fsc = self.use_davens_fsc)
            self.fsc_dirs.append(self.peet_dir)

        self.res = plot_fsc(self.fsc_dirs, self.peet_dir, self.cutoff, self.peet_apix)
        self.to_json()
        
    def match_tiny_tomos(self, output_binning = False):
        
        if not output_binning:
            output_binning = min(8, self.tomo_binning*2) # there is a risk corrsearch3d will crap out with (no patches fit or smh) if the tomo is too small
            #obv there is a chance that the input volume is too small (e.g. toy dataset). not sure what the min size in pixels is
        if not isdir(self.match_tomos_dir):
            os.makedirs(self.match_tomos_dir)
            
        self.format_align(self.tomo_binning, out_dir = self.match_tomos_dir) #binning should be the fiducial model binning
        imodscript('align.com', self.match_tomos_dir)
        tmp_align = IMOD_comfile(self.match_tomos_dir, 'align.com')
        if not isfile(tmp_align.dict['OutputTransformFile']):
            raise Exception('align.com has failed.')
        
        ref_ali = join(self.match_tomos_dir, 'ref.ali')
        self.format_newst(output_binning, use_rec_dir = True,
                          change_comname = 'ref_newst.com',
                          out_dir = self.match_tomos_dir,
                          output_ali = ref_ali)
        self.format_tilt(output_binning, output_rec = 'ref_full.mrc',
                         input_ali = ref_ali,
                          use_rec_dir = True, out_dir = self.match_tomos_dir,
                          change_comname = 'ref_tilt.com')

        self.format_newst(output_binning, use_rec_dir = False,
                          change_comname = 'query_newst.com',
                          out_dir = self.match_tomos_dir)  
        self.format_tilt(output_binning, output_rec = 'query_full.mrc',
                          use_rec_dir = False, out_dir = self.match_tomos_dir,
                          change_comname = 'query_tilt.com')
        
        # #this will likely break if the original transform file does not end in .xf
        # new_xf = join(self.out_dir, self.base_name + '.xf')
        # new_tlt = join(self.out_dir, self.base_name + '.tlt')
        # if isfile(new_xf):
        #     copyfile(new_xf, join(self.match_tomos_dir, split(new_xf)[1]))
        # else:
        #     raise Exception('%s does not exist.')
        # if isfile(new_tlt):
        #     copyfile(new_tlt, join(self.match_tomos_dir, split(new_tlt)[1]))
        
        imodscript('ref_newst.com', self.match_tomos_dir)
        imodscript('query_newst.com', self.match_tomos_dir)
        SHIFT, OFFSET, global_xtilt = match_tomos('ref_tilt.com', 'query_tilt.com',
                                                  self.match_tomos_dir,
                niters = 3, angrange = 20, plot = True)
        
        #keep these offsets in tilt and let align.com angle offsets be separate..
        self.aligncom.dict['AngleOffset'] += OFFSET
        self.tiltcom.dict['SHIFT'] = SHIFT
        self.tiltcom.dict['XAXISTILT'] = global_xtilt
    
    def prep_peet(self, peet_dir, tomo ):#can be 'init'
        
        if not isdir(peet_dir):
            os.makedirs(peet_dir)
        os.chdir(peet_dir)
        fsc1d = join(peet_dir, 'fsc1/')
        fsc2d = join(peet_dir, 'fsc2/') 
        if not os.path.isdir(fsc1d):
            os.makedirs(fsc1d)
        if not os.path.isdir(fsc2d):
            os.makedirs(fsc2d)  

        #first format parent .prm before splitting it for fsc
        #self.prm needs to be updated because self.prm_tomogram_number may be selecting a subset of model files
        if self.prm and not (self.prm2 and self.prm1):
            self.prm, peet_apix = prepare_prm(
                    self.prm, self.ite, tomo, self.prm_tomogram_number,
                    self.out_dir, self.base_name, peet_dir,
                    search_rad = self.search_rad,
                    phimax_step = self.phimax_step,
                    psimax_step = self.psimax_step,
                    thetamax_step = self.thetamax_step)
            self.prm_tomogram_number = 1 # 
            r_new_prm = PEETPRMFile(self.prm)
            print('Splitting PEET run for FSC.')
            r_new_prm.split_by_classID(0, fsc1d,
                            classes = [1], splitForFSC = True, writeprm = True)
            r_new_prm.split_by_classID(0, fsc2d,
                            classes = [2], splitForFSC = True, writeprm = True)  
            self.prm1 = join(fsc1d, self.base_name + '_fromIter%s_cls1.prm' % 0)
            self.prm2 = join(fsc2d, self.base_name + '_fromIter%s_cls2.prm' % 0)
            
            self.prm = False
            
        elif self.prm1 and self.prm2: 
            #if PEET was already split for FSC
            # if not self.prm1: #first iteration, prm2 was specified
            #     self.prm1 = self.prm
            self.prm1, peet_apix = prepare_prm(
                    self.prm1, self.ite, tomo, self.prm_tomogram_number,
                    self.out_dir, self.base_name, fsc1d,
                    search_rad = self.search_rad,
                    phimax_step = self.phimax_step,
                    psimax_step = self.psimax_step,
                    thetamax_step = self.thetamax_step)

            self.prm2, peet_apix2 = prepare_prm(
                    self.prm2, self.ite, tomo, self.prm_tomogram_number,
                    self.out_dir, self.base_name, fsc2d,
                    search_rad = self.search_rad,
                    phimax_step = self.phimax_step,
                    psimax_step = self.psimax_step,
                    thetamax_step = self.thetamax_step)
            if peet_apix != peet_apix2:
                raise ValueError('The two PEET half data-sets do not have the'
                                 + ' same binning.')

        return peet_apix
    
    def iterate(self, curr_iter = False, num_iterations = False):

        #should be possible to restart from any point of any iteration
        #restart technically onle needs curr_iter and out_dir
        #remake tomos - IMOD control by changing parameters in the previous iteration com files
        
        #def modify_iteration_inputs():
            
            
        def restart():
            init_flags = ['flg_inputs_verified', 'flg_image_data_exist',
                          'flg_shifts_exist', 
                          'flg_tomos_exist', 'flg_peet_ran']
            
            tmp_dict = deepcopy(self.__dict__)
            self.out_dir = join(self.orig_out_dir, 'iteration_%s' % self.curr_iter)
            self.update_dirs()
            if not os.path.isdir(self.out_dir):
                raise Exception('Iteration_%s directory does not exist.' % self.curr_iter)
            else:
                #input params are ignored when restarting, except for 
                self.__init__(json_attr = join(self.out_dir, 'flexo.json'))
                
            #setting PEET to intitial.................
            print('DEV NOTE PEET set to initial on restarting')
            if self.flg_peet_ran: 
                prev_peet_dir = self.out_dir[:-1] + str(self.curr_iter - 1) + '/peet'
                prev_fsc1 = prev_peet_dir + '/fsc1'
                prev_fsc2 = prev_peet_dir + '/fsc2'
                if isdir(prev_peet_dir):
                    self.prm = join(prev_peet_dir, split(self.prm[-1]))
                    self.prm1 = join(prev_fsc1, split(self.prm1[-1]))
                    self.prm2 = join(prev_fsc2, split(self.prm2[-1]))
                else: 
                    self.prm = tmp_dict['prm']
                    self.prm1 = tmp_dict['prm1']
                    self.prm2 = tmp_dict['prm2']

            for key in init_flags:
                if not isinstance(tmp_dict[key], type(None)):
                    self.__dict__[key] = tmp_dict[key]
            
            #if self.flg_aligncom_formatted and not self.flg_tomos_exist:
            print('DEV NOTE align might not be pointing to the right files here, check')
            if self.flg_tomos_exist:
                self.aligncom = IMOD_comfile(self.out_dir, 'align.com') # allow manual modification and re-run.

        def prebin():
            self.curr_iter = 0
            self.out_dir = join(self.orig_out_dir, 'iteration_0')
            self.update_dirs()            
            self.verify_inputs()
            self.out_fid = self.aligncom.dict['ModelFile']
            self.format_align(self.pre_bin)
            self.format_newst(self.pre_bin, use_rec_dir = True)
            tmp_newst = IMOD_comfile(self.out_dir, 'newst.com')
            self.format_tilt(self.pre_bin, use_rec_dir = True, input_ali = tmp_newst.dict['OutputFile'])
            tmp_tilt = IMOD_comfile(self.out_dir, 'tilt.com')
            imodscript('newst.com', self.out_dir)
            self.format_ctfcom()
            tmp_ctf = IMOD_comfile(self.out_dir, 'ctfcorrection.com')
            if isfile(tmp_ctf.dict['AngleFile']):
                os.rename(tmp_ctf.dict['AngleFile'], tmp_ctf.dict['AngleFile'] + '~')
            copyfile(self.ctfcom.dict['AngleFile'], tmp_ctf.dict['AngleFile'])
            if 'TransformFile' in tmp_ctf.dict:
                if isfile(tmp_ctf.dict['TransformFile']):
                    os.rename(tmp_ctf.dict['TransformFile'], tmp_ctf.dict['TransformFile'] + '~')
                copyfile(self.ctfcom.dict['TransformFile'], tmp_ctf.dict['TransformFile'])
            # if isfile(tmp_newst.dict['TransformFile']):
            #     os.rename(tmp_newst.dict['TransformFile'], tmp_newst.dict['TransformFile'] + '~')
            # copyfile(self.newstcom.dict['TransformFile'], tmp_newst.dict['TransformFile'])
            if not self.no_ctf_convolution:
                imodscript('ctfcorrection.com', self.out_dir)
                os.rename(tmp_ctf.dict['OutputFileName'],  tmp_ctf.dict['InputStack'])
                    
            imodscript('tilt.com', self.out_dir)

            if self.tiltcom.namingstyle == 1:
                rotated_rec = join(self.out_dir, self.base_name + '_rec.mrc')
            else: 
                rotated_rec = join(self.out_dir, self.base_name + '.rec')   
            check_output("clip rotx %s %s" % (tmp_tilt.dict['OutputFile'], rotated_rec), shell = True)
            
            if self.pre_filter:
                tmp_rec = rotated_rec + '~'
                if isfile(tmp_rec):
                    os.remove(tmp_rec)
                check_output('mtffilter -3 -hi %s,%s -lo %s,%s %s %s' % (
                             self.pre_filter[0], self.pre_filter[1],
                             self.pre_filter[2], self.pre_filter[3],
                             rotated_rec, tmp_rec), shell = True)
                os.rename(tmp_rec, rotated_rec)
            if not self.keep_peet_binning:
                self.peet_apix = self.prep_peet(self.peet_dir, rotated_rec)
            self.out_tomo = rotated_rec
                
            self.to_json()
            
        ###################################################################
                
        if not num_iterations:
            if not self.num_iterations:
                self.num_iterations = 2
        else:
            self.num_iterations = num_iterations

        if not curr_iter:
            if not self.curr_iter:
                self.curr_iter = 1
            else:
                restart()      
        else:
            self.curr_iter = curr_iter
            restart()

        while self.curr_iter <= self.num_iterations:
            startTime = time.time()
            if self.pre_bin:
                prebin()
                self.pre_bin = False
            else:
                print('\nStarting iteration %s\n' % self.curr_iter)
                if not self.flg_inputs_verified:
                    print('Verifying inputs...')
                    self.verify_inputs()
                if not self.flg_image_data_exist:
                    print('Generating tilt series...')
                    self.generate_image_data()
                if not self.flg_shifts_exist:
                    print('Measuring shifts...')
                    self.extract_and_process_particles()
                if not self.flg_tomos_exist:
                    print('Reconstructing tomograms...')
                    self.combine_fiducial_models()
                    self.match_tiny_tomos()
                    self.make_all_tomos()
                if not self.flg_peet_ran:
                    if self.prm or (self.prm1 and self.prm2):
                        print('Running PEET...')
                        self.run_peet()
                    
            # if (self.prm1 and self.prm2) and self.use_refined_halfmap_models:
            #     self.model_file, self.motl = combine_fsc_halves(self.prm1, self.prm2,
            #                     self.prm_tomogram_number, self.peet_dir, 2)
            # elif self.prm:
            #     print('DEV NOTE there must be strictly one particle model file for the tomogram')
            #     self.prm, peet_apix = prepare_prm(
            #         self.prm, self.ite, self.out_tomo, self.prm_tomogram_number,
            #         self.out_dir, self.base_name, self.peet_dir,
            #         search_rad = self.search_rad,
            #         phimax_step = self.phimax_step,
            #         psimax_step = self.psimax_step,
            #         thetamax_step = self.thetamax_step)
            
            print('Iteration execution time: %s s.' % int(np.round(
                                (time.time() - startTime), decimals = 0)))    
            #finishing touches, updating for next iteration
            self.flg_inputs_verified = False
            self.flg_image_data_exist = False
            self.flg_shifts_exist = False
            #self.flg_aligncom_formatted = False
            self.flg_tomos_exist = False
            self.flg_peet_ran = False
            
            self.model_file_binning = None 
            self.rec_dir = self.out_dir
            self.out_dir = join(self.orig_out_dir, 'iteration_%s' % (self.curr_iter + 1))
            self.tomo = self.out_tomo
            self.update_dirs()
            self.update_comfiles()
            #check resolution improved
            
            #deal with rawtlt? otehr align files
            
            if not self.ignore_resolution_check and (self.prm1 and self.prm2):
                if curr_iter > 0:
                    get_area = True 
                    res = np.array(self.res)
                    r = np.array(res)[:,1]
                    if np.any(np.floor(r) == np.floor(self.peet_apix*2)) or get_area:
                        #in case resolution = resolution at Nyquist, use area under FSC
                        areas = np.round(res[:,2], decimals = 3)
                        if not get_area:
                            print(('Estimated resolution too close to Nyquist. ' + 
                                   'Using area under FSC instead.'))
                        print(('Areas under FSC of initial PEET, iterations %s: %s' % 
                           (np.arange(1,self.curr_iter + 1), (',').join([str(x) for x in areas]))))
                        if areas[-1] <= areas[-2]:
                            print(('No apparent improvement since last iteration.'
                                   + '  Exiting...'))
                            break
                    else:
                        print(('PEET resolution estimates of iterations %s: %s' % 
                               (np.arange(1,self.curr_iter + 1), (',').join([str(x) for x in r]))))
                        if r[-1] >= r[-2]:
                            print(('No apparent improvement since last iteration.'
                                   + '  Exiting...'))
                            break
            
            self.curr_iter += 1      

    