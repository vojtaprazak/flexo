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
from flexo_tools import (get_apix_and_size, find_nearest,
                         
                         
                         optimal_lowpass_list)
from flexo_tools import match_tomos
from flexo_tools import (machines_from_imod_calib, 
                         optimal_lowpass,
                         )
#from PEETModelParser import PEETmodel
#from PEETPRMParser import PEETPRMFile 
#from scipy.spatial import KDTree
from IMOD_comfile import IMOD_comfile
import json
import multiprocessing
import socket
#import matplotlib.pyplot as plt
#import mrcfile
import warnings 


from extracted_particles import Extracted_particles
from flexo_peet_prm import Flexo_peet_prm
from flexo_tools import (run_generic_process, run_processchunks, check_ssh, imodscript, get_binned_size)
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
                 mol_1_ite = 999, cutoff = 0.143, get_area = True,
                 peet_search = 'shift_only',
                 #dev
                 xcor_normalisation = 'none',
                 padding = 10,
                 keep_peet_binning = True,
                 use_refined_halfmap_models = True,
                 noisy_ref_std = 5,
                 apply_2d_mask = True,
                 exclude_worst_pcl_fraction = 0.1,
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
            if self.flg_mol_obj_initiated:
                self.m_objs = [Flexo_peet_prm(json_attr = j) for j in self.m_objs]
            
        else:
            
            for kwkey in kwargs.keys():
                self.__dict__[kwkey] = kwargs[kwkey] #here to catch mol_x_y
                
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
            self.out_fid = [] #new fiducial model with calculated shifts
            
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


            #sorting particles, np arrays
            self.groups = kwargs.get('groups')
            self.sorted_pcls = kwargs.get('sorted_pcls')
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
            self.flg_mol_obj_initiated = kwargs.get('flg_mol_obj_initiated')
            self.flg_image_data_exist = kwargs.get('flg_image_data_exist')
            self.flg_shifts_exist = kwargs.get('flg_shifts_exist')
            #self.flg_aligncom_formatted = kwargs.get('flg_aligncom_formatted')
            self.flg_tomos_exist = kwargs.get('flg_tomos_exist')
            self.flg_peet_ran = kwargs.get('flg_peet_ran')
            


            #tomogram parameters
            #binning
            self.tomo_binning = kwargs.get('tomo_binning') #flexo runs at this binning
            self.peet_binning = kwargs.get('peet_binning') #peet binning doesnt affect tomo_binning. A separate "peet binned" tomo is made

            self.keep_peet_binning = keep_peet_binning #if false, change peet binning to match flexo binning
            
            self.peet_apix = kwargs.get('peet_apix') #
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
            self.m_objs = [] #list of objects storing prm/models for each particle type. overwritten at each iteration

            #each particle type can be specified by mol_[x]_[attribute]
            #any Flexo_peet_prm attribute can be modified this way. 
            #Notable params [prm, prm1, prm2, tomogram_number, average_volume, ite, box_size]
            self.mol_1_prm = kwargs.get('mol_1_prm')
            self.mol_1_prm1 = kwargs.get('mol_1_prm1') 
            self.mol_1_prm2 = kwargs.get('mol_1_prm2') #for gold standard
            self.mol_1_tomogram_number = kwargs.get('mol_1_tomogram_number') #numbered from one (for PEET consistency)
            self.mol_1_average_volume = kwargs.get('mol_1_average_volume')
            self.mol_1_ite = mol_1_ite #peet iteration, default last 


            
            #OR
            self.mol_1_model_file = kwargs.get('mol_1_model_file') #initial 3d model file. read from prm(s) if specified
            self.mol_1_motl = kwargs.get('mol_1_motl') #initial motive list, matching model_File. read from prm(s) if specified
            self.mol_1_model_file_binning = kwargs.get('mol_1_model_file_binning') #this is only necessary if model file is specified instead of prm, and is different binning from input tomogram binning
        
            #peet parameters.
            self.peet_search = kwargs.get('peet_search') #'shift_only', 'use_original_params', 'manual'
  
#            #no need to have these written out here
#            self.mol_1_full_model_file = kwargs.get('mol_1_full_model_file') #generated during model reprojection, fits unrotated full.rec
            
            #MOVING TO FLEXO_PEET_PRM
            #self.reprojected_mod = kwargs.get('reprojected_mod') #3d model projected to 2d fiducial model
#            self.fid_list = [] #split reprojected_mod
            
#            self.split_motls = [] #split motl, matching split3d_models
#            self.split3d_models = [] #3d pcl model after non_overlapping_pcles, in rotated orientation
            
            
            
            self.cutoff = cutoff #fsc cutoff
            self.get_area = get_area #use area under fsc instead of resolution
            
            #DEV
#            self.smooth = kwargs.get('smooth')
#            self.unreasonably_harsh_filter = kwargs.get('unreasonably_harsh_filter')
            self.no_ctf_convolution = kwargs.get('no_ctf_convolution') #this is for running toy data where no ctf convolution should be applied
#            self.poly = kwargs.get('poly')
#            self.poly_order = kwargs.get('poly_order')
#            self.smooth_ends = kwargs.get('smooth_ends')
            
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
            self.pre_filter = kwargs.get('pre_filter') # 4 values for mtffilter -hi cutoff,falloff, -lo cutoff,falloff, list. Helps with noisy_ref = False
            self.noisy_ref_std = noisy_ref_std
            self.exclude_worst_pcl_fraction = exclude_worst_pcl_fraction #exclude particles with the worst median ccc. 0/False disables this
#            self.exclude_lowest_cc_fraction = exclude_lowest_cc_fraction #Removes the worst scoring shifts. NOT RECOMMENDED # gone
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
        sorted_pcls
                    [number of tilts:model points per tilt:7
                0=xcoords, 1=ycoords, 2=tilt number, 3=particle index, 4=group id (from 0), 5=tilt angle, 6=defocus)]
                group id == len(groups) indicates unallocated pcls   
        """

    def to_json(self):

        if not isdir(self.npy_dir):
            os.makedirs(self.npy_dir)
        out_inst = deepcopy(self)
        
        #replace class obj with json paths
        if out_inst.m_objs:
            out_inst.m_objs = [o.out_json for o in out_inst.m_objs]
        
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
        #self.pcle_dir = join(self.out_dir, 'extracted_particles')
        #self.xcor_peak_dir = join(self.out_dir, 'xcor_peaks')
        self.match_tomos_dir = join(self.out_dir, 'match_tomos')
        #self.init_peet_dir = join(self.out_dir, 'init_peet')
        #self.peet_dir = join(self.out_dir, 'peet')

    def update_comfiles(self):
        self.tiltcom = IMOD_comfile(self.rec_dir, 'tilt.com')
        self.newstcom = IMOD_comfile(self.rec_dir, 'newst.com')
        self.aligncom = IMOD_comfile(self.rec_dir, 'align.com')
        self.ctfcom = IMOD_comfile(self.rec_dir, 'ctfcorrection.com')
        
    def initiate_molecules(self):
        """
        Initiate a list of objects, one for each molecule type.
        
        Different molecule types are specified by numbered attribute names:
        mol_[molecule number]_attribute.
        
        Because all kwargs are dumped to attributes, legacy parameters (i.e.
        without the mol_x_ part) should work.
        
        The minimum requirements are prm or prm1+prm2 or model_file+motl
        
        """
        self.m_objs = []
        
        mol_keys = np.array([k for k in self.__dict__.keys() if k.startswith('mol_')])
        m_ids = list(np.unique([int(s.split('_')[1]) for s in mol_keys]))
        
        
        
        for mol in m_ids:
            mol_mask = [k.startswith('mol_%s' % mol) for k in mol_keys]
            tmp_mol_keys = mol_keys[mol_mask]
            translated_keys = [k.replace('mol_%s_' % mol, '') for k in tmp_mol_keys]
            tmp_dict = {j: self.__dict__[k] for k, j in zip(tmp_mol_keys, translated_keys)}
            
            #other necessary attributes
            other_key_mask = np.logical_not([k.startswith('mol_') for k in self.__dict__.keys()])
            other_keys = np.array(list(self.__dict__.keys()))[other_key_mask]
            other_dict = {k: self.__dict__[k] for k in other_keys}
           
            #peet params
            if self.peet_search == 'shift_only' or self.peet_search == 'use_original_params':
                if self.peet_search == 'shift_only' and self.search_rad:
                    other_dict['search_rad'] = self.search_rad
                else:
                    other_dict['search_rad'] = False
                other_dict['phimax_step'] = False
                other_dict['psimax_step'] = False
                other_dict['thetamax_step'] = False
            if self.peet_search == 'manual':
                #mol_x_search_rad etc shuld be converted to search_rad. 
                #replace with original prm if one of these doesn't exist
                for key in ['search_rad', 'phimax_step', 'psimax_step', 'thetamax_step']:
                    if key not in other_dict:
                        other_dict[key] = False

            tmp_dict = {**tmp_dict, **other_dict}
            
            tmp_obj = Flexo_peet_prm(mol, **tmp_dict)
            self.m_objs.append(tmp_obj)
            
        self.flg_mol_obj_initiated = True
        self.to_json()
        
    def export_mol_attr(self):
        for mol_id in range(len(self.m_objs)):
            if self.m_objs[mol_id].prm1:
                #a sinle prm would have been split into halves so only need to check this
                self.__dict__['mol_%s_prm1' % (mol_id + 1)] = self.m_objs[mol_id].prm1
                self.__dict__['mol_%s_prm2' % (mol_id + 1)] = self.m_objs[mol_id].prm2
                self.__dict__['mol_%s_tomogram_number' % (mol_id + 1)] = 1
                self.__dict__['mol_%s_ite' % (mol_id + 1)] = 0
            else:
                self.__dict__['mol_%s_model_file' % (mol_id + 1)] = self.m_objs[mol_id].model_file
                self.__dict__['mol_%s_motl' % (mol_id + 1)] = self.m_objs[mol_id].motl
            self.__dict__['mol_%s_model_file_binning' % (mol_id + 1)] = None#subsequent iters use correctly binned model

        
    def verify_inputs(self):
        """
        base_name is read from imod com files if not specifie
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
        #self.reprojected_mod = join(self.out_dir, self.base_name + '_reprojected.fid')
        if not self.tomo:
            if self.full_tomo.endswith('_full.rec'):
                ext = '_full.rec'
                self.tomo = self.full_tomo[:-len(ext)] + '.rec'
            elif self.full_tomo.endswith('_full_rec.mrc'):
                ext = '_full_rec.mrc'
                self.tomo = self.full_tomo[:-len(ext)] + '_rec.mrc'
        
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
                raise Exception('File not found %s. Run with pre_bin = True' % self.full_tomo)
                
           
        # check binning ##############
        st_apix, stack_size = get_apix_and_size(self.st)
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
        self.st_apix = float(st_apix)
        self.tomo_binning = float(np.round(tomo_apix/st_apix, decimals = 0))
        if self.pre_bin:
            #pre_bin ran within iterate
            if isinstance(self.pre_bin, bool):
                self.pre_bin = self.tomo_binning
                print('Remaking initial tomograms without binning...')
            else:
                self.tomo_binning = self.pre_bin
            
        if not self.particle_interaction_distance:
            self.particle_interaction_distance = float(min(1000./self.apix, np.sort(self.tomo_size)[1]//16)) #ballpark....
 
        tilt_angles = [float(x.strip()) for x in open(self.tlt, 'r')]
        if len(tilt_angles) != ali_size[2]:     
            raise ValueError((
                """The number of tilt angles in tilt file (%s)
                does not match the number of images in aligned stack (%s).
                This could mean that there is "ExcludeSections" entry in
                newst.com. This is not currently supported, excludelist
                should be in align.com or tilt.com.""")
                             % (len(tilt_angles), ali_size[2]))   
                    
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
            if not isinstance(self.sorted_pcls, (bool, type(None))):
                mag_ref = find_nearest(self.sorted_pcls[:, 0, 5], 0) + 1
            else:
                mag_ref = self.st_size[2]//2
        else:
            mag_ref = self.aligncom.dict['MagReferenceView']

        if self.particles:
            self.fidn = [min(j, self.particles.num_pcls - 1) for j in self.fidn]    
    
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
        
        self.peet_binning = self.m_objs[0].peet_binning
        
        if self.peet_binning != self.tomo_binning:
            self.peet_tomo = self.reconstruct_tomo(self.peet_binning, out_ext = '_bin%s' % int(self.peet_binning))
        #one could bin files and update comscripts here to save a reasonable amount of runtime
        self.out_tomo = self.reconstruct_tomo(self.tomo_binning)
        if self.peet_binning == self.tomo_binning:
            self.peet_tomo = self.out_tomo
            
        for mol_id in range(len(self.m_objs)):
                self.m_objs[mol_id].peet_tomo = self.peet_tomo
        
        self.flg_tomos_exist = True
        self.to_json()
        
    # def run_peet(self):  
    #     if not self.mol_1_prm and not (self.mol_1_prm1 and self.mol_1_prm2):
    #         print('PEET prm file not specified, PEET will not run.')
    #     else:
    #         if self.curr_iter == 1:
    #             if self.fsc_dirs: #meaning this run is being restarted, in which case skip init_peet
    #                 self.fsc_dirs = [self.init_peet_dir]
    #             else:
    #                 self.peet_apix = self.prep_peet(self.init_peet_dir, 'init')
    #                 peet_halfmaps(self.init_peet_dir, self.mol_1_prm1, self.mol_1_prm2, self.machines, use_davens_fsc = self.use_davens_fsc)
    #                 self.fsc_dirs.append(self.init_peet_dir)
            
    #         self.peet_apix = self.prep_peet(self.peet_dir, self.peet_tomo)
            
    #         self.flg_peet_ran = True # this needs to be here in case peet crashes and a restart is attempted
    #         self.to_json()
            
    #         peet_halfmaps(self.peet_dir, self.mol_1_prm1, self.mol_1_prm2, self.machines, use_davens_fsc = self.use_davens_fsc)
    #         self.fsc_dirs.append(self.peet_dir)

    #     self.res = plot_fsc(self.fsc_dirs, self.peet_dir, self.cutoff, self.peet_apix)
    #     self.to_json()
    
    
            
        
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
     
            
    def merge_m_objs(self):
        """
        merge flexo_peet_prm objects AFTER flexo_peet_prm.generate_image_data
        
        combine plotback but keep particle models separate 

        """

        if len(self.m_objs) > 1:
            peet_binning = np.unique([mol_obj.peet_binning for mol_obj in self.m_objs])
            if len(peet_binning) > 1:
                raise Exception('Binning of PEET prm projects does not match')
            else:
                self.peet_binning = float(peet_binning)
            #tmp_obj = deepcopy(self.m_objs[0])
            merged_plotback = [join(self.out_dir, self.base_name + '_plotback2D_%02d.mrc' % 0)]
            merged_query = [join(self.out_dir, self.base_name + '_query2D_%02d.mrc' % 0)]            
            
            part_plotback = []
            part_query = []
            
            for mol_id in range(len(self.m_objs)):
                part_plotback.extend(self.m_objs[mol_id].plotback2d)
                part_query.extend(self.m_objs[mol_id].query2d)
                
                self.m_objs[mol_id].plotback2d = merged_plotback
                self.m_objs[mol_id].query2d = merged_query
            
            cmd = ['clip', 'add']
            run_generic_process(cmd  + part_plotback + merged_plotback)
            run_generic_process(cmd  + part_query + merged_query)
            print('DEV NOTE query cannot always be processed this way') #standard is fine, not noisy ref, 
            
            
    def make_ts(self):
        
        for mol_id in range(len(self.m_objs)):
            if not self.m_objs[mol_id].flg_image_data_exist:
                self.m_objs[mol_id].generate_image_data()
        if not self.non_overlapping_pcls:
            self.merge_m_objs()
            
        for mol_id in range(len(self.m_objs)):
            self.m_objs[mol_id].extract_and_process_particles()
            
        self.flg_image_data_exist = True
        self.to_json()
        
        
    def merge_extracted_pcles(self):
        self.particles = deepcopy(self.m_objs[0].particles)
        if len(self.m_objs) > 1:

            self.particles.sorted_pcls = []
            self.particles.pcl_ids = []
            for m in range(len(self.m_objs)):
                #renumber particles
                tmp_ids = self.m_objs[m].particles.pcl_ids
                tmp_pcls = self.m_objs[m].particles.sorted_pcls
                if m > 0:
                    tmp_ids += self.m_objs[m - 1].particles.num_pcls
                    tmp_pcls[:, :, 3] = tmp_ids[None]
                self.particles.pcl_ids.extend(tmp_ids)
                self.particles.sorted_pcls.append(tmp_pcls)
            self.particles.sorted_pcls = np.hstack(self.particles.sorted_pcls)
                
            #self.particles.sorted_pcls = np.hstack([mol_obj.particles.sorted_pcls for mol_obj in self.m_objs])
            #self.particles.pcl_ids = np.concatenate([mol_obj.particles.pcl_ids for mol_obj in self.m_objs])
            self.particles.num_pcls = np.sum([mol_obj.particles.num_pcls for mol_obj in self.m_objs])
            self.particles.groups = np.hstack([mol_obj.particles.groups for mol_obj in self.m_objs])
            self.particles.shifts = np.hstack([mol_obj.particles.shifts for mol_obj in self.m_objs])
            self.particles.shift_mask = np.hstack([mol_obj.particles.shift_mask for mol_obj in self.m_objs])
            self.particles.cc_values = np.hstack([mol_obj.particles.cc_values for mol_obj in self.m_objs])
            self.particles.model_3d = np.vstack([mol_obj.particles.model_3d for mol_obj in self.m_objs])

        self.particles.out_dir = self.out_dir
        
            # self.particles.sorted_pcls = particles.sorted_pcls[:, [0,5,1,6,2,7,3,8,4]] #
            # self.particles.groups = particles.groups[:, [0,5,1,6,2,7,3,8,4]] 
            # self.particles.shifts = particles.shifts[:, [0,5,1,6,2,7,3,8,4]]
            # self.particles.shift_mask = particles.shift_mask[:, [0,5,1,6,2,7,3,8,4]]
            # self.particles.cc_values = particles.cc_values[:, [0,5,1,6,2,7,3,8,4]]
            # self.particles.model_3d = particles.model_3d[[0,5,1,6,2,7,3,8,4]]
            # self.particles.pcl_ids = np.concatenate([mol_obj.particles.pcl_ids for mol_obj in self.m_objs])
            # 
    
    def process_shifts(self):
        """
        Parameters
        ----------
        particle_objs : list
            list of Extracted_particles objects.

        """

        self.merge_extracted_pcles()
        
        self.particles.plot_median_cc_vs_tilt()
        self.particles.plot_particle_med_cc()
        #self.particles.plot_shift_magnitude_v_cc(n_peaks = self.n_peaks)
        self.particles.plot_global_shifts()

        self.particles.pick_shifts_basic_weighting(
            neighbour_distance = self.particle_interaction_distance,
            n_peaks = self.n_peaks,
            cc_weight_exp = 5,
            plot_pcl_n = self.plot_pcls,
            min_neighbours = min(self.min_neighbours, self.particles.num_pcls - 2),
            shift_std_cutoff = self.shift_std_cutoff,
            use_nbr_median = self.use_nbr_median)    
        self.out_fid = self.particles.write_fiducial_model(self.ali, use_local_medians = self.use_local_median_shifts)
        
        self.flg_shifts_exist = True
        self.to_json()
            
        
    
    def iterate(self, curr_iter = False, num_iterations = False):

        #should be possible to restart from any point of any iteration
        #restart technically onle needs curr_iter and out_dir
        #remake tomos - IMOD control by changing parameters in the previous iteration com files
        
        #def modify_iteration_inputs():
            
            
        def restart():
            init_flags = ['flg_inputs_verified', 'flg_mol_obj_initiated',
                          'flg_image_data_exist',
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
                print('DEV NOTE this seems not right')
                
                for mol_id in range(len(self.m_objs)):
                    mol_id += 1
                    prev_peet_dir = self.out_dir[:-1] + str(self.curr_iter - 1) + '/molecule_%s/peet' % mol_id
                    prev_fsc1 = prev_peet_dir + '/fsc1'
                    prev_fsc2 = prev_peet_dir + '/fsc2'
                    if isdir(prev_peet_dir):
                        #self.mol_1_prm = join(prev_peet_dir, split(self.mol_1_prm[-1])) #vp: this should be False after PEET
                        self.__dict__['mol_%s_prm1' % mol_id] = join(prev_fsc1, split(self.__dict__['mol_%s_prm1' % mol_id][-1]))
                        self.__dict__['mol_%s_prm2' % mol_id] = join(prev_fsc2, split(self.__dict__['mol_%s_prm2' % mol_id][-1]))
                    else: 
                        self.__dict__['mol_%s_prm' % mol_id] = tmp_dict['mol_%s_prm' % mol_id]
                        self.__dict__['mol_%s_prm1' % mol_id] = tmp_dict['mol_%s_prm1' % mol_id]
                        self.__dict__['mol_%s_prm2' % mol_id] = tmp_dict['mol_%s_prm2' % mol_id]
                        self.__dict__['mol_%s_tomogram_number' % mol_id] = tmp_dict['mol_%s_tomogram_number' % mol_id]
                        self.__dict__['mol_%s_ite' % mol_id] = tmp_dict['mol_%s_ite' % mol_id]
                        
                #self.__dict__['mol_%s_ite' % mol_id + 1] = 0

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
            self.initiate_molecules()
            
            self.out_fid = self.aligncom.dict['ModelFile'] #i.e. original imod fiducial model
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
                print('#################################################')
                for mol_obj in self.m_objs:
                    mol_obj.prep_peet(mol_obj.peet_dir, rotated_rec)
                self.export_mol_attr()
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
                if not self.flg_mol_obj_initiated:
                    self.initiate_molecules()
                    for mol_obj in self.m_objs:
                        mol_obj.preprocess_models()
                    self.flg_mol_obj_initiated = True

                if not self.flg_image_data_exist:
                    print('Generating tilt series...')
                    self.make_ts()
                if not self.flg_shifts_exist:
                    print('Measuring shifts...')
                    self.process_shifts()
                if not self.flg_tomos_exist:
                    print('Reconstructing tomograms...')
                    self.combine_fiducial_models()
                    self.match_tiny_tomos()
                    self.make_all_tomos()
                if not self.flg_peet_ran:
                    
                    for mol_obj in self.m_objs:
                        if mol_obj.prm or (mol_obj.prm1 and mol_obj.prm2):
                            print('Running PEET with molecule %s...' % mol_obj.molecule_id)
                            mol_obj.fsc_dirs = []
                            print('DEV NOTE: setting fsc_dirs to [] here. Somehow the second m_obj.fsc_dirs get set to the same as the first..../??')
                            mol_obj.run_peet()
                            
                    # for m in range(len(self.m_objs)):
                    #     print(0, self.m_objs[0], self.m_objs[0].fsc_dirs)
                    #     print(1, self.m_objs[1], self.m_objs[1].fsc_dirs)
                    #     if self.m_objs[m].prm or (self.m_objs[m].prm1 and self.m_objs[m].prm2):
                    #         print('Running PEET with molecule %s...' % self.m_objs[m].molecule_id)
                    #         #mol_obj.fsc_dirs = []
                    #         print('DEV NOTE: setting fsc_dirs to [] here. Somehow the second m_obj.fsc_dirs get set to the same as the first..../??')
                    #         self.m_objs[m].run_peet()
                    #         print(0, self.m_objs[0], self.m_objs[0].fsc_dirs)
                    #         print(1, self.m_objs[1], self.m_objs[1].fsc_dirs)



            print('Iteration execution time: %s s.' % int(np.round(
                                (time.time() - startTime), decimals = 0)))    
            #finishing touches, updating for next iteration
            self.flg_inputs_verified = False
            self.flg_image_data_exist = False
            self.flg_shifts_exist = False
            self.flg_mol_obj_initiated = False
            #self.flg_aligncom_formatted = False
            self.flg_tomos_exist = False
            self.flg_peet_ran = False


            self.rec_dir = self.out_dir
            self.out_dir = join(self.orig_out_dir, 'iteration_%s' % (self.curr_iter + 1))
            self.tomo = self.out_tomo
            self.update_dirs()
            self.update_comfiles()
            self.export_mol_attr()

            #check resolution improved
            
            #deal with rawtlt? otehr align files
            
            if not self.ignore_resolution_check and (self.mol_1_prm1 and self.mol_1_prm2):
                if curr_iter > 0:
                    get_area = True 
                    res = np.array(self.m_objs[0].res)
                    r = np.array(res)[:,1]
                    if np.any(np.floor(r) == np.floor(self.m_objs[0].peet_apix*2)) or get_area:
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

    