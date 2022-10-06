# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:43:24 2022

@author: vojta
"""

import os
from os.path import isdir, isfile, split, join, abspath, realpath
from copy import deepcopy
from PEETPRMParser import PEETPRMFile 
from PEETMotiveList import PEETMotiveList
from PEETModelParser import PEETmodel
from IMOD_comfile import IMOD_comfile
from extracted_particles import Extracted_particles
import numpy as np
from subprocess import check_output, Popen, PIPE
import mrcfile
import glob
#from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy import fftpack as fft
import matplotlib.pyplot as plt
import warnings
from itertools import zip_longest
import json

from flexo_tools import (get_apix_and_size, make_lamella_mask,
                         get_mod_motl_and_tomo,
                         make_non_overlapping_pcl_models,
                         fid2ndarray, run_processchunks)
from flexo_tools import (check_refpath,
                  get_tiltrange, bin_model, run_generic_process,
                 write_to_log, progress_bar, kill_process, imodscript)
from flexo_tools import replace_pcles, reproject_volume, mask_from_plotback, get2d_mask_from_plotback
from flexo_tools import (write_mrc, ctf_convolve_andor_dosefilter_wrapper,
                         extract_2d, ncc, get_peaks, get_resolution)

class Flexo_peet_prm:
    """
    Handling particle models, 3D/2D data and validating via PEET.
    
    """
    def __init__(self, molecule_id = False, #int
                 json_attr = False,
                  **kwargs):
        
        
        if json_attr:
            with open(json_attr, 'r') as f:
                json_dict = json.load(f)
            for key in json_dict:
                if isinstance(json_dict[key], str) and json_dict[key].endswith('.npy'):
                        setattr(self, key, np.load(json_dict[key]))   
                else:
                    setattr(self, key, json_dict[key])   
        else:
            self.molecule_id = int(molecule_id)
        
            for kwkey in kwargs.keys():
                self.__dict__[kwkey] = kwargs[kwkey]
            
            self.m_objs = False
            self.m_ids = False
    
            self.st_apix = kwargs.get('st_apix')                    
            self.tomo_binning = kwargs.get('tomo_binning')
            
            if not self.molecule_id:
                raise Exception('Molecule ID required.')
            
            self.prm = kwargs.get('prm')
            self.prm1 = kwargs.get('prm1')
            self.prm2 = kwargs.get('prm2')
            self.base_name = kwargs.get('base_name')
            self.prm_tomogram_number = kwargs.get('prm_tomogram_number')
            self.ite = kwargs.get('ite')
            if not self.ite:
                self.ite = 0
            self.search_rad = kwargs.get('search_rad')
            self.phimax_step = kwargs.get('phimax_step')
            self.psimax_step = kwargs.get('psimax_step')
            self.thetamax_step = kwargs.get('thetamax_step')
            self.average_volume = kwargs.get('average_volume')
            self.box_size = kwargs.get('box_size')
            
            self.peet_tomo = kwargs.get('peet_tomo') #path or 'init'
            self.use_davens_fsc = kwargs.get('use_davens_fsc')
            self.machines = kwargs.get('machines')
            self.peet_apix = kwargs.get('peet_apix')
            self.keep_peet_binning = True #if false, change peet binning to match flexo binning
            
            self.cleanup_list = kwargs.get('cleanup_list')
            if not self.cleanup_list:
                self.cleanup_list = []
            self.excluded_particles = kwargs.get('excluded_particles')
            if not self.excluded_particles:
                self.excluded_particles = []
    
            
            #combined model/motl for plotback
            self.model_file = kwargs.get('model_file')
            self.motl = kwargs.get('motl')
            self.model_file_binning = kwargs.get('model_file_binning')
            
            #derived 
            self.out_dir = join(kwargs.get('out_dir'), 'molecule_%s' % self.molecule_id) #used for tilt.log path and binned references
            self.reprojected_mod = join(self.out_dir, self.base_name + '_reprojected.fid')
            
            self.update_dirs()
            
        self.update_comfiles()
        
        
    def update_dirs(self):
        self.out_json = join(self.out_dir, 'mol_%s.json' % self.molecule_id)
        self.npy_dir = join(self.out_dir, 'npy')
        self.pcle_dir = join(self.out_dir, 'extracted_particles')
        self.xcor_peak_dir = join(self.out_dir, 'xcor_peaks')
        self.init_peet_dir = join(self.out_dir, 'init_peet')
        self.peet_dir = join(self.out_dir, 'peet')
        
    def update_comfiles(self):
        self.tiltcom = IMOD_comfile(self.rec_dir, 'tilt.com')
        self.newstcom = IMOD_comfile(self.rec_dir, 'newst.com')
        self.aligncom = IMOD_comfile(self.rec_dir, 'align.com')
        self.ctfcom = IMOD_comfile(self.rec_dir, 'ctfcorrection.com')
        
    def to_json(self):

        if not isdir(self.out_dir):
            os.makedirs(self.out_dir)
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
        
    def preprocess_models(self):
        """
        Check input models and reproject to 2D

        """
        if not isdir(self.out_dir):
            os.makedirs(self.out_dir)
            
        if self.prm or (self.prm1 and self.prm2):
        
            if not self.prm_tomogram_number:
                self.prm_tomogram_number = 1   
            
            if not self.prm and self.prm1 and not self.prm2:
                self.prm = self.prm1
        
            if self.prm1 and self.prm2:
                modfiles, motls = Flexo_peet_prm.combine_fsc_halves(self.prm1, self.prm2,
                                self.prm_tomogram_number, self.peet_dir, self.ite)
                _, _, tomos = get_mod_motl_and_tomo(self.prm1, self.ite) #try to get last iteration            
                prm_tomo = tomos[self.prm_tomogram_number - 1]
            elif self.prm and not (self.prm1 and self.prm2):
                motls, modfiles, tomos = get_mod_motl_and_tomo(self.prm, self.ite) #try to get last iteration
                prm_tomo = tomos[self.prm_tomogram_number - 1]
            self.peet_apix, prm_tomo_size = get_apix_and_size(prm_tomo)
            self.model_file = modfiles[self.prm_tomogram_number - 1]
            self.motl = motls[self.prm_tomogram_number - 1]     
            
            if not self.keep_peet_binning:
                self.peet_binning = self.tomo_binning
            else:
                self.peet_binning = float(np.round(self.peet_apix/self.st_apix, decimals = 0)) #work with "absolute binning
                
        elif self.model_file and self.motl:
            if not self.model_file_binning:
                self.model_file_binning = self.tomo_binning
            self.peet_binning = self.tomo_binning
            
        to_bin = self.tomo_binning/self.peet_binning
        mod_str = ('.').join(split(self.model_file)[1].split('.')[:-1])
        output_model = join(self.out_dir, mod_str + '_bin%s.mod' % to_bin)
        output_motl = join(self.out_dir, mod_str + '_bin%s.csv' % to_bin) #write a new motl without offsets to avoid adding them several times by mistake
        print('Binning model file %s' % output_model)
        bin_model(self.model_file, output_model, to_bin, motl = self.motl,
                  out_motl = output_motl)
        self.model_file = output_model
        self.motl = output_motl
        
        if not self.average_volume or not isfile(self.average_volume):
            raise Exception("Average volume not found %s." % self.average_volume)

        if not self.box_size:
            average_apix, average_size = get_apix_and_size(self.average_volume)
            map_binning = average_apix/self.st_apix
            rel_bin = self.tomo_binning/map_binning
            average_size = average_size[0]/rel_bin
            if average_size%2:
                average_size += 1
            weebit = int(np.ceil(average_size/20))*2
            self.box_size = [int(average_size + weebit), int(average_size + weebit)]
        elif isinstance(self.box_size, int):
            self.box_size = [self.box_size, self.box_size]
            
        self.reproject_model()
        self.to_json()
            
        
 
    @staticmethod 
    def prepare_prm(prm, ite, tomo, tom_n, base_name, new_prm_dir,
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
                            tiltlog_dir = False
                            ):
        """
        Modify prm file to run PEET with Flexo tomogram.
        
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
            
        Returns:
            str, new prm path
            float, apix
        """
        
        def get_searchrange(key, ite, shift_rad = False):
            d = prm.prm_dict[key]
            if len(d) < ite - 1:
                ite = 0
            if shift_rad:
                return int(d.replace('{','').replace('}','').replace('[','').replace(']','').split()[ite])
            else:
                v = [float(n) for n in d[ite].split(':')]
                return v[-1], v[1]
        
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
        if tiltlog_dir:
            trange = get_tiltrange(join(tiltlog_dir, 'tilt.log'))
            trange = [trange for x in tom_n]  
        else:
            trange = prm.prm_dict['tiltRange']
            trange = [trange[n] for n in tom_n]

        
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
            new_reference = join(split(new_prm_dir)[0], 'bin_' + split(reference)[-1])
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
        
        
        if isinstance(phimax_step, (bool,type(None))):
            phimax_step = get_searchrange('dPhi', ite)
        if isinstance(psimax_step, (bool,type(None))):
            psimax_step = get_searchrange('dPsi', ite)
        if isinstance(thetamax_step,(bool,type(None))):
            thetamax_step = get_searchrange('dTheta', ite)    
            
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
            search_rad = get_searchrange('searchRadius', ite, shift_rad = True)
            #search_rad = [r_size[0]/4]
        elif isinstance(search_rad, int):
            search_rad = [search_rad]
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
    
    @staticmethod
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
        
        Flexo_peet_prm.run_split_peet(prm_base1, fsc1d, prm_base2, fsc2d, machines)
        #calcUnbiasedFSC
        os.chdir(peet_dir)
        
        
        if use_davens_fsc:
            vol1 = glob.glob(fsc1d + '/unMasked*.mrc')
            vol1 = sorted(vol1, key=lambda x: int(x.split('_')[-1].split('.')[0].strip('Ref')))[-1]
            vol2 = glob.glob(fsc2d + '/unMasked*.mrc')
            vol2 = sorted(vol2, key=lambda x: int(x.split('_')[-1].split('.')[0].strip('Ref')))[-1]        
            Flexo_peet_prm.get_fsc(vol1, vol2, peet_dir)
        else:
            fsc_log = join(peet_dir, 'calcUnbiasedFSC.log')
            print('Running calcUnbiasedFSC...')
            run_generic_process(['calcUnbiasedFSC', prm1, prm2], out_log = fsc_log)
    
        os.chdir(cwd)
    
    @staticmethod
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

    
    @staticmethod
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
            
    @staticmethod
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
            process = Popen(cmd, stdout = PIPE, stderr = PIPE, 
                            preexec_fn=os.setsid)
            write_to_log(c_log1, out_dir + '\n' + (' ').join(cmd) + '\n')
            #process2
            os.chdir(out_dir2)  
            cmd2 = ['/bin/sh', 'processchunks', '-g', '-n', '18', '-P', 
                   (',').join(m2), base_name2]
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
            
    @staticmethod
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
            return new_mods, new_motls
        else:
            return [new_mods[0]], [new_motls[0]]
            
    def prep_peet(self, peet_dir = False, tomo = False):
        """
        Prepare prm for running PEET with half maps.

        Parameters
        ----------
        peet_dir : str, optional
            Directory where fsc1/fsc2 directories will be created.
            The default is False.
        tomo : str, optional
            Path to volume to run PEET with. "init" copies path from input prm.
            The default is False.

        """
        
        if not peet_dir:
            peet_dir = self.peet_dir
        if not tomo:
            tomo = self.peet_tomo
        
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
            self.prm, peet_apix = self.prepare_prm(
                    self.prm, self.ite, tomo, self.prm_tomogram_number,
                    self.base_name, peet_dir,
                    search_rad = self.search_rad,
                    phimax_step = self.phimax_step,
                    psimax_step = self.psimax_step,
                    thetamax_step = self.thetamax_step,
                    tiltlog_dir = split(self.out_dir)[0])
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
            #first iteration, prm2 was specified
            self.prm1, peet_apix = self.prepare_prm(
                    self.prm1, self.ite, tomo, self.prm_tomogram_number,
                    self.base_name, fsc1d,
                    search_rad = self.search_rad,
                    phimax_step = self.phimax_step,
                    psimax_step = self.psimax_step,
                    thetamax_step = self.thetamax_step,
                    tiltlog_dir = split(self.out_dir)[0])

            self.prm2, peet_apix2 = self.prepare_prm(
                    self.prm2, self.ite, tomo, self.prm_tomogram_number,
                    self.base_name, fsc2d,
                    search_rad = self.search_rad,
                    phimax_step = self.phimax_step,
                    psimax_step = self.psimax_step,
                    thetamax_step = self.thetamax_step,
                    tiltlog_dir = split(self.out_dir)[0])
            if peet_apix != peet_apix2:
                raise ValueError('The two PEET half data-sets do not have the'
                                 + ' same binning.')

        self.peet_apix = peet_apix
        
        
    def run_peet(self):  
        if not self.prm and not (self.prm1 and self.prm2):
            print('PEET prm file not specified, PEET will not run.')
        else:
            if self.curr_iter == 1:
                if self.fsc_dirs: #meaning this run is being restarted, in which case skip init_peet
                    self.fsc_dirs = [self.init_peet_dir]
                else:
                    self.prep_peet(peet_dir = self.init_peet_dir, tomo = 'init')
                    Flexo_peet_prm.peet_halfmaps(self.init_peet_dir, self.prm1,
                            self.prm2, self.machines, use_davens_fsc = self.use_davens_fsc)
                    self.fsc_dirs.append(self.init_peet_dir)
            self.prep_peet(peet_dir = self.peet_dir, tomo = self.peet_tomo)
            
            self.flg_peet_ran = True # this needs to be here in case peet crashes and a restart is attempted
            self.to_json()
            
            Flexo_peet_prm.peet_halfmaps(self.peet_dir, self.prm1, self.prm2,
                                         self.machines, use_davens_fsc = self.use_davens_fsc)
            self.fsc_dirs.append(self.peet_dir)

        self.res = Flexo_peet_prm.plot_fsc(self.fsc_dirs, 
                                           self.peet_dir, self.cutoff, self.peet_apix)
        self.to_json()

    
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
    
    def make_and_reproject_plotback(self, average_volume, model_file, plotback3d, plotback2d, motl,
                                    mask2d = False):
        """
        Generates a plotback and reprojects it into a 2D tilt series.

        Parameters
        ----------
        average_volume : str
            Path to segmented volume. 
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
        replace_pcles(average_volume, self.tomo_size, motl,
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
                
    def generate_image_data(self, starting_pcle_id = False):
        
        if not isinstance(starting_pcle_id, bool):
            self.starting_pcle_id = starting_pcle_id
            
        if self.lamella_model and not self.lamella_mask_path:
            
            self.lamella_mask_path = join(self.out_dir, self.base_name + '_lamella_mask.mrc')
            make_lamella_mask(self.lamella_model, tomo_size = [self.tomo_size[0], self.tomo_size[2], self.tomo_size[1]],
                              out_mask = self.lamella_mask_path, rotx = False)
            print('DEV NOTE: making lamella mask is unfinished. Could check for coordinates outside volume and rotate if needed.')
        
        #sorted_pcls is used to keep track of particle params. shape: [number of tilts:model points per tilt:7
        #   0=xcoords, 1=ycoords, 2=tilt number, 3=particle index, 4=group id (from 0), 5=tilt angle, 6=defocus)]
        self.sorted_pcls = fid2ndarray(self.reprojected_mod,
                defocus_file = self.defocus_file, ali = self.ali,
                excludelist = self.excludelist,
                base_name = self.base_name, out_dir = self.out_dir,
                apix = self.apix, tlt = self.tlt)
        
        if self.non_overlapping_pcls:
            (self.fid_list, self.groups, remainder, self.sorted_pcls,
             self.split3d_models, self.split_motls
             ) = make_non_overlapping_pcl_models(self.sorted_pcls,
            self.box_size, self.out_dir, model3d = self.model_file,
            motl3d = self.motl)
                                                 
            self.excluded_particles.extend(remainder)
        else:
            self.fid_list = [self.reprojected_mod]
            self.groups = np.ones(self.sorted_pcls.shape[1], dtype = bool)
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
            '>from flexo_peet_prm import Flexo_peet_prm',
            #'>f = Flexo(json_attr = "%s")' % self.out_json,
            '>f = Flexo_peet_prm(json_attr = "%s")' % self.out_json,
            '>f.make_and_reproject_plotback("%s", "%s", "%s", "%s", "%s", "%s")' % (
                self.average_volume,
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
                '>from flexo_peet_prm import Flexo_peet_prm',
                '>f = Flexo_peet_prm(json_attr = "%s")' % self.out_json,
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
                    '>from flexo_peet_prm import Flexo_peet_prm',
                    '>from flexo_tools import tomo_subtraction',
                    '>f = Flexo_peet_prm(json_attr = "%s")' % self.out_json,
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
                    '>from flexo_peet_prm import Flexo_peet_prm',
                    '>f = Flexo_peet_prm(json_attr = "%s")' % self.out_json,
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
        """
        Extract 2D particles and cross correlate.
        pcl_indices : list or 1D ndarray
        """
        def extract_filter_write(n):
            #aid = ali_f_id[x]
            if self.apply_2d_mask:
                normalise_first = True
            else:
                normalise_first = False
                
            query = extract_2d(arr_ali[ali_arr_ids[n]],
                               self.sorted_pcls[:, pcl_indices[n], :3],
                               self.box_size, normalise_first = normalise_first)
            ref = extract_2d(arr_plotback[plotback_arr_ids[n]],
                             self.sorted_pcls[:, pcl_indices[n], :3]
                             , self.box_size, normalise_first = normalise_first)
            if self.apply_2d_mask:
                mask = extract_2d(arr_mask[plotback_arr_ids[n]],
                             self.sorted_pcls[:, pcl_indices[n], :3], self.box_size)
            #there is support for running averages of neighbouring tilts, could be worth playing with
            #use offsets = True in extract_2d_simplified to get offsets

            partial_list = np.zeros(len(query), dtype = float)
            if self.ignore_partial:
                #skip CC if > 20% of a query is flat
                for l in range(len(query)):
                    #can just check one line of each dim
                    dc = np.max(np.unique(np.diagonal(query[l]), return_counts = True)[1])
                    partial_list[l] = dc/query[0].shape[0]

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
        pcl_groups = self.sorted_pcls[0, pcl_indices, 4] #group ids of this subset of particles
        unique_groups = np.array(np.unique(pcl_groups), dtype = int)
        num_digits = len(str(self.sorted_pcls.shape[1]))
        if write_pcles:
            if not isdir(self.pcle_dir):
                os.makedirs(self.pcle_dir)
        if not isdir(self.xcor_peak_dir):
            os.makedirs(self.xcor_peak_dir)
            
        cc_peaks = np.zeros((self.sorted_pcls.shape[0], self.n_peaks, 4)) #[number of tilts, number of peaks, 4] (x coord, y coord, peak value, mask)

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
            defoci = np.zeros((self.sorted_pcls.shape[0], len(pcl_indices)))
        else:
            defoci = self.sorted_pcls[:, pcl_indices, 6]
        
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
        
        self.particles = Extracted_particles(self.sorted_pcls, 
                                        apix = self.apix, 
                                        out_dir = self.out_dir,
                                        excludelist = [],#self.excludelist, #EEEEEEEEEEHMmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
                                        base_name = self.base_name,
                                        chunk_base = 'xcor',
                                        n_peaks = self.n_peaks,
                                        groups = self.groups,
                                        tilt_angles = self.sorted_pcls[:, 0, 5], #this way I shouldn't have to worry about excludelist
                                        model_3d = self.model_file,
                                        exclude_worst_pcl_fraction = self.exclude_worst_pcl_fraction)
        
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
                    '>from flexo_peet_prm import Flexo_peet_prm',
                    '>f = Flexo_peet_prm(json_attr = "%s")' % self.out_json,
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
            
        self.flg_shifts_exist = True
        self.to_json()
