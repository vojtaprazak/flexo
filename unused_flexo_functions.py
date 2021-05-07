#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:09:57 2020

@author: vojta
"""

def CC(ref, query, limit = 10, interp = 10, scaling = 1, verbose = False):
    """Calculates translation between two image stacks.
        Query translated by 1,1 relative to ref will output -1,-1 shift
    Input arguments:
        target/reference [numpy array]
        probe/query [numpy array]
        maximum distance to search [even real number] 
            - determines the size of box around the center of CC map
                that will be searched for peaks
        interpolation [int]
        try to smooth big outliers? [bool]
        scaling factor - larger numbers favour smaller shifts
    Returns:
        XY shifts [2D array], warnings
    """
    flexo_warnings = np.zeros((ref.shape[0]))
    if np.any(np.array(ref[0].shape)/2 - 1 < limit):
        print 'Maximum shift cannot be >= box_size/2.\
        Defaulting to maximum allowed shift.'
        limit = (np.array(ref[0].shape)/2 - 1)[0]
    #make stack of ccmaps
    ccmaps = []
    for x in range(len(ref)):
        ccmaps.append(get_cc_map(ref[x], query[x], limit, interp))
    ccmaps = np.array(ccmaps)
    #get mean of low tilt peaks
    init_peaks = []
    #define an arbitrary middle section of ref/query stacks... 7? perfect!
    # small stacks will just be treated in a single step
   
    if len(ref) < 7:
        bot = 0
        top = len(ref) 
    else:
        bot = int(np.floor(len(ref)/2)) - 3
        top = int(np.floor(len(ref)/2)) + 4


    #initial CC of middle section
    for x in range(bot, top):
        pk, val, warn = buggy_peaks(ccmaps[x])
        if len(pk.shape) > 1:
            pk = pk[0]
        init_peaks.append(pk)
        if len(pk) == 0:
            raise ValueError('get_cc_map returned a flat array which suggests that the input model does not match input volume.')
    
    shifts = np.zeros((len(ccmaps), 2))
    shifts[bot : top] = init_peaks
    

#pick reasonable peaks
    for x in range(bot, top): 
    #refine middle
        #take median of middle tilts
        mn = np.median(shifts[bot : top, 0]),\
                np.median(shifts[bot : top, 1])
        pk, val, warn = buggy_peaks(ccmaps[x])
        shifts = what_peak(x, pk, val, mn, shifts, scaling)

    if len(ref) < 7:
        pass
    else:
        #walk up   
        for x in range(top, len(ref)): 
            mn = np.mean(shifts[x -7: x, 0]), np.mean(shifts[x -7: x, 1]) 
            #walking average, previous 7 values
            #std = np.std(shifts[x-7:x-1,0]), np.std(shifts[x-7:x-1:,1])    
            pk, val, warn = buggy_peaks(ccmaps[x])
            shifts = what_peak(x, pk, val, mn, shifts, scaling)
            flexo_warnings[x] = warn
            
        #walk down                
        for x in range(bot -1, -1, -1): 
            mn = np.mean(shifts[x +1: x +8, 0]), np.mean(shifts[x +1: x +8, 1])
            #walking average, last 6 values
            pk, val, warn = buggy_peaks(ccmaps[x])
            shifts = what_peak(x, pk, val, mn, shifts, scaling)
            flexo_warnings[x] = warn            
            
    return (np.divide(shifts, float(interp))-float(limit)), flexo_warnings, ccmaps

def what_peak(x, pk, val, mn, shifts, scaling):
    """
    Now obsolete.
    Questionable usefulness.
    Selects the highest peak close to a specified point.  
    Peak score = peak value/ (scaling_factor + distance)^2
    Input arguments:
        x - x index of array to be overwritten (must be a n by 2 array)
        list of peak coordinates [2D array]
        peak values
        XY coordinates of search origin [tuple/list]
        array to be overwritten
        scaling factor 
    Returns:
         overwritten array       
        
    """
    pk = pk.squeeze()    
    if len(pk.shape) > 1: #signle peak len() = 2, need to check the shape
        tree = KDTree(pk)
        dst, pos = tree.query(mn, len(pk))
        if pos[0] != 0: #if closest peak is not the highest
            sorted_peaks = zip(*sorted(zip(pos, val)))[1] # use positions                
            weighted_peaks = np.divide(sorted_peaks, np.square(scaling + dst)) #divide by the square distance from mean
            shifts[x] = pk[weighted_peaks == max(weighted_peaks)]
        else:
            shifts[x] = pk[pos[0]]
    else:
        shifts[x] = pk
    return shifts   

def mask_from_plotback(volume, out_mask = False, grey_dilation_level = 5,
                       gopen = 0, gerode = 0):
        """Creates a mask out of a backplotted volume.  Curently used
        only by make_nonoverlapping_alis function.
        
        Input can be path [string] or np.array """
        if type(volume) == str:
            m = deepcopy(mrcfile.open(volume).data)
        else:
            m = volume
        if not gerode:
            m = grey_dilation(-m, grey_dilation_level) 
        else: 
            from scipy.ndimage.morphology import grey_erosion
            m = grey_erosion(-m, gerode) 
        if gopen:
            from scipy.ndimage.morphology import grey_opening
            m = grey_opening(-m, gopen) 
        
        m = m != 0 #binarise.  Assuming background is zero
        edge = 1 #masktomrec acts weird if the mask touches the edge
            #of the volume.  Create 1 voxel thick black edge
        m[:edge] = m[-edge:] = m[:,:edge] = m[:,-edge:]\
        = m[:,:,:edge] = m[:,:,-edge:] = 0
        if out_mask:
            write_mrc(out_mask, m)
        else:
            return m
        
def run_masktomrec(N, mask, tlt, ali, tomo, out_dir, tomo_size, base_name,\
                   subtracted):
    
    
    wdir = os.getcwd()
    tmpdir = join(out_dir, 'mtr_tmp_%02d' % N)    
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    os.chdir(tmpdir)
    outtom = join(tmpdir, 'masktomrec_%02d.rec') % N
    outali = join(tmpdir, 'masktomrec_%02d.ali') % N
    #note: this gets renamed to subtacted_%02d.ali
    
    # what is this?? commented out VP 4 Feb 2020
    #rotated_tomo = join(tmpdir, base_name + '_backrotated.rec')
        
# VP 4 Feb 2020 using imod masktomrec by default 
#    if not imod_masktomrec:
#        
#        check_output('masktomrec -a %s -i %s -o %s -O %s -m %s -z %s -l 8'\
#            % (tlt, ali, outtom, outali, mask, tomo_size[2]), shell = True)
#        check_output('clip subtract %s %s %s' \
#                     % (ali, outali, subtracted), shell = True)
    
    print " use cleaning..._local.csh if local exists"
    if not (isfile    
    ('/raid/45/vojta/software/scripts/cleaning_with_imod_using_tilt.csh')):
        print 'It might be a good idea tomake sure to look for\
        cleaning_with_imod_using_tilt.csh in the installation directory.'

    outali = realpath(join(tmpdir, 'ts_iter008.mrc'))
    check_output('rotatevol -input %s -output %s -size %s,%s,%s -angles 0,0,90'\
                 % (tomo, rotated_tomo, tomo_size[0], tomo_size[2],\
                    tomo_size[1]), shell = True)
    
    imodmtr_path = '/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/test_new_toy_tomo/cleaning_with_imod_using_tilt.csh'
    #imodmtr_path = '/raid/45/vojta/software/scripts/cleaning_with_imod_using_tilt.csh'
    
    check_output('%s %s %s %s %s %s 8' % (imodmtr_path, tlt, ali,
                rotated_tomo, tomo_size[2], mask), shell = True)
    check_output('clip subtract %s %s %s'
                 % (ali, outali, subtracted), shell = True)

    os.chdir(wdir)
    return subtracted, outtom

def backtransform_model(mod, base_name, rec_dir, out_dir, tomo, ali,
                        model_file_binning, verbose = True):
    """
    transform model to match raw stack
    
    I'm not sure how this deals with local aligned tomograms...!

    
    """
    
    modpath, modname = split(mod)
    
    tmpmod = join(out_dir, 'tmp_' + modname)
    binmod = join(out_dir, 'bin_' + modname)
    scaled_mod = join(out_dir, 'full_' + modname)
    full_tomo = join(rec_dir, base_name + '_full.rec')
    full_rotated = join(rec_dir, base_name + '.rec')
    origali = join(rec_dir, base_name + '.ali')
    if float(model_file_binning) != 1.:
        print 'Model file binning %s' % model_file_binning
        m = np.array(PEETmodel(mod).get_all_points()) * model_file_binning
        mout = PEETmodel()
        for p in range(len(m)):
            mout.add_point(0, 0, m[p])
        mout.write_model(binmod)
        mod = binmod
    check_output('imodtrans -I %s %s %s' % (tomo, mod, tmpmod), shell = True)
    check_output('imodtrans -i %s %s %s'%(tomo, tmpmod, scaled_mod), shell = True)

    pwd = os.getcwd()
    os.chdir(rec_dir)
    out_mod = join(realpath(out_dir), 'rawtiltcoords.mod')
    check_output(('rawtiltcoords -root %s -volume %s -ali %s -full %s -center'
                 ' %s -output %s') % (base_name, tomo, origali, full_tomo,
                                scaled_mod, out_mod), shell = True)
    rawmod = PEETmodel(out_mod).get_all_points()

    max_tilt = int(np.round(max(rawmod[:,2])+1, decimals = 0))
    sorted_pcls = np.array([np.array(rawmod[x::max_tilt]) for x in range(max_tilt)])
    if float(model_file_binning) != 1.:
        sorted_pcls[:,:,:2] *= model_file_binning
    os.chdir(pwd)
    return sorted_pcls, out_mod

def format_unwarp_comfiles(base_name, ali, out_dir, rec_dir, outmod_name,
            tlt, tomo_binning, tomo_size, n_patches, separate_group,
            thickness, global_xtilt, excludelist, axiszshift, align_with_ali, 
            origxf, tilt_axis_angle, fidn = (10,2), SHIFT = False,
            OFFSET = False, global_only = False, globalXYZ = True, 
            use_stack_size = False, force_thickness = False, force_xf = False):

    
    """tomo_size now defines aligned stack size.      
        use_stack_size - if True - will determine size and thickness from
                        .st size, specified thickness and tomo_binning
        force_thickness - if False, thickness is automatically binned by 
                        tomo_binning (in newst.com)
        force_xf - specifically for match_tomos: need original xf
    """
    overlap = 0.5
    base_output = abspath(join(out_dir, base_name))
    output_model = str(base_output) + '.3dmod'
    output_resid = str(base_output) + '.resid'
    output_xyz = str(base_output) + '.xyz'
    output_tilt = str(base_output) + '.tlt'
    output_xtilt = str(base_output) + '.xtlt'
    output_tltxf = str(base_output) + '.tltxf'
    output_nogapsfid = str(base_output) + '_nogaps.fid'
    output_localxf = str(base_output) + '_local.xf'
    output_xf = str(base_output) + '.xf'
    #output_fidtlt = str(base_output) + '_fid.tlt'
    #output_fidxf = str(base_output) + '_fid.xf'
    output_resmod = str(base_output) + '.resmod'
    output_ali = abspath(join(out_dir, base_name + '.ali'))
    output_rec = abspath(join(out_dir, base_name + '_full.rec'))
    #zero_xf = abspath(join(out_dir, 'init.xf'))
    fid_n1, fid_n2 = fidn

    tilt_angles = [float(x.strip('\n')) for x in open(tlt)]
    zero_tlt = find_nearest(tilt_angles, 0)
    st = join(realpath(rec_dir), str(base_name) + '.st')

    if use_stack_size:
        st_size = get_apix_and_size(st)[1]
        tomo_size = get_binned_size(st_size, tomo_binning)[0]
        #thickness = get_binned_size((thickness, 0), tomo_binning)[0][0]
        if (135 > np.absolute(np.median(tilt_axis_angle)) > 45
            or 315 > np.absolute(np.median(tilt_axis_angle)) > 225):
            tomo_size = (tomo_size[1], tomo_size[0])
            


    with open(join(out_dir, 'align.com'), 'w') as f:
        f.write('$tiltalign\t-StandardInput')
        f.write('\nModelFile\t%s' % outmod_name)
        if align_with_ali:
            f.write('\nImageFile\t%s' % ali)
            f.write('\nImagesAreBinned\t%s' % tomo_binning)
        else:
            f.write('\nImageFile\t%s' % st)
        f.write('\nOutputModelFile\t%s' % output_model) 
        f.write('\nOutputResidualFile\t%s' % output_resid) 
        f.write('\nOutputFidXYZFile\t%s' % output_xyz) 
        f.write('\nOutputTiltFile\t%s' % output_tilt) 
        f.write('\nOutputXAxisTiltFile\t%s' % output_xtilt) 
        f.write('\nOutputTransformFile\t%s' % output_tltxf) 
        f.write('\nOutputFilledInModel\t%s' % output_nogapsfid) 
        #rotation angle is adjusted by offset by checking final ali .xf
        if align_with_ali:
            f.write('\nRotationAngle\t0')
        else:
            f.write('\nRotationAngle\t%s' % -np.median(tilt_axis_angle)) 
        if separate_group != '':
            f.write('\nSeparateGroup\t%s' % separate_group) 
        f.write('\nTiltFile\t%s' % tlt) 
        f.write('\nAngleOffset\t0.0') 
#        if excludelist:
#            tt = [str(x) for x in excludelist]
#            f.write('\nExcludeList %s\n' % (',').join(tt))
        f.write('\nRotOption\t1') 
        f.write('\nRotDefaultGrouping\t5') 
        f.write('\nTiltOption\t2')
        f.write('\nTiltDefaultGrouping\t5')
        f.write('\nMagReferenceView\t%s' % zero_tlt)
        f.write('\nMagOption\t1')
        f.write('\nMagDefaultGrouping\t4')
        f.write('\nXStretchOption\t0')
        f.write('\nSkewOption\t0')
        f.write('\nXStretchDefaultGrouping\t7')
        f.write('\nSkewDefaultGrouping\t11')
        f.write('\nBeamTiltOption\t0')
        f.write('\nXTiltOption\t4')
        f.write('\nXTiltDefaultGrouping\t2000')
        f.write('\nResidualReportCriterion\t3.0')
        f.write('\nSurfacesToAnalyze\t2')
        f.write('\nMetroFactor\t0.25')
        f.write('\nMaximumCycles\t1000')
        f.write('\nKFactorScaling\t1')
        f.write('\nNoSeparateTiltGroups\t1')
        f.write('\nAxisZShift\t%s' % axiszshift)
        f.write('\nShiftZFromOriginal\t1')
        f.write('\nShiftZFromOriginal\t1')
        if not global_only:
            f.write('\nLocalAlignments\t1')
            f.write('\nOutputLocalFile\t%s' % output_localxf)
            f.write('\nMinSizeOrOverlapXandY\t%s,%s' % (overlap, overlap))
            f.write('\nMinFidsTotalAndEachSurface\t%s,%s' % (fid_n1, fid_n2))
            if globalXYZ:
                f.write('\nFixXYZCoordinates\t1')
            else:
                print 'global xyz coordinates can be relaxed in the last iteration'
                f.write('\nFixXYZCoordinates\t0')
            f.write('\nLocalOutputOptions\t1,0,1')
            f.write('\nLocalRotOption\t3')
            f.write('\nLocalRotDefaultGrouping\t6')
            f.write('\nLocalTiltOption\t5')
            f.write('\nLocalTiltDefaultGrouping\t6')
            f.write('\nLocalMagReferenceView\t%s' % zero_tlt)
            f.write('\nLocalMagOption\t3')
            f.write('\nLocalMagDefaultGrouping\t7')
            f.write('\nLocalXStretchOption\t0')
            f.write('\nLocalXStretchDefaultGrouping\t7')
            f.write('\nLocalSkewOption\t0')
            f.write('\nLocalSkewDefaultGrouping\t11')
            f.write('\nNumberOfLocalPatchesXandY\t%s,%s' % (n_patches[0], n_patches[1]))
        f.write('\nRobustFitting\t')
        if align_with_ali:
            f.write('\n$xfproduct -StandardInput')
            f.write('\nInputFile1\t%s' % origxf) # looks like it has to be zeros?
            f.write('\nInputFile2\t%s' % output_tltxf)
            f.write('\nOutputFile\t%s' % output_xf)
        #f.write('\n$b3dcopy -p %s %s' % (output_tltxf, output_xf))
        #f.write('\n$b3dcopy -p %s %s' % (output_tilt, output_fidtlt))
        f.write('\n$if (-e %s) patch2imod -s 10 %s %s' % (output_resid, output_resid, output_resmod))
        #f.write('\n$if (-e ./savework) ./savework')
    with open (join(out_dir, 'newst.com') ,'w') as f:
        f.write('\n$newstack -StandardInput')
        f.write('\nInputFile\t%s' % st)
        f.write('\nOutputFile\t%s' % output_ali)
        if force_xf:
            f.write('\nTransformFile\t%s' % force_xf)
        else:
            f.write('\nTransformFile\t%s' % output_xf)
        f.write('\nTaperAtFill\t1,0')
        f.write('\nAdjustOrigin')
        f.write('\nSizeToOutputInXandY\t%s,%s' %
                (tomo_size[0], tomo_size[1]))
        f.write('\nOffsetsInXandY\t0.0,0.0')
#        if excludelist:
#            f.write('\nExcludeSections\t%s' % (',').join(str(x) for x in excludelist))
        print 'Newst binning %s' % tomo_binning
        f.write('\nBinByFactor\t%s' % int(tomo_binning))
        f.write('\nAntialiasFilter\t-1')
        #f.write('\n$if (-e ./savework) ./savework')
               
    with open (join(out_dir, 'tilt.com') ,'w') as f:
        f.write('$tilt -StandardInput')
        f.write('\nInputProjections\t%s' % output_ali)
        f.write('\nOutputFile\t%s' % output_rec)
#need to use the newly made tlt!!!!!!!
#        if excludelist:
#            f.write('\nTILTFILE\t%s' % join(realpath(out_dir),
#                                            'fixed_' + base_name + '.tlt'))
#        else:
        f.write('\nTILTFILE\t%s' % output_tilt)
        if not force_thickness:
            f.write('\nIMAGEBINNED\t%s' % int(tomo_binning))
        f.write('\nTHICKNESS\t%s' % thickness)
        f.write('\nRADIAL\t0.5\t0.1')
        f.write('\nFalloffIsTrueSigma\t1')
        f.write('\nXAXISTILT\t%s' % global_xtilt)
        f.write('\nSCALE\t0.0\t0.2')
        f.write('\nPERPENDICULAR')
        if excludelist:
            tt = [str(x) for x in excludelist]
            f.write('\nEXCLUDELIST %s' % (',').join(tt))
        #f.write('\nMODE\t%s' % mode)
        #f.write('\nFULLIMAGE\t%s,%s' % (st_size[0], st_size[1]))
        f.write('\nSUBSETSTART\t0 0')
        f.write('\nAdjustOrigin')
        f.write('\nActionIfGPUFails 1,2')
        f.write('\nXTILTFILE\t%s' % output_xtilt) 
        if OFFSET:
            try:
                OFFSET[0]
                f.write('\nOFFSET %s' % OFFSET[0]) 
            except:
                f.write('\nOFFSET %s' % OFFSET) 
        if np.any(SHIFT):
            f.write('\nSHIFT %s %s' % (SHIFT[0]/tomo_binning, SHIFT[1]/tomo_binning))
        if not global_only:
            f.write('\nLOCALFILE\t%s' % output_localxf)
        if zfac:
            f.write('\nZFACTORFILE\t%s' % join(realpath(rec_dir), base_name + ".zfac"))
        #f.write('\n$if (-e ./savework) ./savework')
    return output_ali, output_rec, output_xf

def OLD_make_non_overlapping_pcl_models(sorted_pcls, box_size, out_dir,\
                                    threshold = 0.01):
    """ Generates non-overlapping particle models i.e. particle model points 
    that come into proximity (defined by box size) at any view of the tilt
    series are split into independent models.
    threshold [float] can be used to get rid of models that have very few points
    in them.  E.g. 0.01 removes models that have less than 1% of total particles.
    """

    #box size can be 1 int, a tuple or a list
    if isinstance(box_size, int):
        box_size = [box_size, box_size]

    ssorted_pcls = np.dstack((sorted_pcls,
                              np.zeros((sorted_pcls.shape[0],
                                        sorted_pcls.shape[1], 2))))
    ssorted_pcls[:,:,3] =  range(sorted_pcls.shape[1])    
    #[number of tilts, model points per tilt, xytilt_number coords,
    # model point ordinal, group id from 1!]
    indices = np.arange(0, sorted_pcls.shape[1])
    pvp = np.zeros((sorted_pcls.shape[0],\
                    sorted_pcls.shape[1],\
                    sorted_pcls.shape[1], 2))
    for x in range(sorted_pcls.shape[0]): #need to check every tilt
        tree = KDTree(ssorted_pcls[x,:,:2])
        dst, pos = tree.query(ssorted_pcls[x,:,:2],\
                              ssorted_pcls.shape[1],\
                              distance_upper_bound = box_size[0])#dist from Y
        pvp[x,:,:,0] = dst
        pvp[x,:,:,1] = pos
    mm = np.zeros((sorted_pcls.shape[1], sorted_pcls.shape[1]))    
    for x in range(pvp.shape[1]):
        tmp = np.zeros((pvp.shape[0], pvp.shape[1]))
        for y in range(pvp.shape[0]):        
            tmp[y] = np.isin(indices, pvp[y,x,:,1], invert = True)
        mm[x] = np.all(tmp, axis = 0)
    for x in range(mm.shape[0]):
        mm[x,x] = 1

    groups = []
    indices = np.arange(len(mm))
    x = 0
    while x < len(indices):
        tmp_indices = indices[indices != len(indices) + 1]
        y = 0
        a = np.array(mm[tmp_indices][0], dtype = 'bool')#mm[x,x] is always True
        while tmp_indices[y] < tmp_indices[-2]:
            y += 1
            if  np.logical_and(a, mm[tmp_indices[y]])[tmp_indices[y]]:
                a = np.logical_and(a, mm[tmp_indices[y]])
        if x != 0:
            if len(np.array(groups).shape) == 1:
                test = np.logical_and(a, groups)
            else:
                test = np.array(groups).sum(axis = 0) + a  > 1
            mmm = np.where(test == True)
            if np.any(mmm):
                a[np.where(test == True)] = False    
        groups.append(a)
        indices[a] = len(indices) + 1
        if (indices < len(indices) + 1).sum() < 2:
            x = len(indices) #which means the man while loop will terminate
        else:
            x = indices[indices != len(indices) + 1][0]
    groups = np.array(groups)
    yoyoyo = groups.sum(axis = 1)/float(mm.shape[0]) < threshold
    if np.any(yoyoyo):
        print '%s groups had less than %s times the total number of particles\
        and were removed.' % (len(yoyoyo), threshold)
        groups = groups[np.logical_not(yoyoyo)]
    print 'Number of non-overlapping models ' + str(len(groups))
    outmods = []
    pcl_ids = []
    for x in range(len(groups)):
        out_name = join(out_dir, 'model_group_%02d.mod' % (x))
        outmods.append(out_name)
        outmod = PEETmodel()
        g = np.swapaxes(sorted_pcls[:, groups[x]], 1, 0)
        pcl_ids.append(np.array(ssorted_pcls[0, groups[x], 3], dtype = 'int32'))
        for p in range(len(g)):
            if p != 0:
                    outmod.add_contour(0)
            for r in range(g.shape[1]):
                outmod.add_point(0, p, g[p,r])
        outmod.write_model(out_name)    
    novlp_sorted_pcls = []
    for g in range(len(groups)):
        rawmod = PEETmodel(outmods[g]).get_all_points()
        max_tilt = int(np.round(max(rawmod[:,2])+1, decimals = 0))
        novlp_sorted_pcls.append(np.array([np.array(rawmod[h::max_tilt])\
                                    for h in range(max_tilt)]))
        
    return outmods, novlp_sorted_pcls, pcl_ids, groups 
    #pcl_ids show which particles go into which outmod, groups = masks
    
def bias_increase(tilt_angle, centre_bias, bpower):
    """
    used within biased_cc and cc_walk
    """
    cos_centre_bias = (centre_bias
                   + 1/np.cos(np.radians(tilt_angle))**bpower - 1) 
    return cos_centre_bias

def power_drop(tilt_angle, power, ppower):
    """
    used within biased_cc and cc_walk
    """
    cos_power = power * np.cos(np.radians(tilt_angle))**ppower 
    return cos_power

def distance_bias(centre_x, centre_y, interp, limit, centre_bias,
                  power, invert = False):
    """
    Intended as a weighting function for 2D CC maps where CC values
    drop with {power} power.
    Higher {centre_bias} makes CC*distance drop faster
    {interp} and {limit} are multiplied to generate box with the same
    size as CC map
    {invert} = True inverts the trend
    Sensible values:
        decent SNR data:
            {power} = 2 and {centre_bias} between 0.1 and 0.5 
            {power} = 3 and {centre_bias} between 0.05 and 0.5 
        noisy data:
            {power} = 2 and {centre_bias} between 0.5 and 0.8
            {power} = 3 and {centre_bias} between 0.5 and 1.5
    MAKE SURE CC VALUES ARE POSITIVE BEFORE MULTIPLYING WITH THIS ARRAY
    WHEN ENTERING PEAK COORDINATES FROM buggy_peaks:
        {centre_x} = peak[0,1], {centre_y} = peak[0,0] 
    """
    #keep power >= 1
    power = max(1, power)
    
    centre_values = (centre_y, centre_x) #Y, X
    size = interp*limit*2
    shift = centre_values[0] - (size/2), centre_values[1] - (size/2)
    shift = np.array(shift)/(size/2.)

    x = np.linspace(1 + shift[0], -1 + shift[0], size)
    y = np.linspace(1 + shift[1], -1 + shift[1], size)
    xx, yy = np.meshgrid(x,y)
    #distance matrix from centre
    xy = np.sqrt(xx**2 + yy**2)
    if invert:
        #if cc values are negative
        d = (xy**power)
        #shift min to 1
        d = d - np.min(d) + 1
    else:
        #power and invert
        d = (xy**power)*-1
        #shift min to 0
        d = d - np.min(d)
    d **=centre_bias
    return d


def make_nonoverlapping_alis(out_dir, base_name, average_map, tomo, ali, tlt,
                             pmodel, csv_file, apix, rsorted_pcls, tomo_size,
                             var_str,
                             num_cores, pmodel_bin = False,
                             grey_dilation_level = 5, 
                             average_volume_binning = 1,
                             lamella_mask_path = False):

    #made obsolte by the processchunk version
    
    """
    
    Inputs:
        out_dir [str]
        
        average_map [str] - average volume used for plotback, white on black
                            (generally chimera segmented volume)
        
        apix [float]
        
        pmodel [str] path to imod 3d model - has to be identical to model used
        to generate rsorted_pcls!!!!
        
        rsorted_pcls [np.ndarray] particle coordinates in a very specific format....
    Output:
        plotback - oriented edgeon (i.e. same orientation as IMOD _full.rec)
        
        
        """
        
    average_apix, average_size = get_apix_and_size(average_map)
    if average_volume_binning != 1:
        average_apix /= average_volume_binning
        average_size = np.array(np.array(average_size, dtype = 'float16')
        * average_volume_binning, dtype = 'int16')
    if np.round(average_apix, decimals = 1) != np.round(apix, decimals = 1):
        print ('WARNING:\n Average map pixel size does not match input ' + 
                ' tomogram pixel size: %s %s!' % (average_apix, apix))
    print 'Using average map size to generate non overlapping models.'
    
    
    (outmods, novlp_sorted_pcls, pcl_ids, groups, remainder, ssorted_pcls
     ) = make_non_overlapping_pcl_models(
                                        rsorted_pcls,
                                        average_size,
                                        out_dir
                                        ) 
    #names of input and output files
    plotback_list = []
    sub_ali_list = []
    masked_tomo_list = []
    out_mask_list = []
    smooth_mask_list = []
    plotback_ali_list = []

    for x in range(len(groups)):
        plotback_list.append(join(out_dir, 'plotback_%02d.mrc' % x))
        out_mask_list.append(join(out_dir, 'mask_%02d.mrc' % x))
        smooth_mask_list.append(join(out_dir, 'smooth_mask_%02d.mrc' % x))
        sub_ali_list.append(join(out_dir, 'subtracted_%02d.mrc' % x))
        masked_tomo_list.append(join(out_dir, 'masked_%02d.mrc' % x))
        plotback_ali_list.append(join(out_dir, 'plotback_%02d.ali' % x))
    #reading into memory for parallelisation
    average_map = MapParser.readMRC(average_map) 
    csv_file = PEETMotiveList(csv_file)
    print 'making backplots'
    pmodel = PEETmodel(pmodel).get_all_points()    
    #make plotbacks
    _ = Parallel(n_jobs = num_cores, verbose = 2)(delayed(replace_pcles)(
            average_map, tomo_size, csv_file, pmodel, plotback_list[a],
                  apix, groups[a], pmodel_bin, 
                  average_volume_binning, True) for a in range(len(groups)))
    print 'reprojecting backplots'
    if lamella_mask_path:
        Parallel(n_jobs=num_cores, verbose = 2)(delayed(mask_and_reproject)(
            out_dir, base_name, lamella_mask_path, plotback_list[a], 
            plotback_list[a], plotback_ali_list[a], ali, tlt,
            tomo_size[2], var_str) for a in range(len(groups)))        
    else:
        Parallel(n_jobs=num_cores, verbose = 2)(delayed(reproject_volume)(
            plotback_list[a], ali, tlt, tomo_size[2], plotback_ali_list[a],
            var_str) for a in range(len(groups)))
    
    #make masks   
    print 'generating masks'
    _ = Parallel(n_jobs = num_cores, verbose = 2)(delayed(
            NEW_mask_from_plotback)(plotback_list[a], out_mask_list[a],
            grey_dilation_level, False, False, smooth_mask_list[a]
            ) for a in range(len(groups)))

    if lamella_mask_path:
        #combined masks with an existing lamella mask
        #making this separately so that the same lamella mask doesn't need to  
        #be written out multiple times
        Parallel(n_jobs = num_cores, verbose = 2)(delayed(mult_mask)(
            lamella_mask_path, smooth_mask_list[a]) for a in range(len(groups)))

#    WHAT IS THIS???????????????????? vp 4 feb 2020
#    for x in range(len(out_mask_list)):
#        tmpout = join(out_dir, 'tmpmask_%02d.mrc' % x)
#        check_output('bint -translate 2,2,2 %s %s' % (out_mask_list[x], tmpout), shell = True)
#        os.rename(tmpout, out_mask_list[x])
        
    print 'reprojecting masked tomos'
    Parallel(n_jobs=num_cores, verbose = 2)(delayed(mask_and_reproject)(
            out_dir, base_name, smooth_mask_list[a], tomo, 
            masked_tomo_list[a], sub_ali_list[a],
             ali, tlt, tomo_size[2], var_str) for a in range(len(groups)))
        
#    _ = Parallel(n_jobs=num_cores, verbose = 2)\
#            (delayed(run_masktomrec)(x, out_mask_list[x], tlt, ali, tomo,
#                                     out_dir, tomo_size, base_name, 
#                                     sub_ali_list[x])
#                    for x in range(len(out_mask_list)))

#        plotback_ali = (join(out_dir, 'plotback_%02d.ali' % x))
#        fs = join(out_dir, 'fs%02d.ali' % x)
#        check_output('mtffilter -l 0.02,0.05 %s %s' % (plotback_ali, fs), shell = True)
#        check_output('bint -invert %s %s' % (fs, fs), shell = True)
#        check_output('newstack -scale 0,1 %s %s' % (fs,fs), shell = True)
#        check_output('clip truncate -hi 0.6 %s %s' % (fs,fs), shell = True)
#        check_output('newstack -scale 0,1 %s %s' % (fs,fs), shell = True)
#        check_output('clip multiply %s %s %s' % (fs, sub_ali_list[x], sub_ali_list[x]), shell = True)

    return (outmods, groups, novlp_sorted_pcls, plotback_list, sub_ali_list,
             out_mask_list, plotback_ali_list)
    
def extract_and_cc(ordinal, ali, plotback, sorted_pcl, box, tlt,
                   zero_tlt, dose, apix, V, Cs, wl, ampC, ps, defocus,
                   butter_order = 4, dosesym = False, orderL = True,
                   pre_exposure = 0, dosefilter = True, limit = 10,
                   interp = 10, smooth = True, centre_bias = 0.1,
                   mask_diameter = 0, debug = 0,
                   unreasonably_harsh_filter = False, test_dir = False,
                   nplots = False, allow_large_jumps = False, 
                   excludelist = False, xf = False):
    thickness_scaling = 1
#    bpower = 1#2
#    ppower = 1#1.1
    step = 5 #think this has to be odd
    #extract
    tilt_angles = [float(x.strip('\n')) for x in open(tlt)]
    read_ali = deepcopy(mrcfile.open(ali).data)
    read_plotback = deepcopy(mrcfile.open(plotback).data)

    query = extract_2d_simplified(read_ali, sorted_pcl, box, excludelist)
    ref = extract_2d_simplified(read_plotback, sorted_pcl, box, excludelist)

    #TBD    
    #the point of this is to be able to re-extract particles from raw stack
    #this will also require offsets caused by integer pixel extraction
    #to be tracked.  extract_2d_simplified should do something like this but
    #it is probably untested
    if xf:
        tmat = []
        with open(xf) as f1:
            for x in f1.readlines():
                tmat.append([float(y) for y in x.split()])
        tmat = np.array(tmat)[:,0:4]

    if test_dir:
        if ordinal > 99:
            testpcl = join(test_dir, 'testquery%04d.mrc' % ordinal)
            testref = join(test_dir, 'testref%04d.mrc' % ordinal)  
        else:
            testpcl = join(test_dir, 'testquery%02d.mrc' % ordinal)
            testref = join(test_dir, 'testref%02d.mrc' % ordinal)
        
#    if not os.path.isfile(testpcl):
        write_mrc(testpcl, query)
        write_mrc(testref, ref)


#    if not mask_diameter:
#        mask_diameter = int(box[0] - box[0]/10)
#    cmask = create_circular_mask(box, mask_diameter/2)
#    cmask = np.ones((ref.shape)) * cmask
        #sharp circular mask was a bad idea, was causing CC artefacts

    #filter
    if not dosefilter:
        dose = 0
    ref = ctf_convolve_andor_dosefilter_wrapper(ref, zero_tlt, dose, apix, V,
        Cs, wl, ampC, ps, defocus, butter_order, dosesym, orderL, pre_exposure)
    query = ctf_convolve_andor_dosefilter_wrapper(query, zero_tlt, dose, apix,
        V, Cs, wl, ampC, ps, 0, butter_order, dosesym, orderL, pre_exposure)

    if test_dir:
        if ordinal > 99:
            testpcl = join(test_dir, 'ftestquery%04d.mrc' % ordinal)
            testref = join(test_dir, 'ftestref%04d.mrc' % ordinal)  
        else:
            testpcl = join(test_dir, 'ftestquery%02d.mrc' % ordinal)
            testref = join(test_dir, 'ftestref%02d.mrc' % ordinal)
        write_mrc(testpcl, query)
        write_mrc(testref, ref)
    
    if unreasonably_harsh_filter:
        for x in range(len(query)):
            query[x] = filter_tilts_butter_p(query[x], 2000, apix)
            ref[x] = filter_tilts_butter_p(ref[x], 2000, apix)
        testout = '/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/test_new_toy_tomo/check_pcls/testpcl.mrc'
        if not os.path.isfile(testout):
            write_mrc(testout, query)
            write_mrc('/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/test_new_toy_tomo/check_pcls/testref.mrc', ref)

    #reduce number of plots being written
    if not isinstance(nplots, bool):
        if np.any(np.isin(nplots, ordinal)):
            debug = 2
        else:
            debug = 1

    
    #CC          
    shifts, debug_out, ccmaps = cc_but_better(
            ordinal, ref, query, limit, interp, tilt_angles,
            zero_tlt, centre_bias, thickness_scaling, step, test_dir,
            debug, allow_large_jumps)
    if debug > 0:
        log = join(test_dir, 'pcl%02d.log' % ordinal)
        debug_out = np.ravel(debug_out)
        with open(log, 'w') as l:
            for o in range(len(debug_out)):
                l.write('\n%s' % debug_out[o])
    
#    shifts, debug_out, ccmaps = CC(
#            ref * cmask, query * cmask, limit, interp, scaling)
#    if debug  > 1:
#        f, axes = plt.subplots(ref.shape[0], 2, figsize = (2, ref.shape[0]))
#        for x in range(ref.shape[0]):        
#            axes[x, 0].imshow(ref[x])
#            axes[x, 1].imshow(query[x])
#        plt.show()
    return shifts, debug_out#, ref, query, ccmaps

def shifts_to_model_points(ali, shifts, out_dir, rec_dir, zero_tlt, base_name,
            xf, tomo_binning, rsorted_pcls, debug = 0, excludelist = False):
    if debug > 1:
        plot_shifts(shifts, out_dir)
    
    if not isinstance(excludelist, bool):
        #trim rsorted_pcls using excludelist (shifts would have been trimmed
        excludelist = np.array(excludelist) - 1
        exc_mask = np.isin(range(len(rsorted_pcls)), excludelist, invert = True)
        rsorted_pcls = rsorted_pcls[exc_mask]

    np.save(abspath(join(out_dir, 'shifts.npy')), shifts)  
#    #i think this is an artefact of using sorted_pcls (generated from 
#    #rawtiltcoords)    
#    # need refined rotation angle:
    rotation = check_output('xf2rotmagstr %s' % xf, shell = True)
    rotation = np.median(np.array(
            [float(x.split()[2][0:-1]) for x in rotation.split('\n')[2:-1]]))
#    #backrotate shifts to match unrotated tomo.  Need negative of rotation angle from xf
#
#    shifts = rotate_shifts(shifts, rotation) 
#    np.save(abspath(join(out_dir, base_name + '_rotated_shifts.npy')), shifts)  

#    #scale shifts to bin 1
#    scaled_shifts = shifts * tomo_binning
#    np.save(abspath(join(out_dir, base_name + 'multiplied_shifts.npy')), shifts) 

    #pcls = np.swapaxes(rsorted_pcls, 0, 1)
    # writing out flexo fiducial model
    print 'Writing unwarped particle model...'

    #shifts subtracted from model point positions
    ali_pcls = np.swapaxes(rsorted_pcls, 0, 1)  
#    shifts[:,:,0] = -shifts[:,:,0]
    
#    c = '/raid/fsj/grunewald/vojta/ribosome_flexo/11/point_tomo/yshifted/p.xf'
#    c1 = []
#    with(open(c)) as f:
#            for y in f.readlines():
#                t1 = [float(y.split()[4]), float(y.split()[5])]
#                c1.append(t1)
#    c1 = -np.array(c1) 
#    shifts[:] = c1

    ali_cont_pcl = np.dstack((ali_pcls[:,:,0:2]
                    - shifts[:,:,0:2], ali_pcls[:,:,2]))
    outmod = PEETmodel() 
    for p in range(len(ali_cont_pcl)):
        if p != 0:
                outmod.add_contour(0)
        for r in range(ali_cont_pcl.shape[1]):
            outmod.add_point(0, p, ali_cont_pcl[p,r])
    ali_outmod = abspath(join(out_dir, 'flexo_ali.fid'))
    outmod.write_model(ali_outmod)

#    print 'writing out testing_flexo_particles.mod'
#    outmod_name = abspath(join(out_dir, 'flexo_particles.mod'))
#    outmod.write_model(outmod_name)
#    

    check_output('imodtrans -I %s %s %s' % 
                 (ali, ali_outmod, ali_outmod), shell = True)
    return ali_outmod, rotation, ali_pcls
