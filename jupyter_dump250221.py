# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
execfile('/raid/fsj/grunewald/vojta/software/scripts/definite_functions_for_flexo.py')
os.chdir('/raid/fsj/grunewald/vojta/tetrapod_model/independent_flexo_model_from_peet_testing/nec_test18_tomo10/iteration_1/')
execfile('../input_for_flexo_model_form_peet.py')
execfile('./tmp_output/tmp_output_file.py')


particles = extracted_particles(ssorted_pcls, apix = 13.64, out_dir = out_dir, excludelist = excludelist, base_name = base_name, chunk_base = "tomo10_flexo", n_peaks = 100, groups = groups, tilt_angles = tlt)
#particles.remove_tilts_using_excludelist()
#particles.remove_group_outliers()
#particles.read_cc_peaks()
#particles.read_3d_model()
particles.get_shift_magnitude()
#particles.apix = 13.64
#print np.sum(np.sum(particles.groups, axis = 0, dtype = bool)), particles.groups.shape

fparticles = extracted_particles(ssorted_pcls, apix = 13.64, out_dir = out_dir, excludelist = excludelist, base_name = base_name, chunk_base = "tomo10_flexo", n_peaks = 100, groups = groups, tilt_angles = tlt)
#fparticles.remove_tilts_using_excludelist()
#fparticles.remove_group_outliers()
fparticles.read_cc_peaks(name_ext = 'f_peaks')
#fparticles.read_3d_model()
fparticles.get_shift_magnitude()



particles.classify_initial_shifts(plot = True, out_dir = False, training_peak_indices = [0,-1], shift_apix_weight1 = 2)#, max_input_values = 1000)


#also it's should be perfectly reasonable to have a hard limit of ...10% of the box? seems small
fparticles.classify_initial_shifts(plot = True, out_dir = False, gap_mask = True, training_peak_indices = [0, -1])



b1 = particles.cc_values[particles.shift_mask]
b2 = np.absolute(np.mean(particles.shifts[particles.shift_mask], axis = 1))


def compress_masked_array(vals, axis=-1, fill=1000):
    #https://stackoverflow.com/questions/46354509/transfer-unmasked-elements-from-maskedarray-into-regular-array
    cnt = vals.mask.sum(axis=axis)
    shp = vals.shape
    num = shp[axis]
    mask = (num - cnt[..., np.newaxis]) > np.arange(num)
    n = fill * np.ones(shp)
    n[mask] = vals.compressed()
    n = np.ma.masked_where(n == fill, n)
    return n
startTime = time.time() 
b1 = compress_masked_array(np.ma.masked_array(particles.cc_values, np.logical_not(particles.shift_mask)))
b2 = compress_masked_array(np.ma.masked_array(particles.shifts, axis = 3, 
                                              mask = np.repeat(np.logical_not(particles.shift_mask)[..., None], 2, axis = -1)))
print time.time() - startTime                                                                                                             
test = np.mean(b1[10:13,:,0], axis = (0))
print test.shape
plt.plot(test)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ConstantKernel as C
def gpr(xyz, dxdy, ccc, figsize = (16,10), test = False):
    
    std_dxdy = np.std(dxdy)
    med_mask = np.ma.masked_where(np.absolute(dxdy) < 5*std_dxdy, dxdy).mask[:, 0]
    xyz = xyz[med_mask]
    dxdy = dxdy[med_mask]
    
    if test:
        dxdy -= np.mean(dxdy, axis = 0)

    
    #to be included
    #xyz = xyz*apix/10
    #dxdy = dxdy*apix/10
    
    
    
    #this may backfire if all particles are concentrated in one small spot....:
    max_len = np.max(xyz)/4  ###########maybe use volume size instead
    

    
    #this protects from RBF forming a spike and return to mean with sparse points
    sort_coord = np.sort(xyz)
    min_len = np.mean(sort_coord[:, 1:] - sort_coord[:, :-1])/4
    #tmp = np.zeros((dxdy.shape[0], 3))
    dxdy = np.hstack((dxdy, np.zeros((dxdy.shape[0], 1))))
    #could use structure size (better correlation) or 
    var = np.std(dxdy)**2
    
    #min_len = 1e-1
    max_len = 80
####min/max probably needs to be tweaked based on avearging results
    
    #
    #kernel =RBF(min_len, (min_len, max_len))+ WhiteKernel(noise_level = 0.5, noise_level_bounds=(1e-1,1))
    #gp = GaussianProcessRegressor(kernel=kernel, alpha = var, n_restarts_optimizer=9)
    
    kernel =RBF(min_len, (min_len, max_len))+ WhiteKernel(noise_level = 0.5, noise_level_bounds=(1e-1,1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    
    gp.fit(xyz, dxdy)  


    print gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)
    #X, Y, Z = numpy.mgrid[np.min(xyz[:,0]):np.max(xyz[:,0]):100j,
    #                      np.min(xyz[:,1]):np.max(xyz[:,1]):100j,
    #                      np.min(xyz[:,2]):np.max(xyz[:,2]):100j]
    #XYZ = np.vstack((np.ravel(X), np.ravel(Y), np.ravel(Z))).T
    
    pred, sigma = gp.predict(xyz, return_std = True)
    print np.mean(sigma)
    
    
    if True:
        f, ax = plt.subplots(figsize = figsize)
        ax.quiver(xyz[:, 0], xyz[:, 1] ,pred[:,0], pred[:,1], color = 'r', alpha = 0.8)
        #ax.quiver(xyz[:, 0], xyz[:, 1] ,pred[:,0]+sigma, pred[:,1]+sigma, color = 'b', alpha = 0.5)
        #ax.quiver(xyz[:, 0], xyz[:, 1] ,pred[:,0]-sigma, pred[:,1]-sigma, color = 'g', alpha = 0.5)
        ax.quiver(xyz[:, 0], xyz[:, 1] ,dxdy[:,0], dxdy[:,1], alpha = 0.3)
    if False:
        f2,ax2 = plt.subplots(figsize = figsize)
        ax2 = plt.subplot(111, projection = '3d')
        ax2.quiver(xyz[:, 0], xyz[:, 1], xyz[:, 2], pred[:,0], pred[:,1], np.zeros(pred.shape[0]),
                   length=200)

    
    if True:
        f3,ax3 = plt.subplots(1, 6, figsize = (12,2))
        ax3[0].hist2d(xyz[:, 2], dxdy[:,0])
        ax3[1].hist2d(xyz[:, 2], dxdy[:,1])
        ax3[2].hist2d(xyz[:, 2], pred[:,0])
        ax3[3].hist2d(xyz[:, 2], pred[:,1])
        ax3[4].hist2d(dxdy[:,0], pred[:,0])
        ax3[5].hist2d(dxdy[:,1], pred[:,1])
        ax3[0].set_title('ZdX')
        ax3[1].set_title('ZdY')
        ax3[2].set_title('ZpredX')
        ax3[3].set_title('ZpredY')
        ax3[4].set_title('dXpredX')
        ax3[5].set_title('dYpredY')
        
    if False:
        f4, ax4 = plt.subplots(2, 1,figsize = figsize)
        orderx = np.argsort(pred[:, 0])
        ordery = np.argsort(pred[:, 1])
        ax4[0].plot(range(len(pred)), pred[:, 0][orderx])
        ax4[0].scatter(range(len(pred)), dxdy[:, 0][orderx])
        ax4[0].fill_between(range(len(pred)), pred[:, 0][orderx] + sigma[orderx], pred[:, 0][orderx] - sigma[orderx],
                           alpha = 0.2)
        
        ax4[1].plot(range(len(pred)), pred[:, 1][ordery])
        ax4[1].scatter(range(len(pred)), dxdy[:, 1][ordery])
        ax4[1].fill_between(range(len(pred)), pred[:, 1][ordery] + sigma[ordery], pred[:, 1][ordery] - sigma[ordery],
                           alpha = 0.2)
    
    if True:
        f5,ax5 = plt.subplots(1, 4, figsize = (8,2))
        scaled_pred = pred[:, :2]*(np.percentile(dxdy[:, :2], 90, axis = 0)/np.percentile(pred[:, :2], 90, axis = 0))
        sub = np.absolute(dxdy[:, :2] - scaled_pred)
        max_o_sub = dxdy[:, :2]/sub
        ax5[0].hist2d(ccc, sub[:, 0], 8)
        ax5[1].hist2d(ccc, sub[:, 1], 8)
        ax5[2].hist2d(dxdy[:,0], sub[:,0], 8)
        ax5[3].hist2d(dxdy[:,1], sub[:, 1], 8)
        
        ax5[0].set_title('ccc vs X resid')
        ax5[1].set_title('ccc vs Y resid')
        ax5[2].set_title('dX vs X resid')
        ax5[3].set_title('dY vs Y resid')
    
    return gp

def prep_gpr(ma_shifts, ma_ccc, tilt_subset = [], zmask = False):
    if ma_shifts.ndim > 3:
        #when multiple peaks are passed
        ma_shifts = ma_shifts[:, :, 0]
        
    med_ma_shifts = np.ma.median(ma_shifts[tilt_subset], axis = 0)
    med_ma_ccc = np.ma.median(ma_ccc[tilt_subset], axis = 0)
    
    mask = np.logical_not(med_ma_shifts[:, 0].mask)
    
    med_ma_shifts = med_ma_shifts.data[mask]
    med_ma_ccc = med_ma_ccc[mask]
    ma_model_3d = particles.model_3d[mask]

    
    if not isinstance(zmask, bool):
        bot_mask = ma_model_3d[:, 2] > zmask[0]
        top_mask = ma_model_3d[:, 2] < zmask[1]
        zmask = np.logical_and(bot_mask, top_mask)
        med_ma_shifts = med_ma_shifts[zmask]
        med_ma_ccc = med_ma_ccc[zmask]
        ma_model_3d = ma_model_3d[zmask]
    
    
    #ma_model_3d = np.ma.masked_array(particles.model_3d, 
    #                                 mask = np.repeat(np.logical_not(med_ma_shifts[:, 0].mask)[..., None], 3, axis = -1))                                                                                                   
    return ma_model_3d, med_ma_shifts, med_ma_ccc

b1 = compress_masked_array(np.ma.masked_array(particles.cc_values, np.logical_not(particles.shift_mask)))
b2 = compress_masked_array(np.ma.masked_array(particles.shifts, axis = 3, 
                                              mask = np.repeat(np.logical_not(particles.shift_mask)[..., None], 2, axis = -1)))

m3d, ms, mcc = prep_gpr(b2[:, :, 0], b1[:, :, 0], np.arange(15,16), zmask = [0,150])
ms -= np.mean(ms, axis = 0)

def rand_sub(X,y,z, frac = 0.2):
    rand_mask = np.random.choice([True, False], size = len(X), p = [frac, 1 - frac])
    not_rand_mask = np.logical_not(rand_mask)
    return X[rand_mask], y[rand_mask], z[rand_mask], X[not_rand_mask], y[not_rand_mask], z[not_rand_mask]
sub = True
if sub:
    m3d1, ms1, mcc1, m3d2, ms2, mcc2 = rand_sub(m3d, ms, mcc)

gp = gpr(m3d, ms, mcc, test = False)
gp1 = gpr(m3d1, ms1, mcc1, test = False)
gp2 = gpr(m3d2, ms2, mcc2, test = False)





all_gp1 = gp1.predict(m3d)
all_gp2 = gp2.predict(m3d)
all_gp = gp.predict(m3d)
print np.std(all_gp1, axis = 0)

f, ax = plt.subplots(figsize = (18,12))
ax.quiver(m3d[:, 0], m3d[:, 1] ,all_gp1[:,0], all_gp1[:,1], alpha = 0.8, color = 'k')
ax.quiver(m3d[:, 0], m3d[:, 1] ,all_gp2[:,0], all_gp2[:,1], alpha = 0.8, color = 'b')
ax.quiver(m3d[:, 0], m3d[:, 1] ,all_gp[:,0], all_gp[:,1], alpha = 0.8, color = 'r')
f2,ax2 = plt.subplots(1,2)




from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ConstantKernel as C
def sequential_gpr(s_xyz, s_dxdy, figsize = (16,10), test = False, ntilts = 4):
    f, ax = plt.subplots(figsize = figsize)
    
    for x in range(ntilts):
        dxdy = s_dxdy[x]
        xyz = s_xyz[x]
        std_dxdy = np.std(dxdy)
        med_mask = np.ma.masked_where(np.absolute(dxdy) < 5*std_dxdy, dxdy).mask[:, 0]
        xyz = xyz[med_mask]
        dxdy = dxdy[med_mask]

        if test:
            dxdy -= np.mean(dxdy, axis = 0)


        #to be included
        #xyz = xyz*apix/10
        #dxdy = dxdy*apix/10



        #this may backfire if all particles are concentrated in one small spot....:
        max_len = np.max(xyz)/4  ###########maybe use volume size instead



        #this protects from RBF forming a spike and return to mean with sparse points
        sort_coord = np.sort(xyz)
        min_len = np.mean(sort_coord[:, 1:] - sort_coord[:, :-1])/4
        #tmp = np.zeros((dxdy.shape[0], 3))
        dxdy = np.hstack((dxdy, np.zeros((dxdy.shape[0], 1))))
        #could use structure size (better correlation) or 
        var = np.std(dxdy)**2

        #min_len = 1e-1
        max_len = 80

        #
        kernel =RBF(min_len, (min_len, max_len))+ WhiteKernel(noise_level = 0.5, noise_level_bounds=(1e-1,1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha = var,
                                      n_restarts_optimizer=9)
        gp.fit(xyz, dxdy)  


        print gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)
        #X, Y, Z = numpy.mgrid[np.min(xyz[:,0]):np.max(xyz[:,0]):100j,
        #                      np.min(xyz[:,1]):np.max(xyz[:,1]):100j,
        #                      np.min(xyz[:,2]):np.max(xyz[:,2]):100j]
        #XYZ = np.vstack((np.ravel(X), np.ravel(Y), np.ravel(Z))).T

        pred, sigma = gp.predict(xyz, return_std = True)



        if True:

            ax.quiver(xyz[:, 0], xyz[:, 1] ,pred[:,0], pred[:,1], color = 'r', alpha = 0.5, scale = 50)
            #ax.quiver(xyz[:, 0], xyz[:, 1] ,pred[:,0]+sigma, pred[:,1]+sigma, color = 'b', alpha = 0.5)
            #ax.quiver(xyz[:, 0], xyz[:, 1] ,pred[:,0]-sigma, pred[:,1]-sigma, color = 'g', alpha = 0.5)
            ax.quiver(xyz[:, 0], xyz[:, 1] ,dxdy[:,0], dxdy[:,1], alpha = 0.2, scale = 50)
        
        #subtract pred from dxdy
        if True:
            f2, ax2 = plt.subplots(1, 2,figsize = (8,3))
            scaled_pred = pred*(np.percentile(dxdy, 90, axis = 0)/np.percentile(pred, 90, axis = 0))
            sub = dxdy - scaled_pred
            bins= np.linspace(np.min((dxdy, scaled_pred)), np.max((dxdy, scaled_pred)), 50)
            ax2[0].hist2d(dxdy[:, 0], scaled_pred[:, 0])
            ax2[1].hist2d(dxdy[:, 1], scaled_pred[:, 1])
        if False:
            f3,ax3 = plt.subplots(1, 2, figsize = (8,4), sharex = True, sharey= True)
            ax3[0].hist2d(dxdy[:,0], pred[:,0])
            ax3[1].hist2d(dxdy[:,1], pred[:,1])

ntilts =4
seq_ms = []
seq_m3d = []
for x in range(ntilts):
    tmp1,tmp2,tmp3 = prep_gpr(b2, b1, [15+x], zmask = [0,150])
    #seq_ms.append(tmp2 - np.mean(tmp2, axis = 0))
    seq_ms.append(tmp2)
    seq_m3d.append(tmp1)

gp = sequential_gpr(seq_m3d, seq_ms, ntilts = ntilts)





fb1 = compress_masked_array(np.ma.masked_array(fparticles.cc_values, np.logical_not(fparticles.shift_mask)))
fb2 = compress_masked_array(np.ma.masked_array(fparticles.shifts, axis = 3, 
                                              mask = np.repeat(np.logical_not(fparticles.shift_mask)[..., None], 2, axis = -1)))

fm3d, fms, fmcc = prep_gpr(fb2[:, :, 0], fb1[:, :, 0], np.arange(15,17), zmask = [0,150])
fms -= np.mean(fms, axis = 0)

def rand_sub(X,y,z, frac = 0.5):
    rand_mask = np.random.choice([True, False], size = len(X), p = [frac, 1 - frac])
    not_rand_mask = np.logical_not(rand_mask)
    return X[rand_mask], y[rand_mask], z[rand_mask], X[not_rand_mask], y[not_rand_mask], z[not_rand_mask]
sub = True
if sub:
    fm3d1, fms1, fmcc1, fm3d2, fms2, fmcc2 = rand_sub(m3d, ms, mcc)

fgp = gpr(fm3d, fms, fmcc, test = False)
fgp1 = gpr(fm3d1, fms1, fmcc1, test = False)
fgp2 = gpr(fm3d2, fms2, fmcc2, test = False)






all_gp = gp.predict(m3d)
fall_gp = fgp.predict(fm3d)


f, ax = plt.subplots(figsize = (18,12))
ax.quiver(m3d[:, 0], m3d[:, 1] ,all_gp[:,0], all_gp[:,1], alpha = 0.8, color = 'r')
ax.quiver(fm3d[:, 0], fm3d[:, 1] ,fall_gp[:,0], fall_gp[:,1], alpha = 0.8, color = 'b')




tlt = 15
f,ax = plt.subplots(1, 3,figsize = (15,7))

mask_b2 = np.logical_not(b2.mask[tlt, :, 0, 0])
testb2 = b2[tlt, :, 0, 0].data[mask_b2]

#remove 1000s
testb2_2 = b2[tlt, :, 1, 0].data[mask_b2]
tmp_mask = testb2_2 < 1000
testb2_2 = testb2_2[tmp_mask]
tmp_pred2 = all_gp[:, 0][tmp_mask]

#remove 1000s
testb2_3 = b2[tlt, :, 2, 0].data[mask_b2]
tmp_mask = testb2_3 < 1000
testb2_3 = testb2_3[tmp_mask]
tmp_pred3 = all_gp[:, 0][tmp_mask]

ax[0].scatter(testb2, all_gp[:, 0], alpha = 0.4)
ax[0].scatter(testb2_2, tmp_pred2, alpha = 0.4)
ax[0].scatter(testb2_3, tmp_pred3, alpha = 0.4)

#ax[0].set_xlim(np.min(testb2), np.max(testb2))
comb_pred = np.hstack((all_gp[:, 0], tmp_pred2, tmp_pred3))
comb_data = np.hstack((testb2, testb2_2, testb2_3))
_= ax[1].hist2d(comb_data, comb_pred, 30)



reg = stats.linregress(testb2, all_gp[:, 0])
print reg.slope
ax[0].plot(testb2, reg.intercept + reg.slope*testb2, c = 'r')
ax[1].plot(testb2, reg.intercept + reg.slope*testb2, c = 'r')
ax[2].plot(testb2, reg.intercept + reg.slope*testb2, c = 'r')
ax[0].plot(all_gp[:, 0], all_gp[:, 0], c = 'k')
ax[1].plot(all_gp[:, 0], all_gp[:, 0], c = 'k')
ax[2].plot(all_gp[:, 0], all_gp[:, 0], c = 'k')

pred_unique = np.unique(comb_pred, return_counts = True)
is3_vals =  pred_unique[0][pred_unique[1] == 3]
is3_mask = np.isin(all_gp[:, 0],is3_vals)
print all_gp[is3_mask, 0]

ax[2].scatter(testb2[is3_mask], all_gp[is3_mask, 0], alpha = 0.4)
ax[2].scatter(b2[tlt, :, 1, 0].data[mask_b2][is3_mask], all_gp[is3_mask, 0], alpha = 0.4)
ax[2].scatter(b2[tlt, :, 2, 0].data[mask_b2][is3_mask], all_gp[is3_mask, 0], alpha = 0.4)

def make_scipy_meshgrid(lo1, lo2, hi1, hi2, num):
    xx = np.linspace(lo1, hi1, num)
    yy = np.linspace(lo2, hi2, num)
    X, Y = np.meshgrid(xx, yy)
    zr = np.vstack([X.ravel(), Y.ravel()]).T
    return X, Y, zr
def reshape_bgmm_inputs(inp):
    if isinstance(inp, tuple) and not isinstance(inp[0], (int, float)):
        if not isinstance(inp[0], np.ndarray):
            inp = [np.array(inp[x]) for x in range(len(inp))]
        if inp[0].ndim == 2:
            if inp[0].shape[1] == 1:
                pass
            elif inp[0].shape[0] == 1:
                inp = np.hstack(inp)
            else:
                inp = [np.ravel(inp[x]) for x in range(len(inp))]
                inp = np.vstack(inp).T
        else:
            inp = np.vstack(inp).T
    else:
        inp = np.array(inp)
        if inp.ndim == 1:
            inp = inp.reshape(-1, 1)
    return inp
def bgmm(inp, max_iter = 500, n_components = 10, weight_concentration_prior = None):
    dm = reshape_bgmm_inputs(inp)
    #if not isinstance(a2, bool):
    #    dm = np.vstack((a1, a2)).T
    #else:
    #    if a1.ndim == 1:
    #        dm = a1.reshape(-1, 1)
    n_components = np.min((n_components, np.max((len(dm)/10, 2))))
    g = BayesianGaussianMixture(n_components=n_components, covariance_type='full', max_iter = max_iter,
                               weight_concentration_prior = weight_concentration_prior).fit(dm)
    return g

c1g = bgmm((testb2, all_gp[:, 0]), n_components = 10)
X, Y, zr = make_scipy_meshgrid(np.min(testb2), np.min(all_gp[:, 0]), np.max(testb2), np.max(all_gp[:, 0]), 50)
Z = np.exp(c1g.score_samples(zr))
Z = Z.reshape(X.shape)
f2,ax2 = plt.subplots(figsize = (10,10))
con = ax2.contour(X, Y, Z,cmap='Spectral')
#ax2.scatter(testb2, all_gp[:, 0], alpha = 0.4)
_= ax2.hist2d(testb2, all_gp[:, 0], 30)
ax2.plot(testb2, reg.intercept + reg.slope*testb2, c = 'r')
ax2.plot(all_gp[:, 0], all_gp[:, 0], c = 'k')





#repeat with filtered particles
n = [0,1,2,3]
#fmn1, fstd1 = np.mean(fparticles.dist_score_matrix), np.std(fparticles.dist_score_matrix)
#fmn2, fstd2 = np.mean(fparticles.dist_matrix), np.std(fparticles.dist_matrix)

f_norm_sc = norm(fparticles.dist_score_matrix) *10
f_norm_dst = norm(fparticles.dist_matrix) *10

#weighted_scores = np.einsum('ijkl,i->ijkl', particles.dist_score_matrix[:, :, :n, :n], cos_weight)
b1 = f_norm_sc[:, :, n]
b1 = b1[:, :, :, n]
#b1 = b1[:, np.logical_not(pcl_mask)]
#b1 = np.ravel(b1)
#b1 = np.ma.masked_where(b1 < 0, b1)

bmask = b1 > 0
print b1.shape
b1 = b1[bmask]
b1_shape = bmask.shape
#b1 = np.ma.compressed(b1)


#bmask = b1 > 0
#b1 = b1[bmask]
b2 = f_norm_dst[:, :, n]
b2 = b2[:, :, :, n]
#b2 = b2[:, np.logical_not(pcl_mask)]
#b2 = np.ravel(b2)
#b2 = b2[bmask]
#b2 = np.ma.masked_array(b2, bmask)
#b2 = np.ma.compressed(b2)
b2 = b2[bmask]
f_init_map_mask, f_init_peak1gmm, f_init_map_accepted_classes = bgmm_cluster(b1, b2, apix = 13.64, n_components = 15,
                                                        shift_apix_weight= 1)#, max_input_values=len(np.ravel(b2)))

b1 = b2 = None

