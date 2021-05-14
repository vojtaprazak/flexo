execfile('/beegfs/cssb/user/prazakvo/testing/flexo/flexo_tomo02/flexo.prm')
execfile('/beegfs/cssb/user/prazakvo/testing/flexo/flexo_tomo02/iteration_1/tmp_output/tmp_output_file.py')
execfile('/beegfs/cssb/user/prazakvo/software/flexo/definite_functions_for_flexo.py')

particles = extracted_particles(ssorted_pcls, apix = 13.64, out_dir = out_dir, excludelist = excludelist, base_name = base_name, chunk_base = "tomo02_flexo", n_peaks = 100, groups = groups, tilt_angles = tlt)
particles.get_shift_magnitude()

from geneticalgorithm import geneticalgorithm as ga
num_peaks = 5
num_pcles = 10
#random.seed(12)

subs = particles.shifts[:, :, :num_peaks]
subcc = particles.cc_values[:, :, :num_peaks]

pcles = particles.model_3d
shifts =  subs[tilt_n, :num_pcles].reshape(num_pcles*num_peaks, 2)
ccs = 1 - subcc[tilt_n, :num_pcles]


kdtree = KDTree(pcles)
sm = kdtree.sparse_distance_matrix(kdtree, max_distance = 70)

kdtree2 = KDTree(shifts)
shiftm = kdtree2.sparse_distance_matrix(kdtree2, max_distance = 10)

def f(X):
    score = 0
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            cc_weight = ccs[i, int(X[i])]*ccs[j, int(X[j])]
            dist_weight = 1./(sm[i, j])
            shift_dist = shiftm[i*num_peaks+X[i], j*num_peaks+X[j]]
            score += cc_weight*dist_weight*shift_dist
    return score


varbound=np.array([[0,4]]*num_pcles)

model=ga(function=f, dimension=num_pcles, variable_type='int', variable_boundaries=varbound)

model.run()

# old [2. 3. 1. 4. 3. 1. 3. 3. 3. 2.]
