import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

def moran(N, num_A_init, r=1, max_iter=np.inf):
    """
    Args:
        N (int): population size (i.e. total number of individuals)
        num_A_init (int): initial number of individuals of type "A"
        r (numeric): fitness of individuals of type "A"
            (fitness of type "B" individuals is assumed to be 1)
        max_iter (numeric): maximum number of iterations
    """
    num_A = [num_A_init]
    n_iter = 0
    while (num_A[-1] > 0) and (num_A[-1] < N) and n_iter < max_iter:
        n_iter += 1
        i = num_A[-1]
        prob_inc_A = (r * i) * (N - i) / ((r * i + N - i) * (N))
        prob_dec_A = (N - i) * (i) / ((r * i + N - i) * (N))
        weights = [prob_inc_A, prob_dec_A, 1-(prob_inc_A + prob_dec_A)]
        decision = np.random.choice(['inc_A','dec_A','same'], size=1, p=weights)
        if decision == 'inc_A':
            num_A.append(i+1)
        if decision == 'dec_A':
            num_A.append(i-1)
        if decision == 'same':
            num_A.append(i)
    return num_A

def plot_selection(n_trials, population_size, init_num_A, r, max_iter, seed=None):
    if type(seed) == type(None):
        seed = int(str(time.time()).replace('.','')[-6:])
    np.random.seed(seed)
    trajectories = []
    total_time = []
    A_fixes = []
    for i in range(n_trials):
        traj = moran(population_size, init_num_A, r, max_iter)
        trajectories.append(traj)
        total_time.append(len(traj))
        A_fixes.append(traj[-1] == population_size)
    total_time = np.array(total_time)
    A_fixes = np.array(A_fixes)

    plt.figure(figsize=[10,6])
    for t_idx, traj in enumerate(trajectories):
        plt.plot(np.array(traj)/population_size)
        if traj[-1] == population_size:
            plt.scatter([len(traj)-1], [1.05], s=50, marker="|",
                        color='blue')
        if traj[-1] == 0:
            plt.scatter([len(traj)-1], [-0.05], s=30, marker="|",
                        color='red')
    plt.xlabel("time steps", fontsize=16)
    plt.xticks(size=14)
    plt.ylabel("mutant allele frequency", fontsize=16)
    plt.yticks(size=14)
    plt.title(f'Trajectories (mutation took over in {sum(A_fixes)} of {n_trials} trials)',
              fontsize=16)
    plt.show()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.figure(figsize=[10,4.5])
        sns.distplot(total_time[A_fixes], kde=False, color='blue')
        sns.distplot(total_time[np.logical_not(A_fixes)], kde=False, color='red')
        plt.legend(['mutation took over', 'mutation went extinct'], fontsize=14)
        plt.xlabel("stopping time", fontsize=16)
        plt.xticks(size=14)
        plt.ylabel("trials", fontsize=16)
        plt.yticks(size=14)
        plt.title('Stopping time distributions', fontsize=16)
        plt.show()

### moran process with clonal interference

def recursive_fxn(a, list_, n):
    if len(list_) == n: # base case
        a.append(tuple(list_))
    else: # recursive case
        recursive_fxn(a, list_ + [0], n)
        recursive_fxn(a, list_ + [1], n)
def get_bin_perms(n):
    # get a list of all binary vectors of length n
    a = []
    recursive_fxn(a, [], n)
    return a

def get_gtype_mut_to(gtype_mut_from):
    mut_allele = np.random.choice(len(gtype_mut_from))
    gtype_mut_to = list(gtype_mut_from)
    gtype_mut_to[mut_allele] = np.logical_not(gtype_mut_to[mut_allele]).astype('int')
    return tuple(gtype_mut_to)

def get_dim_slices(n_dims):
    # get a bunch of slicing objects for indexing "sides" of many-dimensional
    # tensors
    a = []
    start = tuple(slice(2) for _ in range(n_dims))
    for i in range(n_dims):
        a_ = list(start)
        a_[i] = 1
        a.append(a_)
    return a

def moran_clonal_int(n_loci, mu, pop_size, r, n_iter):
    """
    Args:
        n_loci (int): number of loci under mutation and selection
        mu (numeric): for each cell and each locus at each generation, that cell
                        will suffer a mutation at that locus with probability
                        mu
        pop_size (int): the number of cells in the population.
        r (numeric): relative fitness of mutant over wild-type; fitness effects
                        in this simulation are additive: for a two-locus
                        population where r=1.15, wild-type cells have fitness=1,
                        mutants at one of the two loci have fitness=1.15, and
                        cells with mutations at both loci have fitness=1.3.
        n_iter (int): number of birth/replacement iterations to run
    
    Returns:
        a matrix where rows are iterations and columns are number of mutant
        individuals at each locus
    """
    # initialization
    mu = mu * n_loci # mutation rate is per-locus
    state = np.zeros([2 for _ in range(n_loci)]) # current allele distribution
    state[tuple(0 for _ in range(n_loci))] = pop_size # initial allele distribution
    gtypes = get_bin_perms(n_loci) # list of possible genotypes.
    n_gtypes = len(gtypes) # number of possible genotypes. n_gtypes = 2 ** n_loci
    advantage = np.sum(np.array(gtypes), axis=1) # fitness advantage of muts over wt
    weights = (r - 1) * advantage + 1 # probabilty weights for cell division
    dim_slices = get_dim_slices(n_loci) # list of dimension slices for the ledger
    ledger = np.zeros([n_iter, n_loci]) # ledger to record allele frequencies
    for iteration in range(n_iter):
        # record in ledger
        for i in range(n_loci):
            ledger[iteration, i] = np.sum(state[dim_slices[i]])
        # a single moran step
        p_divide = np.ravel(state) * weights
        p_divide = p_divide / np.sum(p_divide)
        idx_divide = np.random.choice(n_gtypes, p=p_divide)
        p_replaced = np.ravel(state) / pop_size
        idx_replaced = np.random.choice(n_gtypes, p=p_replaced)
        state[gtypes[idx_divide]] += 1
        state[gtypes[idx_replaced]] -= 1
        # a single mutation step
        n_muts = np.random.binomial(pop_size, mu)
        for _ in range(n_muts):
            p_mut = np.ravel(state) / pop_size
            gtype_mut_from = gtypes[np.random.choice(n_gtypes, p=p_mut)]
            gtype_mut_to = get_gtype_mut_to(gtype_mut_from)
            state[gtype_mut_from] -= 1
            state[gtype_mut_to] += 1
    return ledger

# make plots for moran process with clonal interference

def plot_mutation_selection(color, n_replicates, n_loci, mu, pop_size,
               r, n_steps, seed=None):
  if type(seed) == type(None):
    seed = int(str(time.time()).replace('.','')[-6:])
  np.random.seed(seed)
  plt.figure(figsize=[10,6])
  for i in range(n_replicates):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger = moran_clonal_int(n_loci, mu, pop_size, r, n_steps)
    plt.plot(ledger / pop_size, color=color)
    plt.ylim([0, 1])
  plt.xlabel("time steps", fontsize=16)
  plt.xticks(size=14)
  plt.ylabel("mutant allele frequency", fontsize=16)
  plt.yticks(size=14)
  title = f' {n_replicates} replicate of mutation-selection process with {n_loci} genes'
  if n_replicates > 1:
    title = title.replace('replicate', 'replicates')
  if n_loci > 1:
    title = title.replace('gene', 'genes')
  plt.title(title, fontsize=16)
  plt.show()