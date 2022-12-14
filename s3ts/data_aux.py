"""
Auxiliary data preprocessing functions.

@version 2022-12
@author Raúl Coterillo
"""

# sktime
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_UCR_UEA_dataset

# scikit-learn


# numpy / scipy
from scipy.spatial import distance_matrix
from scipy.interpolate import CubicSpline
import numpy as np


from s3ts.data_str import AugProbabilities
import logging

log = logging.Logger(__name__)

# TODO
# ========================================================= #
#                      AUGMENTATIONS                        #
# ========================================================= #

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0, scale=sigma, size=x.shape)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1, scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def time_warp(x, sigma=0.2, knot=4):

    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

# ========================================================= #
#                   AUXILIARY FUNCTIONS                     #
# ========================================================= #

def download_dataset(dataset_name: str) -> None:

    log.info("DOWNLOADING UCR/UEA DATASET")
    log.info("Dataset name:", dataset_name)

    # Download TS dataset from UCR UEA
    X, Y = load_UCR_UEA_dataset(name=dataset_name, 
                            return_type="np2d",
                            return_X_y=True)

    nsamples = X.shape[0]
    s_length = X.shape[1]
    nclasses = len(np.unique(Y))

    log.info("Number of samples:", nsamples)
    log.info("Sample length:", s_length)
    log.info("Number of classes:", nclasses)
    log.info("DOWNLOAD FINISHED")

    try:
        Y = Y.astype(int)
        Y = Y - Y.min()
        mapping = None
    except ValueError:
        mapping = {k: v for v, k in enumerate(np.unique(Y))}
        
    return X, Y, mapping

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_medoids(
        X: np.ndarray, Y: np.ndarray,
        distance_type: str = 'euclidean'
    ) -> tuple[np.ndarray, np.ndarray]: 

    """ Computes the medoids of the classes in the dataset. """

    assert(len(X.shape) == 2)
    s_length = X.shape[1]

    nclasses = len(np.unique(Y))

    # Find the medoids for each class
    medoids = np.empty((nclasses, s_length), dtype=float)
    medoid_ids = np.empty(nclasses, dtype=int)
    for i, y in enumerate(np.unique(Y)):

        index = np.argwhere(Y == y)
        Xi = X[index, :]

        # ...using simple euclidean distance        
        if distance_type == "euclidean":
            medoid_idx = np.argmin(distance_matrix(Xi.squeeze(), Xi.squeeze()).sum(axis=0))
            medoids[i,:] = Xi[medoid_idx,:]
            medoid_ids[i] = index[medoid_idx]

        # ...using Dynamic Time Warping (DTW)
        elif distance_type == "dtw":
            tskm = TimeSeriesKMedoids(n_clusters=1, init_algorithm="forgy", metric="dtw")
            tskm.fit(Xi)
            medoids[i,:] = tskm.cluster_centers_.squeeze()
            medoid_ids[i] = np.where(np.all(Xi.squeeze() == medoids[i,:], axis=1))[0][0]

        else:
            raise NotImplementedError

    return medoids, medoid_ids

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def build_STS(
        X: np.ndarray, 
        Y: np.ndarray, 
        sts_length: int,
        buffer_length: int,
        skip_ids: list[int] = [],
        aug_probs: AugProbabilities = None,
        random_state: int = 0,
        ) -> tuple[np.ndarray, np.ndarray]:

    """
    Builds an STS from and array of samples and labels.

    If sts_length is an int, builds it from randomly picked samples.
    If sts_length is None, builds it with a randomly sampled permutation of them. 
    
    """

    assert(X.shape[0] == Y.shape[0])
    nsamples = X.shape[0]

    assert(len(X.shape) == 2)
    s_length = X.shape[1]

    if sts_length is None:
        STS_X = np.empty((buffer_length + nsamples)*s_length)
        STS_Y = np.empty((buffer_length + nsamples)*s_length)
    else:
        STS_X = np.empty((buffer_length + sts_length)*s_length)
        STS_Y = np.empty((buffer_length + sts_length)*s_length)

    # TODO implement augmentations
    def augment(sample: np.ndarray):
        if rng.random() <= aug_probs.jitter:
            sample = sample
        if rng.random() <= aug_probs.scaling:
            sample = sample
        if rng.random() <= aug_probs.time_warp:
            sample = sample
        if rng.random() <= aug_probs.window_warp:
            sample = sample

    rng = np.random.default_rng(seed=random_state)

    if sts_length is None:
        random_fill = buffer_length
    else:
        random_fill = buffer_length + sts_length

    # fill with randomly picked smples
    for r in range(random_fill):
        while True:
            random_idx = rng.integers(0, nsamples)
            if random_idx in skip_ids:
                continue
            else:
                break

        sample = X[random_idx,:].copy()
        label = Y[random_idx]

        if aug_probs is not None:
            sample = augment(sample)
        
        STS_X[r*s_length:(r+1)*s_length] = sample
        STS_Y[r*s_length:(r+1)*s_length] = label

    # fill with random permutation of samples
    if sts_length is None:

        for r, idx in enumerate(rng.permutation(np.arange(nsamples))):

            r = r + random_fill
            sample = X[idx,:].copy()
            label = Y[idx]

            if aug_probs is not None:
                sample = augment(sample)

            STS_X[r*s_length:(r+1)*s_length] = sample
            STS_Y[r*s_length:(r+1)*s_length] = label

    return STS_X, STS_Y