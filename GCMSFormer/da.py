import numpy as np
import random
from tqdm import tqdm
from scipy import stats
import torch
import pickle


def data_pre(mss, max_mz, maxn, noise):
    """
    Simulates and prepares synthetic spectral data for testing or training.

    Parameters:
    ----------
    mss : np.ndarray
        A matrix of input mass spectra (shape: n_spectra × max_mz).
    max_mz : int
        Number of m/z channels (features) in the spectra.
    maxn : int
        Number of components to simulate (between 1 and 5).
    noise : float
        Proportional noise factor to add to the simulated signal.

    Returns:
    -------
    X : np.ndarray
        Noisy synthetic signal matrix (nums × max_mz).
    X_0 : np.ndarray
        Original noise-free signal matrix (nums × max_mz).
    S : np.ndarray
        Normalized spectra matrix (max_mz × maxn).
    ids4p : list of int
        List of indices used to sample spectra from `mss`.
    total : list
        List of simulation parameters: [ratios, means, sigmas, a, b, nums, maxn]
    """

    # Declare global variables 
    global x0, means, sigmas, nums, ratios, a, b

    # Setup component-specific RT means based on `maxn` value
    if maxn == 5:
        L0 = [random.uniform(0.115, 0.125),
              random.uniform(0.125, 0.15),
              random.uniform(0.15, 0.175),
              random.uniform(0.175, 0.2)]
        L = [random.uniform(0.115, 0.125)]
        for i in range(3):
            L.append(random.choice(L0))
        random.shuffle(L)

        # Define RT means
        m1 = random.randint(100, 300) * 0.001
        means1 = [m1 + sum(L[:i]) for i in range(5)]

        sigmas = random.randint(8, 14) * 0.01
        ratios = [random.uniform(0.3, 2), random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.5, 2),
                  random.uniform(0.3, 2)]
        nums = random.randint(40, 100)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.4, 1.5
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.4, 1.5
            x0 = np.linspace(a, b, nums)
    if maxn == 4:
        L0 = [random.uniform(0.115, 0.145),
              random.uniform(0.145, 0.175),
              random.uniform(0.175, 0.2)]
        L = [random.uniform(0.115, 0.125)]
        for i in range(2):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1 = random.randint(200, 280) * 0.001
        m2 = m1 + L[0]
        m3 = m2 + L[1]
        m4 = m3 + L[2]
        means1 = [m1, m2, m3, m4]
        sigmas = random.randint(8, 14) * 0.01
        ratios = [random.uniform(0.3, 2), random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.3, 2)]
        nums = random.randint(30, 90)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.5
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.3, 1.5
            x0 = np.linspace(a, b, nums)
    if maxn == 3:
        L0 = [random.uniform(0.115, 0.125),
              random.uniform(0.125, 0.15)]
        L = [random.uniform(0.115, 0.125)]
        for i in range(1):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1 = random.randint(200, 280) * 0.001
        m2 = m1 + L[0]
        m3 = m2 + L[1]
        means1 = [m1, m2, m3]
        sigmas = random.randint(8, 14) * 0.01
        ratios = [random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.5, 2)]
        nums = random.randint(20, 80)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.4
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.5, 1.6
            x0 = np.linspace(a, b, nums)
    if maxn == 2:
        L = [random.uniform(0.115, 0.125)]
        m1 = random.randint(200, 280) * 0.001
        m2 = m1 + L[0]
        means1 = [m1, m2]
        sigmas = random.randint(8, 14) * 0.01
        maxn = 2
        ratios = [random.uniform(0.3, 2), random.uniform(0.3, 2)]
        nums = random.randint(20, 70)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.2
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.3, 1.55
            x0 = np.linspace(a, b, nums)
    if maxn == 1:
        m1 = random.randint(200, 280) * 0.001
        means1 = [m1]
        sigmas = random.randint(8, 14) * 0.01
        maxn = 1
        ratios = [random.uniform(0.3, 2)]
        nums = random.randint(20, 70)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.2
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.3, 1.55
            x0 = np.linspace(a, b, nums)

    # Select `maxn` unique spectra from input `mss`
    ids4p = random.sample(range(0, len(mss)), maxn)

    # S: Normalized sampled spectra matrix (max_mz × maxn)
    # C: RT-based Gaussian component matrix (nums × maxn)
    S = np.zeros((max_mz, maxn), dtype=np.float32)
    C = np.zeros((nums, maxn), dtype=np.float32)
    for i in range(maxn):
        C[:, i] = stats.norm.pdf(x0, means[i], sigmas) * ratios[i]
        S[:, i] = mss[ids4p[i]] / np.sum(mss[ids4p[i]])

    # Generate noiseless synthetic data
    X_0 = np.dot(C, S.T)

    # Add Gaussian-like noise scaled by `noise` factor
    para = np.random.randn(nums, max_mz)
    E0 = para - np.min(para)
    E = E0 * (np.max(np.sum(X_0, 1)) * noise / (E0.sum() / nums))

    # Normalize noisy signal
    X = np.float32((np.dot(C, S.T) + E) / np.max(np.dot(C, S.T) + E))

    # Package parameters used for generation
    total = ratios + means + [sigmas, a, b, nums, maxn]

    return X, X_0, S, ids4p, total

## REWRITE TO GET ALL DATA AS NEEDED OMG
def update_lib(lib_path_pk, added_path_msp, new_lib_path_pk=None):
    """
    Updates an existing mass spectral library (in pickle format) by adding entries from an MSP file.

    Parameters:
    ----------
    lib_path_pk : str
        Path to the existing library file in pickle format.
    added_path_msp : str
        Path to the MSP file containing new entries to add.
    new_lib_path_pk : str or None, optional
        If provided, the updated library is saved to this new path.
        If None, the original library is modified in memory only.

    Returns:
    -------
    None
    """

    # Load the existing pickle library into a dictionary
    with open(lib_path_pk, 'rb') as file:
        lib_dic = pickle.load(file)

    # Read the entire MSP file content
    with open(added_path_msp, 'r') as file1:
        f1 = file1.read()

    # Find indices of lines starting with 'Name' and 'Num Peaks'
    a = f1.split('\n')
    m = []
    n = []
    for i, l in enumerate(a):
        if l.startswith('Name'):
            m.append(i)
        if l.startswith('Num Peaks'):
            n.append(i)

    # Parse all but the last entry
    for t in range(len(m) - 1):
        mzs = []
        ins = []

        # Extract the m/z and intensity pairs between 'Num Peaks' and next 'Name'
        for j in range(n[t] + 1, m[t + 1] - 1):
            ps = a[j].split('\n ')
            ps = [p for p in ps if len(p) > 0]  #
            for p in ps:
                mz_in = p.split(' ')
                mzs.append(int(float(mz_in[0])))
                ins.append(np.float32((mz_in[1])))

        # Create dictionary structure for this spectrum
        ms_added_dic = {'m/z': mzs, 'intensity': ins}
        ms_dic = {'ms': ms_added_dic}
        name = a[m[t]].split(':')[1].strip()
        lib_dic.update({f'{name}': ms_dic})

    # Handle the last spectrum entry (from last 'Num Peaks' to end of file)
    mzs = []
    ins = []
    for j in range(n[-1] + 1, len(a)):
        ps = a[j].split('\n ')
        ps = [p for p in ps if len(p) > 0]  #
        for p in ps:
            mz_in = p.split(' ')
            mzs.append(int(float(mz_in[0])))
            ins.append(np.float32((mz_in[1])))
    ms_added_dic = {'m/z': mzs, 'intensity': ins}
    ms_dic = {'ms': ms_added_dic}
    name = a[m[-1]].split(':')[1].strip()
    lib_dic.update({f'{name}': ms_dic})

    # Save the updated library if a new path is specified
    if new_lib_path_pk:
        with open(new_lib_path_pk, "wb") as file2:
            pickle.dump(lib_dic, file2, protocol=pickle.HIGHEST_PROTOCOL)
        print('The new library has been successfully saved as a Pickle file')
    file2.close()


def add_msp_MZmine(lib_path_msp, added_path_msp):
    """
    Append the content of one MSP file to another.

    Parameters:
    ----------
    lib_path_msp : str
        File path to the main MSP library to which new entries will be appended.
    added_path_msp : str
        File path of the MSP file containing new spectra to append.
    """
    # Read new MSP content
    with open(added_path_msp, 'r') as file1:
        content = file1.read()
        
    # Append it to the target MSP library file
    with open(lib_path_msp, 'a') as file2:
        file2.write(content)


def read_msp_MZmine(msp_file_path, d_models):
    """
    Parse an MSP file (formatted for MZmine) into retention times and normalized intensity vectors.

    Parameters:
    ----------
    msp_file_path : str
        Path to the MSP file to parse.
    d_models : int
        Unused here but assumed to be kept for interface consistency.

    Returns:
    -------
    RT : list of float32
        List of retention times corresponding to each spectrum.
    mss : np.ndarray
        2D array (n_samples x 1000) of normalized intensity vectors.
    """

    f = open(msp_file_path).read()
    a = f.split('\n')

    # Indices of spectrum headers and peak counts
    m = []
    n = []
    for i, l in enumerate(a):
        if l.startswith('Name'):
            m.append(i)
        if l.startswith('Num Peaks'):
            n.append(i)

    # Preallocate for spectra
    mss = np.zeros((0, 1000), dtype=np.float32)

    # Parse each spectrum except the last
    for t in range(len(m) - 1):
        mzs = []
        ins = []

        # Extract peak lines for current spectrum
        for j in range(n[t] + 1, m[t + 1] - 1):
            ps = a[j].split('\n ')
            ps = [p for p in ps if len(p) > 0]
            for p in ps:
                mz_in = p.split(' ')
                mzs.append(int(float(mz_in[0])))
                ins.append(np.float32((mz_in[1])))

        # Convert to intensity vector
        ms = np.zeros((1, 1000), dtype=np.float32)
        for i, mz in enumerate(mzs):
            ms[0, mz - 1] = ins[i]
        mss = np.vstack((mss, ms / np.max(ms)))

    # Handle last spectrum separately
    mzs = []
    ins = []
    for j in range(n[-1] + 1, len(a)):
        ps = a[j].split('\n ')
        ps = [p for p in ps if len(p) > 0]
        for p in ps:
            mz_in = p.split(' ')
            mzs.append(int(float(mz_in[0])))
            ins.append(float(mz_in[1]))
    ms = np.zeros((1, 1000), dtype=np.float32)
    for i, mz in enumerate(mzs):
        ms[0, mz - 1] = ins[i]
    mss = np.vstack((mss, ms / np.max(ms)))

    # Extract retention times (RT)
    RT = []
    for t in range(len(m)):
        ps = a[m[t] + 2].split('\n ')
        ps = [p for p in ps if len(p) > 0]
        for p in ps:
            mz_rt = p.split(' ')
            RT.append(np.float32((mz_rt[1])))

    return RT, mss

def data_augmentation(msp_file_path, d_models, n, noise_level=0.001):
    """
    Generate synthetic datasets via data augmentation from MSP files.

    Parameters:
    ----------
    msp_file_path : str
        Path to the MSP file for reading the original spectral data.
    d_models : int
        Dimensionality of the model input (used in synthetic data generation and target vocab formatting).
    n : int
        Number of augmented samples to generate.
    noise_level : float, optional (default=0.001)
        Base noise factor to apply when generating synthetic examples.

    Returns:
    -------
    DATA : list of torch.Tensor
        List of input tensors for each synthetic sample.
    TARGET : list of torch.Tensor
        Corresponding target tensors (spectra) for each sample.
    tgt_vocab : torch.Tensor
        Vocabulary of normalized target spectra, with padding, BOS, and EOS tokens.
    TARGET_ind : list of torch.Tensor
        List of index-based representations of TARGET in the tgt_vocab.
    TOTAL : list of torch.Tensor
        Metadata for each sample (ratios, means, sigmas, etc.).
    """

    # Read retention time and spectral data from the MSP file
    RT, mss = read_msp_MZmine(msp_file_path, d_models)
    DATA = []  # Augmented input data
    TARGET = [] # Corresponding true signal (S matrix)
    TOTAL = [] # Metadata for reconstruction or evaluation

    # Generate synthetic data n times
    for i in tqdm(range(n), desc='Generating Dataset'):
        noise = random.randint(1, 50) * noise_level # Add varying noise levels
        maxn = random.randint(1, 5) # Random number of components (1–5)
        data, data_0, target, ids4p, totals = data_pre(mss, d_models, maxn, noise)

        # Store tensors
        DATA.append(torch.tensor(data))
        TARGET.append(torch.tensor(target.T)) # Transpose so shape matches expectations
        TOTAL.append(torch.tensor(totals))

    # Normalize spectra and prepare target vocabulary
    ms = np.zeros_like(mss)
    for i in range(len(mss)):
        ms[i] = mss[i] / np.sum(mss[i]) # Normalize each mass spectrum

    tgt_vocab = torch.tensor(ms, dtype=torch.float)

    # Define special tokens: pad, BOS (beginning of sequence), EOS (end of sequence)
    bos = torch.cat((torch.ones([1, int(d_models / 2)], dtype=torch.float),
                     torch.zeros([1, int(d_models / 2)], dtype=torch.float)), dim=1)

    eos = torch.cat((torch.zeros([1, int(d_models / 2)], dtype=torch.float),
                     torch.ones([1, int(d_models / 2)], dtype=torch.float)), dim=1)

    pad = torch.zeros([1, d_models], dtype=torch.float)

    # Assemble target vocabulary with special tokens
    tgt_vocab = torch.cat((pad, bos, eos, tgt_vocab), dim=0)

    # Convert target tensors into index-based representations (matching rows in tgt_vocab)
    TARGET_ind = []
    for t in TARGET:
        ind = []
        for i in range(len(t)):
            for j in range(len(tgt_vocab)):
                if t[i].equal(tgt_vocab[j]):
                    ind.append(j)
        TARGET_ind.append(torch.tensor(ind))

    return DATA, TARGET, tgt_vocab, TARGET_ind, TOTAL

def data_split(aug_num, d_models, msp_file_path, validation_split):
    """
    Split augmented mass spectrometry data into training, validation, and test sets.

    Parameters:
    ----------
    aug_num : int
        Total number of augmented data samples to generate.
    d_models : int
        Dimensionality of each data vector (used in augmentation).
    msp_file_path : str
        Path to the MSP file containing original spectra.
    validation_split : float
        Fraction of the dataset to reserve for validation (between 0 and 1).
        Test set is always 10% of the data, and the rest is split into train/validation.

    Returns:
    -------
    TRAIN : tuple
        A tuple of (train_src, train_tgt, train_tgt_ind, train_total).
    VALID : tuple
        A tuple of (valid_src, valid_tgt, valid_tgt_ind, valid_total).
    TEST : tuple
        A tuple of (test_src, test_tgt, test_tgt_ind, test_total).
    tgt_vocab : torch.Tensor
        The constructed vocabulary of target vectors including special tokens.
    """
    # Generate augmented data and corresponding labels and metadata
    DATA, TARGET, tgt_vocab, TARGET_ind, TOTAL = data_augmentation(msp_file_path, d_models, aug_num, noise_level=0.001)

    # Split training data
    train_src = DATA[0:round((1 - validation_split) * aug_num)]
    train_tgt = TARGET[0:round((1 - validation_split) * aug_num)]
    train_tgt_ind = TARGET_ind[0:round((1 - validation_split) * aug_num)]
    train_total = TOTAL[0:round((1 - validation_split) * aug_num)]
    TRAIN = tuple((train_src, train_tgt, train_tgt_ind, train_total))

    # Split validation data
    valid_src = DATA[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    valid_tgt = TARGET[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    valid_tgt_ind = TARGET_ind[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    valid_total = TOTAL[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    VALID = tuple((valid_src, valid_tgt, valid_tgt_ind, valid_total))

    # Split test data (always the last 10%)
    test_src = DATA[round(0.9 * aug_num):aug_num]
    test_tgt = TARGET[round(0.9 * aug_num):aug_num]
    test_tgt_ind = TARGET_ind[round(0.9 * aug_num):aug_num]
    test_total = TOTAL[round(0.9 * aug_num):aug_num]
    TEST = tuple((test_src, test_tgt, test_tgt_ind, test_total))

    return TRAIN, VALID, TEST, tgt_vocab

def gen_datasets(para):
    """
    Generate training, validation, and test datasets from a given MSP file using data augmentation.

    Parameters:
    ----------
    para : dict
        Dictionary of parameters expected to include:
        - 'aug_num': int
            Number of augmented samples to generate.
        - 'name': str
            Path to the MSP file to be used for augmentation.
        - 'mz_range': list or tuple of numbers
            Range of m/z values used to determine the model's input dimension.

    Returns:
    -------
    TRAIN : tuple
        Tuple containing training data (sources, targets, target indices, total intensities).
    VALID : tuple
        Tuple containing validation data (same structure as TRAIN).
    TEST : tuple
        Tuple containing test data (same structure as TRAIN).
    tgt_vocab : torch.Tensor
        Tensor representing the target vocabulary (normalized intensity vectors).
    """

    aug_nums = para['aug_num']
    validation_split = 0.2 # Fraction of data to use for validation
    msp_file_path = para['name']
    d_models = int(max(para['mz_range'])) # Model input dimension derived from max m/z value

    # Split augmented data into train/valid/test sets
    TRAIN, VALID, TEST, tgt_vocab = data_split(
        aug_nums, d_models, msp_file_path, validation_split
        )
        
    return TRAIN, VALID, TEST, tgt_vocab
