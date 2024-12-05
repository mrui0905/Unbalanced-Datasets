import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auprc

from util_final import (
    get_mlp,
    normalize_features,
    get_binary_cross_entropy,
    get_binary_accuracy,
    pbt,
    AdamW
)

from sampling_methods import (
    random_undersample,
    smote,
    knn_undersampling,
    tomek_links
)

def get_binary_auprc(
    config: dict,
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute the binary Area Under the Precision-Recall Curve (AUPRC)
    between a label tensor and a logit tensor. This function supports
    arbitrary ensemble shapes.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing settings or hyperparameters. Not
        used in this function.
    logits : torch.Tensor
        The logit tensor, assumed to have shape 
        `ensemble_shape + (dataset_size, 1)`, where `ensemble_shape`
        represents optional leading dimensions for ensemble members.
    labels : torch.Tensor
        The tensor of true labels, which can have shape `(dataset_size,)` 
        or `ensemble_shape + (dataset_size,)`, where `dataset_size` is
        the number of samples.

    Returns
    -------
    torch.Tensor
        A tensor containing the binary AUPRC values for each ensemble member.
        The output has shape `ensemble_shape`.
    """

    logit_positive = logits[..., 0]
    prob_positive = torch.sigmoid(logit_positive)
    true_positives = labels.broadcast_to(
        prob_positive.shape
    ).to(torch.bool)
    if len(logits.shape) == 1:
        num_tasks = 1
    else:
        num_tasks = logits.shape[0]
    return binary_auprc(prob_positive, true_positives, num_tasks=num_tasks)

def load_data(
        config: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess the dataset for training and testing.

    This function reads a dataset from the specified path, extracts the
    training and testing features and labels, and converts them to the
    specified data types and device for computation.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Preprocessed training feature tensor.
        - train_labels (torch.Tensor): Preprocessed training label tensor.
        - test_features (torch.Tensor): Preprocessed testing feature tensor.
        - test_labels (torch.Tensor): Preprocessed testing label tensor.
    """

    dataset = torch.load(config['dataset_path'], weights_only=True)
    train_features, train_labels = dataset['train_features'], dataset['train_labels']
    test_features, test_labels = dataset['test_features'], dataset['test_labels']

    train_features = train_features.to(dtype=config['features_dtype'], device=config['device'])
    train_labels = train_labels.to(dtype=config['labels_dtype'], device=config['device'])
    test_features = test_features.to(dtype=config['features_dtype'], device=config['device'])
    test_labels = test_labels.to(dtype=config['labels_dtype'], device=config['device'])

    return train_features, train_labels, test_features, test_labels

def train_valid_split(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_size: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the training data into training and validation sets.

    This function splits the given training features and labels into
    training and validation sets using the specified test size ratio. It
    ensures that the splits are stratified based on the labels and 
    converted back to PyTorch tensors.

    Parameters
    ----------
    train_features : torch.Tensor
        The training feature tensor to be split.
    train_labels : torch.Tensor
        The training label tensor to be split.
    test_size : float, optional
        The proportion of the dataset to include in the validation split
        (default is 0.1).

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Features for the training set.
        - train_labels (torch.Tensor): Labels for the training set.
        - valid_features (torch.Tensor): Features for the validation set.
        - valid_labels (torch.Tensor): Labels for the validation set.

    """

    train_features_np = train_features.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()

    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features_np, train_labels_np, test_size=test_size, stratify=train_labels_np, random_state=config['seed']
    )

    train_features = torch.tensor(train_features, device=config['device'], dtype=config['features_dtype'])
    valid_features = torch.tensor(valid_features, device=config['device'], dtype=config['features_dtype'])
    train_labels = torch.tensor(train_labels, device=config['device'], dtype=config['labels_dtype'])
    valid_labels = torch.tensor(valid_labels, device=config['device'], dtype=config['labels_dtype'])

    return train_features, train_labels, valid_features, valid_labels

def get_binary_focal_loss(
    config: dict,
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute the focal loss between a label tensor and a logit tensor.

    This function calculates focal loss to address class imbalance by 
    down-weighting easy-to-classify examples and focusing more on hard 
    examples. It supports arbitrary ensemble shapes.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing:
        - 'alpha' (float): Weighting factor for the positive class.
        - 'gamma' (float): Focusing parameter to adjust the impact of 
          easy-to-classify examples.
    logits : torch.Tensor
        The logit tensor, assumed to have shape 
        `ensemble_shape + (dataset_size, 1)`, where `ensemble_shape`
        represents optional leading dimensions for ensemble members.
    labels : torch.Tensor
        The tensor of true labels, which can have shape `(dataset_size,)` 
        or `ensemble_shape + (dataset_size,)`, where `dataset_size` is
        the number of samples.

    Returns
    -------
    torch.Tensor
        A tensor containing the computed focal loss for each ensemble member, 
        with shape `ensemble_shape`.
    """

    alpha, gamma = config["alpha"], config["gamma"]
    
    logits = logits[..., 0]
    labels = labels.broadcast_to(logits.shape)
    
    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction='none'
    )

    probs = torch.sigmoid(logits)
    probs = torch.clip(probs, 1e-7, 1-1e-7)

    p_t = torch.where(labels == 1, probs, 1 - probs)
    modulating_factor = (1 - p_t) ** gamma
    alpha_factor = torch.where(labels == 1, alpha, 1 - alpha)
    focal_loss = alpha_factor * modulating_factor * bce_loss

    return focal_loss.mean(dim=-1)

def grid_search(
    config: dict,
    hyperparameters: list,
    sampling_procedure: callable,
    loss_func: callable = get_binary_cross_entropy
) -> dict:
    """
    Perform a grid search over a set of hyperparameters.

    This function evaluates different hyperparameter configurations using
    a specified sampling procedure and tracks the performance metrics 
    including AUPRC (Area Under the Precision-Recall Curve) and accuracy.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    hyperparameters : list
        A list of hyperparameter configurations to evaluate.
    sampling_procedure : callable
        A function that generates the required data splits 
        (train, validation, and test) for a given hyperparameter configuration.
    loss_func : callable, optional
        The loss function used during training (default is 
        `get_binary_cross_entropy`).

    Returns
    -------
    dict
        A dictionary containing the results for each hyperparameter configuration.
        Each key corresponds to a hyperparameter configuration (as a string),
        and the value is a dictionary with the following keys:
        - "auprc": Maximum AUPRC achieved during testing.
        - "accuracy": Maximum accuracy achieved during testing.
        - "output": Training output (e.g., metrics tracked during training).
    """

    log = {}

    for param in hyperparameters:
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = sampling_procedure(config, param)

        model = get_mlp(config, train_features.shape[-1], 1, 3, 128)
        optimizer = AdamW(model.parameters())

        output = pbt(
            config,
            loss_func,
            get_binary_accuracy,
            model,
            optimizer,
            train_features,
            train_labels,
            valid_features,
            valid_labels
        )

        pred_logits = model(test_features)
        auprc = get_binary_auprc(
            config,
            pred_logits,
            test_labels
        )
        
        accuracy = get_binary_accuracy(
            config,
            pred_logits,
            test_labels
        )

        log[str(param)] = {
            "auprc" : auprc.max().item(),
            "accuracy" : accuracy.max().item(),
            "output" : output
        }

    return log

def baseline(
        config: dict,
        param: tuple | int | str
) -> tuple:
    """
    Generate baseline data splits for training, validation, and testing.

    This function loads the dataset, splits the training data into training
    and validation sets, and normalizes the features across all splits. 
    It prepares the data pipeline for baseline model evaluation.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    param : tuple | int | str
        This is included to maintain compatibility with other sampling procedures, though
        it is not used in this function.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    train_features, train_labels, test_features, test_labels = load_data(config)
    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels, test_size=test_features.shape[0])

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def undersample_random(
        config: dict,
        size: int
) -> tuple:
    """
    Perform random undersampling of the majority class in the training data.

    This function reduces the size of the majority class in the training data
    to balance the class distribution, shuffles the data, and splits it into
    training, validation, and testing sets. The features are normalized across
    all splits.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    size : int
        The size parameter for undersampling, which specifies how many samples
        to retain for the majority class.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    train_features, train_labels, test_features, test_labels = load_data(config)

    positive_mask = train_labels > 0
    positive_features = train_features[positive_mask]
    negative_features = train_features[~positive_mask]
    positive_labels = train_labels[positive_mask]
    negative_labels = train_labels[~positive_mask]

    negative_features, negative_labels = random_undersample(negative_features, negative_labels, size)

    train_features = torch.cat((positive_features, negative_features), dim=0)
    train_labels = torch.cat((positive_labels, negative_labels), dim=0)
    indices = torch.randperm(train_features.shape[0])
    train_features = train_features[indices]
    train_labels = train_labels[indices]

    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels)

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def undersample_tomek(
        config: dict,
        param: int | tuple | str
) -> tuple:
    """
    Perform undersampling using Tomek links to reduce overlapping samples.

    This function removes Tomek links from the training data to enhance class
    separability, splits the data into training, validation, and testing sets,
    and normalizes the features across all splits.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    param : tuple | int | str
        This is included to maintain compatibility with other sampling procedures, though
        it is not used in this function.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    train_features, train_labels, test_features, test_labels = load_data(config)
    train_features, train_labels = tomek_links(config, train_features, train_labels, 1)
    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels)

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def undersample_knn(
        config: dict,
        k: int
) -> tuple:
    """
    Perform undersampling using K-Nearest Neighbors (KNN) to reduce majority class samples.

    This function uses a KNN-based undersampling technique to remove redundant or
    overlapping samples from the majority class, splits the data into training,
    validation, and testing sets, and normalizes the features across all splits.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    k : int The number of nearest neighbors to consider for undersampling.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    train_features, train_labels, test_features, test_labels = load_data(config)
    train_features, train_labels = knn_undersampling(config, train_features, train_labels, 1, k)
    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels)

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def oversample_smote(
        config: dict,
        N: int
) -> tuple:
    """
    Perform oversampling using Synthetic Minority Over-sampling Technique (SMOTE).

    This function generates synthetic samples for the minority class to balance
    the class distribution, splits the data into training, validation, and testing
    sets, and normalizes the features across all splits.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    N : int
        The number of synthetic samples to generate as a percentage of the
        minority class size.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    train_features, train_labels, test_features, test_labels = load_data(config)

    positive_mask = train_labels > 0
    positive_features = train_features[positive_mask]
    negative_features = train_features[~positive_mask]
    positive_labels = train_labels[positive_mask]
    negative_labels = train_labels[~positive_mask]

    positive_features, positive_labels = smote(config, positive_features, positive_labels, 1, N)

    train_features = torch.cat((positive_features, negative_features), dim=0)
    train_labels = torch.cat((positive_labels, negative_labels), dim=0)
    indices = torch.randperm(train_features.shape[0])
    train_features = train_features[indices]
    train_labels = train_labels[indices]

    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels)

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def smote_random_undersample(
        config: dict,
        params: tuple[int, int]
) -> tuple:
    """
    Perform a combination of SMOTE oversampling and random undersampling.

    This function generates synthetic samples for the minority class using SMOTE
    and reduces the majority class using random undersampling to balance the class
    distribution. The resulting dataset is split into training, validation, and
    testing sets, with feature normalization applied across all splits.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    params : tuple[int, int]
        A tuple containing:
        - N (int): The number of synthetic samples to generate as a percentage of the
        minority class size.
        - size (int): The number of majority class samples to retain after 
          random undersampling.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    N, size = params
    train_features, train_labels, test_features, test_labels = load_data(config)

    positive_mask = train_labels > 0
    positive_features = train_features[positive_mask]
    negative_features = train_features[~positive_mask]
    positive_labels = train_labels[positive_mask]
    negative_labels = train_labels[~positive_mask]

    positive_features, positive_labels = smote(config, positive_features, positive_labels, 1, N)
    negative_features, negative_labels = random_undersample(negative_features, negative_labels, size)

    train_features = torch.cat((positive_features, negative_features), dim=0)
    train_labels = torch.cat((positive_labels, negative_labels), dim=0)
    indices = torch.randperm(train_features.shape[0])
    train_features = train_features[indices]
    train_labels = train_labels[indices]

    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels)

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def smote_tomek(
        config: dict,
        N: int
) -> tuple:
    """
    Perform a combination of SMOTE oversampling and Tomek link undersampling.

    This function generates synthetic samples for the minority class using SMOTE
    and removes overlapping samples using the Tomek links algorithm to enhance 
    class separability. The resulting dataset is split into training, validation, 
    and testing sets, with feature normalization applied across all splits.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    N : int
        The number of synthetic samples to generate as a percentage of the
        minority class size.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    train_features, train_labels, test_features, test_labels = load_data(config)

    positive_mask = train_labels > 0
    positive_features = train_features[positive_mask]
    negative_features = train_features[~positive_mask]
    positive_labels = train_labels[positive_mask]
    negative_labels = train_labels[~positive_mask]

    positive_features, positive_labels = smote(config, positive_features, positive_labels, 1, N)

    train_features = torch.cat((positive_features, negative_features), dim=0)
    train_labels = torch.cat((positive_labels, negative_labels), dim=0)
    indices = torch.randperm(train_features.shape[0])
    train_features = train_features[indices]
    train_labels = train_labels[indices]

    train_features, train_labels = tomek_links(config, train_features, train_labels, 1)

    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels)

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def smote_knn(
        config: dict,
        params: tuple[int, int]
) -> tuple:
    """
    Perform a combination of SMOTE oversampling and KNN undersampling.

    This function generates synthetic samples for the minority class using SMOTE
    and removes redundant or overlapping samples from the majority class using
    KNN-based undersampling. The resulting dataset is split into training, 
    validation, and testing sets, with feature normalization applied across 
    all splits.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing the following keys:
        - 'dataset_path' (str): Path to the dataset file.
        - 'features_dtype' (torch.dtype): Desired data type for the features.
        - 'labels_dtype' (torch.dtype): Desired data type for the labels.
        - 'device' (torch.device or str): Device to which tensors should be moved.
    params : tuple[int, int]
        A tuple containing:
        - N (int): The number of synthetic samples to generate as a percentage of the
        minority class size.
        - k (int): The number of nearest neighbors to consider for KNN-based 
          undersampling.

    Returns
    -------
    tuple
        A tuple containing:
        - train_features (torch.Tensor): Normalized training feature tensor.
        - train_labels (torch.Tensor): Training label tensor.
        - valid_features (torch.Tensor): Normalized validation feature tensor.
        - valid_labels (torch.Tensor): Validation label tensor.
        - test_features (torch.Tensor): Normalized testing feature tensor.
        - test_labels (torch.Tensor): Testing label tensor.
    """

    N, k = params
    train_features, train_labels, test_features, test_labels = load_data(config)

    positive_mask = train_labels > 0
    positive_features = train_features[positive_mask]
    negative_features = train_features[~positive_mask]
    positive_labels = train_labels[positive_mask]
    negative_labels = train_labels[~positive_mask]

    positive_features, positive_labels = smote(config, positive_features, positive_labels, 1, N)


    train_features = torch.cat((positive_features, negative_features), dim=0)
    train_labels = torch.cat((positive_labels, negative_labels), dim=0)
    indices = torch.randperm(train_features.shape[0])
    train_features = train_features[indices]
    train_labels = train_labels[indices]

    train_features, train_labels = knn_undersampling(config, train_features, train_labels, 1, k)
    train_features, train_labels, valid_features, valid_labels = train_valid_split(train_features, train_labels)

    normalize_features(
        train_features,
        (valid_features, test_features),
        verbose=False
    )

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

if __name__ == '__main__':
    device = 'mps'
    config = {
        "alpha": 0.25,
        "dataset_path": "creditcard.pt",
        "device": device,
        "ensemble_shape": (64,),
        "features_dtype": torch.float32,
        "gamma": 0.2,
        "labels_dtype": torch.float32,
        "float_dtype": torch.float32,
        "hyperparameter_raw_init_distributions": {
            "epsilon": torch.distributions.Uniform(
                torch.tensor(-10, device=device, dtype=torch.float32),
                torch.tensor(-5, device=device, dtype=torch.float32)
            ),
            "first_moment_decay": torch.distributions.Uniform(
                torch.tensor(-3, device=device, dtype=torch.float32),
                torch.tensor(0, device=device, dtype=torch.float32)
            ),
            "learning_rate": torch.distributions.Uniform(
                torch.tensor(-5, device=device, dtype=torch.float32),
                torch.tensor(-1, device=device, dtype=torch.float32)
            ),
            "second_moment_decay": torch.distributions.Uniform(
                torch.tensor(-5, device=device, dtype=torch.float32),
                torch.tensor(-1, device=device, dtype=torch.float32)
            ),
            "weight_decay": torch.distributions.Uniform(
                torch.tensor(-5, device=device, dtype=torch.float32),
                torch.tensor(-1, device=device, dtype=torch.float32)
            )
        },
        "hyperparameter_raw_perturb": {
            "epsilon": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "first_moment_decay": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "learning_rate": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "second_moment_decay": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "weight_decay": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            )
        },
        "hyperparameter_transforms": {
            "epsilon": lambda log10: 10 ** log10,
            "first_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
            "learning_rate": lambda log10: 10 ** log10,
            "second_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
            "weight_decay": lambda log10: 10 ** log10
        },
        "improvement_threshold": 1e-4,
        "minibatch_size": 128,
        "minibatch_size_eval": 1 << 8,
        "pbt": True,
        "seed": 0,
        "steps_num": 100_000,
        "steps_without_improvement": 1000,
        "valid_interval": 1000,
        "welch_confidence_level": .95,
        "welch_sample_size": 10,
    }

    torch.manual_seed(config["seed"])

    # Baseline
    output = grid_search(config, ['0'], baseline)
    print(f'baseline: {output['0']['auprc']}')
    print('\n')

    # Random Undersample
    train_features, train_labels, test_features, test_labels = load_data(config)
    output = grid_search(config, [int(n * train_labels.sum().item()) for n in range(1, 9)], undersample_random)
    for key in output:
        print(f'{key}: {output[key]['auprc']}')
    print('\n')

    # Undersample Tomek Links
    output = grid_search(config, ['0'], undersample_tomek)
    print(f'tomek: {output['0']['auprc']}')
    print('\n')

    # Undersample KNN
    output = grid_search(config, [k for k in range(50, 250, 50)], undersample_knn)
    for key in output:
        print(f'{key}: {output[key]['auprc']}')
    print('\n')

    # SMOTE
    output = grid_search(config, [n for n in range(2, 11)], oversample_smote)
    for key in output:
        print(f'{key}: {output[key]['auprc']}')
    print('\n')

    # SMOTE w/ Random Undersampling
    params = [(N, int(size * train_labels.sum().item() * N)) for size in range(5, 9) for N in range(2, 11)]
    output = grid_search(config, params, smote_random_undersample)
    for key in output:
        print(key, output[key]['auprc'])
    print('\n')

    # SMOTE w/ Tomek Links
    output = grid_search(config, [n for n in range(2, 11)], smote_tomek)
    for key in output:
        print(key, output[key]['auprc'])
    print('\n')

    # SMOTE w/ KNN
    params = [(N, k) for N in range(9, 11) for k in range(50, 250, 50)]
    output = grid_search(config, params, smote_knn)
    for key in output:
        print(output[key]['auprc'])
    print('\n')

    # Baseline Focal Loss
    for gamma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            config = config | {"alpha": alpha, "gamma": gamma}
            output = grid_search(config, ['0'], baseline, get_binary_focal_loss)
            print(f'gamma: {gamma}, alpha: {alpha}, auprc: {output["0"]["auprc"]}')
    print('\n')

    config['gamma'] = 0.2
    config['alpha'] = 0.25

    # Random Undersample (Focal Loss)
    output = grid_search(config, [int(n * train_labels.sum().item()) for n in range(1, 9)], undersample_random, get_binary_focal_loss)
    for key in output:
        print(key, output[key]['auprc'])
    print('\n')

    # Undersample Tomek Links (Focal Loss)
    output = grid_search(config, ['0'], undersample_tomek, get_binary_focal_loss)
    print(f'tomek (focal): {output['0']['auprc']}')
    print('\n')

    # Undersample KNN (Focal Loss)
    output = grid_search(config, [k for k in range(50, 250, 50)], undersample_knn, get_binary_focal_loss)
    for key in output:
        print(f'{key}: {output[key]['auprc']}')
    print('\n')


    

