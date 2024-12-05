import torch
from sklearn.neighbors import NearestNeighbors

def random_undersample(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform random undersampling of a dataset.

    This function reduces the size of the dataset by randomly selecting
    a subset of the features and corresponding labels.

    Parameters
    ----------
    features : torch.Tensor
        The feature tensor to be undersampled. Expected to have shape 
        `(dataset_size, feature_dim)`.
    labels : torch.Tensor
        The label tensor corresponding to the features. Expected to have 
        shape `(dataset_size,)`.
    size : int
        The desired size of the undersampled dataset.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - features (torch.Tensor): Undersampled feature tensor of shape `(size, feature_dim)`.
        - labels (torch.Tensor): Corresponding undersampled label tensor of shape `(size,)`.
    """

    indices = torch.randperm(features.shape[0])
    features = features[indices]
    labels = labels[indices]

    features = features[:size]
    labels = labels[:size]

    return features, labels

def smote(
    config: dict, 
    features: torch.Tensor, 
    labels: torch.Tensor, 
    minority_class: int, 
    N: int, 
    k: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Synthetic Minority Over-sampling Technique (SMOTE).

    This function generates synthetic samples for the minority class by
    interpolating between existing samples and their nearest neighbors. 

    Parameters
    ----------
    config : dict
        A configuration dictionary containing:
        - 'device' (torch.device or str): Device to which tensors should be moved.
        - 'features_dtype' (torch.dtype): Data type for the feature tensors.
    features : torch.Tensor
        The feature tensor for the minority class. Expected to have shape 
        `(n_samples, feature_dim)`.
    labels : torch.Tensor
        The label tensor for the minority class. Expected to have shape `(n_samples,)`.
    minority_class : int
        The label value representing the minority class.
    N : int
        The oversampling multiplier. For example, `N=2` means generating 
        synthetic samples equal in number to the original samples.
    k : int, optional
        The number of nearest neighbors to consider for interpolation (default is 5).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - features (torch.Tensor): The combined feature tensor after SMOTE, 
          including the original and synthetic samples.
        - labels (torch.Tensor): The combined label tensor after SMOTE.
    """

    features_np = features.detach().cpu().numpy()
    n_samples = features_np.shape[0]
    n_generate = n_samples * (N-1)

    nbrs = NearestNeighbors(n_neighbors=k).fit(features_np)
    _, neighbors = nbrs.kneighbors(features_np)

    sample_indices = torch.randint(0, n_samples, (n_generate,))
    neighbor_indices = torch.randint(1, k, (n_generate,))

    base_features = torch.tensor(features_np[sample_indices.numpy()], device=config['device'], dtype=config['features_dtype'])
    neighbor_features = torch.tensor(features_np[neighbors[sample_indices.numpy(), neighbor_indices.numpy()]], device=config['device'], dtype=config['features_dtype'])

    lambdas = torch.rand(n_generate, 1, device=config['device'], dtype=config['features_dtype'])
    synthetic_features = base_features + lambdas * (neighbor_features - base_features)

    features = torch.cat([features, synthetic_features], dim=0)
    labels = torch.cat([labels, torch.full((n_generate,), minority_class, device=config['device'], dtype=config['features_dtype'])], dim=0)

    return features, labels

def knn_undersampling(
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor,
    minority_class: int,
    k: int,
    t: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform K-Nearest Neighbors (KNN) undersampling.

    This function removes majority class samples that have a high number of 
    minority class neighbors, reducing overlap and enhancing class separability.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing:
        - 'device' (torch.device or str): Device to which tensors should be moved.
    features : torch.Tensor
        The feature tensor for the dataset. Expected to have shape 
        `(n_samples, feature_dim)`.
    labels : torch.Tensor
        The label tensor for the dataset. Expected to have shape `(n_samples,)`.
    minority_class : int
        The label value representing the minority class.
    k : int
        The number of nearest neighbors to consider for each sample.
    t : int, optional
        The threshold for the number of minority class neighbors required to 
        remove a majority class sample (default is 1).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - features (torch.Tensor): The feature tensor after KNN undersampling.
        - labels (torch.Tensor): The label tensor after KNN undersampling.
    """

    features_np = features.detach().cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k).fit(features_np)
    _, indices = nbrs.kneighbors(features_np)

    neighbor_indices = torch.tensor(indices[:, 1:], device=config['device'], dtype=torch.int)
    neighbor_labels = labels[neighbor_indices]

    minority_count = (neighbor_labels == minority_class).sum(axis=-1)
    keep_mask = ~((minority_count >= t) & (labels != minority_class))

    features = features[keep_mask]
    labels = labels[keep_mask]

    return features, labels

def tomek_links(
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor,
    minority_class: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform undersampling using the Tomek links method.

    This function identifies and removes Tomek links, which are pairs of 
    samples from different classes that are each other's nearest neighbors. 

    Parameters
    ----------
    config : dict
        A configuration dictionary containing:
        - 'device' (torch.device or str): Device to which tensors should be moved.
        - 'features_dtype' (torch.dtype): Data type for tensor computations.
    features : torch.Tensor
        The feature tensor for the dataset. Expected to have shape 
        `(n_samples, feature_dim)`.
    labels : torch.Tensor
        The label tensor for the dataset. Expected to have shape `(n_samples,)`.
    minority_class : int
        The label value representing the minority class.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - features (torch.Tensor): The feature tensor after removing Tomek links.
        - labels (torch.Tensor): The label tensor after removing Tomek links.
    """

    features_np = features.detach().cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=2).fit(features_np)
    _, indices = nbrs.kneighbors(features_np)

    neighbor_indices = torch.tensor(indices[:, 1], device=config['device'], dtype=torch.int)
    neighbor_labels = labels[neighbor_indices]

    is_tomek = (labels != neighbor_labels) & (neighbor_indices[neighbor_indices] == torch.arange(neighbor_indices.shape[0], device=config['device'], dtype=config['features_dtype']))
    keep_mask = ~(is_tomek & (labels != minority_class))

    features = features[keep_mask]
    labels = labels[keep_mask]

    return features, labels