from abc import (
    ABC,
    abstractmethod
)

from collections import defaultdict
from collections.abc import Generator, Sequence, Iterable, Callable
import torch
import torch.nn.functional as F
import scipy
from typing import Optional
import math
import tqdm

def get_shuffled_indices(
    dataset_size: int,
    device="cpu",
    ensemble_shape=(),
) -> torch.Tensor:
    """
    Get a tensor of a batch of shuffles of indices `0,...,dataset_size - 1`.

    Parameters
    ----------
    dataset_size : int
        The size of the dataset the indices of which to shuffle
    device : int | str | torch.device, optional
        The device to store the resulting tensor on. Default: "cpu"
    ensemble_shape : tuple[int], optional
        The batch shape of the shuffled index tensors. Default: ()
    """
    total_shape = ensemble_shape + (dataset_size,)
    uniform = torch.rand(
        total_shape,
        device=device
    )
    indices = uniform.argsort(dim=-1)

    return indices

def get_random_reshuffler(
    dataset_size: int,
    minibatch_size: int,
    device="cpu",
    ensemble_shape=()
) -> Generator[torch.Tensor]:
    """
    Generate minibatch indices for a random shuffling dataloader.
    Supports arbitrary ensemble shapes.

    Parameters
    ----------
    dataset_size : int
        The size of the dataset to yield batches of minibatch indices for.
    minibatch_size : int
        The minibatch size.
    device : int | str | torch.device, optional
        The device to store the index tensors on. Default: "cpu"
    ensemble_shape : tuple[int], optional
        The ensemble shape of the minibatch indices. Default: ()
    """
    q, r = divmod(dataset_size, minibatch_size)
    minibatch_num = q + min(1, r)
    minibatch_index = minibatch_num
    while True:
        if minibatch_index == minibatch_num:
            minibatch_index = 0
            shuffled_indices = get_shuffled_indices(
                dataset_size,
                device=device,
                ensemble_shape=ensemble_shape
            )

        yield shuffled_indices[
            ...,
            minibatch_index * minibatch_size
        :(minibatch_index + 1) * minibatch_size
        ]

        minibatch_index += 1

def get_dataloader_random_reshuffle(
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a feature and a label tensor,
    creates a random reshuffling (without replacement) dataloader
    that yields pairs `minibatch_features, minibatch_labels` indefinitely.
    Support arbitrary ensemble shapes.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        ensemble_shape : tuple[int]
            The required ensemble shapes of the outputs.
        minibatch_size : int
            The size of the minibatches.
    features : torch.Tensor
        Tensor of dataset features.
        We assume that the first dimension is the batch dimension
    labels : torch.Tensor
        Tensor of dataset labels.

    Returns
    -------
    A generator of tuples `minibatch_features, minibatch_labels`.
    """
    for indices in get_random_reshuffler(
        len(labels),
        config["minibatch_size"],
        ensemble_shape=config["ensemble_shape"]
    ):
        yield features[indices], labels[indices]

class Linear(torch.nn.Module):
    """
    Ensemble-ready affine transformation `y = x^T W + b`.

    Arguments
    ---------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
        `"float_dtype"` : `torch.dtype`
            The floating point datatype to use for the parameters.
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    bias : `bool`, optional
        Whether the model should include bias. Default: `True`.
    init_multiplier : `float`, optional
        The weight parameter values are initialized following
        a normal distribution with center 0 and std
        `in_features ** -.5` times this value. Default: `1.`

    Calling
    -------
    Instance calls require one positional argument:
    features : `torch.Tensor`
        The input tensor. It is required to be one of the following shapes:
        1. `ensemble_shape + batch_shape + (in_features,)`
        2. `batch_shape + (in_features,)

        Upon a call, the model thinks we're in the first case
        if the first `len(ensemble_shape)` many entries of the
        shape of the input tensor is `ensemble_shape`.
    """
    def __init__(
        self,
        config: dict,
        in_features: int,
        out_features: int,
        bias=True,
        init_multiplier=1.,
    ):
        super().__init__()

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(
                config["ensemble_shape"] + (out_features,),
                device=config["device"],
                dtype=config["float_dtype"]
            ))
        else:
            self.bias = None

        self.weight = torch.nn.Parameter(torch.empty(
            config["ensemble_shape"] + (in_features, out_features),
            device=config["device"],
            dtype=config["float_dtype"]
        ).normal_(std=out_features ** -.5) * init_multiplier)


    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        ensemble_shape = self.weight.shape[:-2]
        ensemble_dim = len(ensemble_shape)
        ensemble_input = features.shape[:ensemble_dim] == ensemble_shape
        batch_dim = len(features.shape) - 1 - ensemble_dim * ensemble_input
        
        # (*e, *b, i) @ (*e, *b[:-1], i, o)
        weight = self.weight.reshape(
            ensemble_shape
          + (1,) * (batch_dim - 1)
          + self.weight.shape[-2:]
        )
        features = features @ weight

        if self.bias is None:
            return features
        
        # (*e, *b, o) + (*e, *b, o)
        bias = self.bias.reshape(
            ensemble_shape
          + (1,) * batch_dim
          + self.bias.shape[-1:]
        )
        features = features + bias

        return features
    
def get_mlp(
    config: dict,
    in_features: int,
    out_features: int,
    hidden_layer_num: Optional[int] = None,
    hidden_layer_size: Optional[int] = None,
    hidden_layer_sizes: Optional[Iterable[int]] = None,
) -> torch.nn.Sequential:
    """
    Creates an MLP with ReLU activation functions.
    Can create a model ensemble.

    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
        `"float_dtype"` : `torch.dtype`
            The floating point datatype to use for the parameters.
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    hidden_layer_num : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_size : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_sizes: `Iterable[int]`, optional
        If given, each entry gives a hidden layer with the given size.
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = (hidden_layer_size,) * hidden_layer_num

    layers = []
    layer_in_size = in_features
    for layer_out_size in hidden_layer_sizes:
        layers.extend([
            Linear(
                config,
                layer_in_size,
                layer_out_size,
                init_multiplier=2 ** .5
            ),
            torch.nn.ReLU()
        ])
        layer_in_size = layer_out_size
    
    layers.append(Linear(
        config,
        layer_in_size,
        out_features,
    ))

    return torch.nn.Sequential(*layers)

def evaluate_model(
    config: dict,
    features: torch.Tensor,
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    values: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate a model on a supervised dataset.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pair:
        `"minibatch_size_eval"` : `int`
            Size of consecutive minibatches to take from the dataset.
            To be set according to RAM or GPU memory capacity.
    features : `torch.Tensor`
        Feature tensor.
    get_metric : `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
        Function to get the metric from a pair of
        predicted and target value tensors.
    model : `torch.nn.Module`
        The model to evaluate.
    values : `torch.Tensor`
        Target value tensor.
    """
    dataset_size = len(features)
    minibatch_num = math.ceil(dataset_size / config["minibatch_size_eval"])
    metric = 0
    with torch.no_grad():
        for i in range(minibatch_num):
            minibatch_features, minibatch_values = (
                t[
                    i * config["minibatch_size_eval"]
                   :(i + 1) * config["minibatch_size_eval"]
                ]
                for t in (features, values)
            )
            minibatch_predict = model(minibatch_features)
            minibatch_metric = get_metric(
                config,
                minibatch_predict,
                minibatch_values
            )
            minibatch_size = len(minibatch_features)

            metric += minibatch_metric * minibatch_size

    return metric / dataset_size

def normalize_features(
    train_features: torch.Tensor,
    additional_features=(),
    verbose=False
):
    """
    Normalize feature tensors by
    1. subtracting the total mean of the training features, then
    2. dividing by the total std of the offset training features.

    Optionally, apply the same transformation to additional feature tensors,
    eg. validation and test feature tensors.

    Parameters
    ----------
    train_features : `torch.Tensor`
        Training feature tensor.
    additional_features : `Iterable[torch.Tensor]`, optional
        Iterable of additional features to apply the transformation to.
        Default: `()`.
    verbose : `bool`, optional
        Whether to print the total mean and std
        gotten for the transformation.
    """
    sample_mean = train_features.mean(dim=0)
    train_features -= sample_mean
    for features in additional_features:
        features -= sample_mean

    sample_std = train_features.std(dim=0)
    train_features /= sample_std
    for features in additional_features:
        features /= sample_std

    if verbose:
        print(
            "Training feature tensor statistics before normalization:",
            f"mean {sample_mean.cpu().item():.4f}",
            f"std {sample_std.cpu().item():.4f}",
            flush=True
        )

def welch_one_sided(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence_level=.95
) -> torch.Tensor:
    """
    Performs Welch's t-test with null hypothesis: the expected value
    of the random variable the target tensor collects samples of
    is larger then the expected value
    of the random variable the source tensor collects samples of.

    In the tensors, dimensions after the first 
    are considered batch dimensions.

    Parameters
    ----------
    source : `torch.Tensor`
        Source sample, of shape `(sample_size,) + batch_shape`.
    target : `torch.Tensor`
        Target sample, of shape `(sample_size,) + batch_shape`.
    confidence_level : `float`, optional
        Confidence level of the test. Default: `.95`.
    Returns
    -------
    A Boolean tensor of shape `batch_shape` that is `False`
    where the null hypothesis is rejected.
    """
    sample_num = len(source)
    source_sample_mean, target_sample_mean = (
        t.mean(dim=0)
        for t in (source, target)
    )
    source_sample_var, target_sample_var = (
        t.var(dim=0)
        for t in (source, target)
    )
    var_sum = source_sample_var + target_sample_var

    t = (
        (target_sample_mean - source_sample_mean)
      * (sample_num / var_sum).sqrt()
    )

    nu = (
        var_sum.square()
      * (sample_num - 1)
      / (source_sample_var ** 2 + target_sample_var ** 2)
    )

    p = scipy.stats.t(
        nu.cpu().numpy()
    ).cdf(
        t.cpu().numpy()
    )

    return torch.asarray(
        p > confidence_level,
        device=source.device
    )

def get_seed(
    upper=1 << 31
) -> int:
    """
    Generates a random integer by the `torch` PRNG,
    to be used as seed in a stochastic function.

    Parameters
    ----------
    upper : int, optional
        Exclusive upper bound of the interval to generate integers from.
        Default: 1 << 31.

    Returns
    -------
    A random integer.
    """
    return int(torch.randint(upper, size=()))

class Optimizer(ABC):
    """
    Optimizer base class.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"learning_rate"`
        `"weight_decay"`

    Instance attributes
    -------------------
    config : `dict`
        The hyperparameter dictionary.
    parameters : `list[torch.nn.Parameter]`
        The list of tracked parameters.
    step_id : `int`
        Train step counter.
    """
    keys=(
        "learning_rate",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        self.config = dict()
        self.parameters = list(parameters)
        self.step_id = 0

        if config is not None:
            self.update_config(config)


    def get_hyperparameter(
        self,
        key: str,
        parameter: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the hyperparameter with name `key`,
        transform it to `torch.Tensor` with the same
        `device` and `dtype` as `parameter`
        and reshape it to be broadcastable
        to `parameter` by postfixing to its shape
        an appropriate number of dimensions of 1.
        """        
        hyperparameter = torch.asarray(
            self.config[key],
            device=parameter.device,
            dtype=parameter.dtype
        )

        return hyperparameter.reshape(
            hyperparameter.shape
            + (
                len(parameter.shape)
                - len(hyperparameter.shape)
            )
            * (1,)
        )


    def step(self):
        """
        Update optimizer state, then apply parameter updates in-place.
        Assumes that backpropagation has already occurred by
        a call to the `backward` method of the loss tensor.
        """
        self.step_id += 1
        with torch.no_grad():
            for i, parameter in enumerate(self.parameters):
                self._update_parameter(parameter, i)


    def update_config(self, config: dict):
        """
        Update hyperparameters by the values in `config: dict`.
        """
        for key in self.keys:
            self.config[key] = config[key]


    def zero_grad(self):
        """
        Make the `grad` attribute of each tracked parameter `None`.
        """
        for parameter in self.parameters:
            parameter.grad = None


    def _apply_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_update: torch.Tensor
    ):
        parameter += parameter_update


    @abstractmethod
    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        if self.config["weight_decay"] is None:
            return torch.zeros_like(parameter)
        
        return -(
            self.get_hyperparameter("learning_rate", parameter)
          * self.get_hyperparameter("weight_decay", parameter)
          * parameter
        )


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        pass


    def _update_parameter(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        self._update_state(parameter, parameter_id)
        parameter_update = self._get_parameter_update(
            parameter,
            parameter_id
        )
        self._apply_parameter_update(
            parameter,
            parameter_update
        )

class AdamW(Optimizer):
    """
    Adam optimizer with optionally weight decay.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"epsilon"`,
        `"first_moment_decay"`,
        `"learning_rate"`
        `"second_moment_decay"`,
        `"weight_decay"`
    """
    keys = (
        "epsilon",
        "first_moment_decay",
        "learning_rate",
        "second_moment_decay",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        super().__init__(parameters, config)
        self.first_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]
        self.second_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]


    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        parameter_update = super()._get_parameter_update(
            parameter,
            parameter_id
        )

        epsilon = self.get_hyperparameter(
            "epsilon",
            parameter
        )
        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        learning_rate = self.get_hyperparameter(
            "learning_rate",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment_debiased = (
            first_moment
          / (1 - first_moment_decay ** self.step_id)
        )
        second_moment_debiased = (
            second_moment
          / (1 - second_moment_decay ** self.step_id)
        )        

        parameter_update -= (
            learning_rate
          * first_moment_debiased
          / (
                second_moment_debiased.sqrt()
              + epsilon
            )
        )

        return parameter_update


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        if parameter.grad is None:
            return

        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment[:] = (
            first_moment_decay
          * first_moment
          + (1 - first_moment_decay)
          * parameter.grad
        )
        second_moment[:] = (
            second_moment_decay
          * second_moment
          + (1 - second_moment_decay)
          * parameter.grad.square()
        )

def pbt(
    config: dict,
    get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_features: torch.Tensor,
    train_values: torch.Tensor,
    valid_features: torch.Tensor,
    valid_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Population-based training on a supervised learning task.
    Tuned hyperparameters are given by raw values and transformations.
    This way, the hyperparameters are perturbed by
    additive noise on raw values.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"ensemble_shape"` : tuple[int]
            Ensemble shape. We assume this is a 1-dimensional tuple
            with dimensions the population size.
        `"hyperparameter_raw_init_distributions"` : `dict`
            Dictionary that maps tuned hyperparameter names
            to `torch.distributions.Distribution` of raw hyperparameter values.
            Required keys:
            `"learning_rate"`:
                The learning rate of stochastic gradient descent.
        `"hyperparameter_raw_perturbs"` : `dict`
            Dictionary that maps tuned hyperparameter names
            to `torch.distributions.Distribution` of additive noise.
        `"hyperparameter_transforms"` : `dict`
            Dictionary that maps tuned hyperparameter names
            to transformations of raw hyperparameter values.
        `"improvement_threshold"` : `float`
            A new metric score has to be this much better
            than the previous best to count as an improvement.
        `"minibatch_size"` : `int`
            Minibatch size to use in a training step.
        `"minibatch_size_eval"` : `int`
            Minibatch size to use in evaluation.
            On CPU, should be about the same as `minibatch_size`.
            On GPU, should be as big as possible without
            incurring an Out of Memory error.
        `"pbt"` : `bool`
            Whether to use PBT updates in validations.
            If `False`, the algorithm just samples hyperparameters at start,
            then keeps them constant.
        `"steps_num"` : `int`
            Maximum number of training steps.
        `"steps_without_improvement`" : `int`
            If the number of training steps without improvement
            exceeds this value, then training is stopped.
        `"valid_interval"` : `int`
            Frequency of evaluations, measured in number of training steps.
        `"welch_confidence_level"` : `float`
            The confidence level in Welch's t-test
            that is used in determining if a population member
            is to be replaced by another member with perturbed hyperparameters.
        `"welch_sample_size"` : `int`
            The last this many validation metrics are used
            in Welch's t-test.
    `get_loss` : `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
        A function that maps a pair of predicted and target value tensors
        to a tensor of losses per ensemble member.
    `get_metric` : `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
        A function that maps a pair of predicted and target value tensors
        to a tensor of metrics per ensemble member.
        We assume a greater metric is better.
    `model` : `torch.nn.Module`
        The model ensemble to tune.
    `optimizer` : `Optimizer`
        An optimizer that tracks the parameters of `model`.
    `train_features` : `torch.Tensor`
        Training feature tensor.
    `train_values` : `torch.Tensor`
        Training value tensor.
    `valid_features` : `torch.Tensor`
        Validation feature tensor.
    `valid_values` : `torch.Tensor`
        Validation value tensor.

    Returns
    -------
    An output dictionary with the following key-value pairs:
        `"source mask"` : `torch.Tensor`
            The source masks of population members
            that were replace by other members in a PBT update
        `"target indices"` : `torch.Tensor`
            The indices of population members
            that the member where the source mask is to were replaced with.
        `"training loss"` : `torch.Tensor`
            The training losses at evaluation steps.
        `"training metric"` : `torch.Tensor`
            The training metrics at evaluation steps.
        `"validation loss"` : `torch.Tensor`
            The validation losses at evaluation steps.
        `"validation metric"` : `torch.Tensor`
            The validation metrics at evaluation steps.

        In addition, for each tuned hyperparameter name,
        we include a `torch.Tensor` of values per update.
    """
    ensemble_shape = config["ensemble_shape"]
    if len(ensemble_shape) != 1:
        raise ValueError(f"The number of dimensions in the ensemble shape should be 1 for the  population size, but it is {len(ensemble_shape)}")

    population_size = ensemble_shape[0]
    config_local = dict(config)
    output = defaultdict(list)

    for name, distribution in config[
        "hyperparameter_raw_init_distributions"
    ].items():
        value_raw = distribution.sample(ensemble_shape)
        config_local[name + "_raw"] = value_raw
        value = config[
            "hyperparameter_transforms"
        ][name](value_raw)
        config_local[name] = value
        output[name].append(value)

    optimizer.update_config(config_local)

    best_valid_metric = -torch.inf
    progress_bar = tqdm.trange(config["steps_num"])
    steps_without_improvement = 0
    train_dataloader = get_dataloader_random_reshuffle(
        config,
        train_features,
        train_values
    )

    for step_id in progress_bar:
        minibatch_features, minibatch_values = next(train_dataloader)
        optimizer.zero_grad()

        predict = model(minibatch_features)
        loss = get_loss(config, predict, minibatch_values).sum()
        loss.backward()
        optimizer.step()

        if step_id % config["valid_interval"] == 0:
            with torch.no_grad():
                for features, values, split_name in (
                    (train_features, train_values, "training"),
                    (valid_features, valid_values, "validation")
                ):
                    loss, metric = (
                        evaluate_model(
                            config,
                            features,
                            f,
                            model,
                            values
                        )
                        for f in (get_loss, get_metric)
                    )
                    output[f"{split_name} loss"].append(loss)
                    output[f"{split_name} metric"].append(metric)

                best_last_metric = output["validation metric"][-1].max()
                if (
                    best_valid_metric + config["improvement_threshold"]
                ) < best_last_metric:
                    best_valid_metric = best_last_metric
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += config["valid_interval"]
                    if steps_without_improvement > config[
                        "steps_without_improvement"
                    ]:
                        break

                if config["pbt"] and (len(output["validation metric"]) >= config[
                    "welch_sample_size"
                ]):
                    validation_metrics = torch.stack(
                        output["validation metric"][
                            -config["welch_sample_size"]:
                        ]
                    )
                    target_indices = torch.randint(
                        device=validation_metrics.device,
                        high=population_size,
                        size=(population_size,)
                    )
                    source_mask = welch_one_sided(
                        validation_metrics,
                        validation_metrics[:, target_indices],
                        confidence_level=config["welch_confidence_level"]
                    )
                    output["source mask"].append(source_mask)
                    output["target indices"].append(target_indices)

                    if source_mask.any():
                        for parameter in model.parameters():
                            parameter[source_mask] = parameter[
                                target_indices[source_mask]
                            ]

                        for name, transform in config[
                            "hyperparameter_transforms"
                        ].items():
                            value_raw: torch.Tensor = config_local[
                                name + "_raw"
                            ]

                            additive_noise = config[
                                "hyperparameter_raw_perturb"
                            ][name].sample(
                                (source_mask.sum(),)
                            )
                            perturbed_values = value_raw[
                                target_indices
                            ][source_mask] + additive_noise
                            value_raw[source_mask] = perturbed_values
                            value = transform(value_raw)
                            config_local[name] = value
                            output[name].append(value)

                        optimizer.update_config(config_local)


    progress_bar.close()
    for key, value in output.items():
        if isinstance(value, list):
            output[key] = torch.stack(value)

    return output

def get_binary_cross_entropy(
    config: dict,
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary cross-entropy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size,)`.
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size, 1)`.

    Returns
    -------
    The tensor of binary cross-entropies per ensemble member
    of shape `ensemble_shape`.
    """

    return F.binary_cross_entropy_with_logits(
        logits[..., 0],
        labels.broadcast_to(logits.shape[:-1]),
        reduction="none"
    ).mean(dim=-1)

def get_binary_accuracy(
    config: dict,
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary accuracy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size, 1)`.
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of binary accuracies per ensemble member
    of shape `ensemble_shape`.
    """
    predict_positives = logits[..., 0] > 0
    true_positives = labels.broadcast_to(
        predict_positives.shape
    ).to(torch.bool)

    return (
        predict_positives == true_positives
    ).to(torch.float32).mean(dim=-1)