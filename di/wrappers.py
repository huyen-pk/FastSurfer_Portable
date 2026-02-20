from abc import ABC, abstractmethod
from typing import Any
from FastSurferCNN.train import Trainer as FastSurferCNNTrainerBase
from fl.api import FederatedClientAPI

class TrainerBase(ABC):
    @abstractmethod
    def train(self, *args, **kwargs) -> tuple[float, dict[str, float], int, dict[str, Any]]:
        """
        Abstract method to perform training using the provided trainer.

        Parameters
        ----------
        trainer : FastSurferTrainer
            The trainer object containing the model and training configuration.
        *args : tuple
            Optional positional arguments for training.
        **kwargs : dict
            Optional keyword arguments for training.

        Returns
        -------
        float
            The best loss achieved during training.
        metrics: dict[str, float]
            The best metric achieved during training.
        num_examples: int
            The number of examples used during training.
        """
        pass

    @abstractmethod
    def evaluate(self, epoch: int = 0, *args, **kwargs) -> tuple[float, dict[str, float], int]:
        """
        Optional method to perform evaluation using the provided trainer.

        Parameters
        ----------
        epoch : int, optional
            The current epoch number, by default 0.
        *args : tuple
            Optional positional arguments for evaluation.
        **kwargs : dict
            Optional keyword arguments for evaluation.

        Returns
        -------
        float
            The loss achieved during evaluation.
        metrics: dict[str, float]
            The metric achieved during evaluation.
        num_examples: int
            The number of examples evaluated.
        """
        pass
    def clone(self) -> "TrainerBase":
        """
        Create a deep copy of the trainer instance.

        Returns
        -------
        TrainerBase
            A new instance of the trainer with the same configuration.
        """
        import copy
        return self.__class__(**copy.deepcopy(self.__dict__))
