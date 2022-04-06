import numpy as np

from nn_lib.mdl.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class CrossEntropyLoss(Loss):
    """
    cross entropy loss
    """
    # In order to avoid over- or underflow we clip prediction logits into [-MAX_LOG, MAX_LOG]
    MAX_LOG = 50

    def _clip(self, value: Tensor) ->Tensor:
        return F.clip(value, Tensor(-self.MAX_LOG, True), Tensor(self.MAX_LOG, True))

    def forward(self, prediction_logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute a loss Tensor based on logit predictions and ground truth labels
        """
        y = self._clip(prediction_logits)

        losses = F.log(F.reduce(F.exp(y), axis=1, reduction='sum', keepdims=True)) - F.reduce(y * target, axis=1, reduction='sum', keepdims=True)
        return F.reduce(losses) if self.reduce else losses