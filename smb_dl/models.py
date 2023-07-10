import numpy as np
import torch
from typing import List
import config as local_cfg


class NLL_LOSS(torch.nn.Module):
    """
        Negative Log Likelihood loss, with the option of fixing sigma for a certain number of epochs.
        The network predicts log(sigma) as suggested in Kendall et al., 2017.
    """

    def __init__(self, ct_sigma=None, max_epochs_ct_sigma=None):
        if max_epochs_ct_sigma is not None:
            assert ct_sigma is not None
        self.ct_log_sigma_sq = torch.tensor(np.log(ct_sigma ** 2), requires_grad=False).to(local_cfg.DEVICE)
        self.max_epochs_ct_sigma = max_epochs_ct_sigma
        super().__init__()

    def forward(self, preds, target, crt_epoch):
        """Compute Loss.

        Args:
          preds: batch_size x 2, consisting of mu and log sigma
          target: batch_size x 1, regression targets
          crt_epoch: the index of the current training epoch
        """
        mu, log_sigma_sq = preds[:, 0].unsqueeze(-1), preds[:, 1].unsqueeze(-1)
        if self.max_epochs_ct_sigma is not None and crt_epoch <= self.max_epochs_ct_sigma:
            loss = 0.5 * self.ct_log_sigma_sq + 0.5 * torch.exp(-self.ct_log_sigma_sq) * (target - mu) ** 2
        else:
            loss = 0.5 * log_sigma_sq + 0.5 * torch.exp(-log_sigma_sq) * (target - mu) ** 2
        loss = torch.mean(loss, dim=0)
        return loss


class MLP(torch.nn.Module):
    """Multi-layer perceptron for regression predictions."""

    def __init__(self, dropout_p: float = 0.0, n_inputs: int = 1, n_hidden: List[int] = [100], n_outputs: int = 1,
                 predict_sigma: bool = False, predict_ct_sigma: bool = False) -> None:
        """Initialize a new instance of MLP.

        Args:
          dropout_p: dropout probability
          n_inputs: size of input dimension
          n_hidden: list of hidden layer sizes
          n_outputs: number of model outputs
          predict_sigma: whether the model intends to predict sigma term when minimizing NLL
          predict_ct_sigma: whether to assume a constant sigma (~homoscedasticity)
        """
        super().__init__()
        layers = []
        layer_sizes = [n_inputs] + n_hidden
        for idx in range(1, len(layer_sizes)):
            layers += [
                torch.nn.Linear(layer_sizes[idx - 1], layer_sizes[idx]),
                torch.nn.Tanh(),
                torch.nn.Dropout(dropout_p) if idx != 1 else torch.nn.Identity(),
            ]
        layers += [torch.nn.Linear(layer_sizes[-1], n_outputs)]
        self.net = torch.nn.Sequential(*layers)
        self.predict_sigma = predict_sigma
        self.predict_ct_sigma = predict_ct_sigma

    def forward(self, x) -> torch.Tensor:
        out = self.net(x)
        if self.predict_ct_sigma:
            out[:, 1] = self.net[-1].bias[1].repeat(len(x))
        return out
