import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from dowhy.gcm.fcms import PredictionModel
from sklearn.linear_model import LinearRegression
import numpy as np
import constants

class MyMLPModel(nn.Module, PredictionModel):
    """Defines my own polynomial model

    Args:
        PredictionModel (_type_): _description_
    """

    def __init__(
        self,
        num_parents,
        num_weak_parents=0,
        node: str = None,
        num_hidden: int = 4,
        num_layers: int = 1,
        act_fn=nn.ELU,
        use_layer_norm: bool = False,
    ) -> None:
        super(MyMLPModel, self).__init__()
        self.num_parents = num_parents
        self.num_weak_parents = num_weak_parents
        self.node = node
        self.num_hidden = num_hidden

        self.input_layer = nn.Linear(num_parents, num_hidden, bias=False)

        self.hidden_layers = nn.ModuleList()
        for layer in range(num_layers):
            self.hidden_layers.add_module(
                f"fc{layer}", nn.Linear(num_hidden, num_hidden, bias=False)
            )
            if use_layer_norm:
                self.hidden_layers.add_module(f"ln{layer}", nn.LayerNorm(num_hidden))
            self.hidden_layers.add_module(f"act{layer}", act_fn())
        self.output_layer = nn.Linear(num_hidden, 1, bias=False)
        self.model = nn.Sequential(
            self.input_layer, *self.hidden_layers, self.output_layer
        )
        self.grad_enabled = True

    def init_parameters(self, weak=False):
        # Random initialization
        a = constants.NOISE_DIST["mlp_model"]["a"]
        b = constants.NOISE_DIST["mlp_model"]["b"]
        b = a + (b - a) * constants.WEAK_FACTOR if weak else b
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=a, b=b)

    def init_weak_parameters(self):
        """Adjust the weights coming from the weak paths

        Args:
            variance (int, optional): _description_. Defaults to 1.
        """
        a = constants.NOISE_DIST["mlp_model"]["a"]
        b = constants.NOISE_DIST["mlp_model"]["b"]
        b = a + (b - a) * constants.WEAK_FACTOR

        assert self.num_weak_parents > 0, "No weak parents to initialize"
        # previous weights d x parents
        old_wts = self.input_layer.weight.to("cpu").detach()

        # weak wts with low variance
        weak_wts = torch.tensor(
            np.random.uniform(a, b, size=(self.num_weak_parents, self.num_hidden)),
            dtype=torch.float32,
        ).T

        # cat weak and old weights d x (parents + num_weak)
        self.input_layer = nn.Linear(self.num_parents, self.num_hidden, bias=False)
        self.input_layer.weight = nn.Parameter(torch.cat([weak_wts, old_wts], dim=1))

        # Get the new sequence of layers
        self.model = nn.Sequential(
            self.input_layer, *self.hidden_layers, self.output_layer
        )
        self.model.to(constants.DEVICE)

    def set_grads_enabled(self, flag: bool):
        """Switches off the gradients for the model if flag is set to False"""
        self.grad_enabled = flag
        for param in self.model.parameters():
            param.requires_grad = flag

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fits a Quadratic model

        Args:
            X (np.ndarray): _description_
            Y (np.ndarray): _description_
        """
        self.model, trn_hist = fit_torch_model(self.model, epochs=500, X=X, Y=Y)
        if self.node is not None:
            constants.logger.info(f"Trn Hist for node {self.node}\t {trn_hist}")

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=constants.DEVICE)
        if X.device != constants.DEVICE:
            X = X.to(constants.DEVICE)
        return self.model(X)

    @torch.no_grad()
    def predict(self, X: np.array) -> np.ndarray:
        """Predicts the output

        Args:
            X (np.array): _description_

        Returns:
            np.ndarray: _description_
        """
        self.model.eval()
        return self.forward(X).cpu().numpy()

    def clone(self):
        clone_mdl = MyMLPModel(self.num_parents)
        clone_mdl.model.load_state_dict(self.model.state_dict())
        return clone_mdl


class MyNonInvMLPModel(MyMLPModel):
    def __init__(
        self,
        num_parents,
        num_weak_parents=0,
        node: str = None,
        num_hidden: int = 4,
        num_layers: int = 1,
        act_fn=nn.ELU,
        use_layer_norm: bool = False,
        exog_dist_fn=None,
    ) -> None:
        super().__init__(
            num_parents + 1,
            num_weak_parents,
            node,
            num_hidden,
            num_layers,
            act_fn,
            use_layer_norm,
        )
        self.invertible = False
        self.exog_dist_fn = exog_dist_fn

    # Override every method in the parent class that uses invertible
    def init_parameters(self, weak=False):
        super().init_parameters(weak=weak)

    def init_weak_parameters(self):
        super().init_weak_parameters()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().fit(X, Y)

    def forward(self, X):
        if X.shape[1] == self.num_parents - 1:
            noise = torch.tensor(
                self.exog_dist_fn(), dtype=torch.float32, device=constants.DEVICE
            ).view(1, 1)
            X = torch.cat([X, noise], dim=1)
        return super().forward(X)

    def predict(self, X: np.array) -> np.ndarray:
        noise = self.exog_dist_fn(X.shape[0]).reshape(-1, 1)
        X = np.hstack([X, noise])
        return super().predict(X)

    def clone(self):
        colne_model = MyNonInvMLPModel(num_parents=self.num_parents - 1)
        colne_model.invertible = self.invertible
        colne_model.exog_dist_fn = self.exog_dist_fn
        colne_model.model.load_state_dict(self.model.state_dict())
        return colne_model


def fit_torch_model(model: MyMLPModel, epochs: int, X, Y) -> Tuple[nn.Module, dict]:
    """Fits a PyTorch model on the given data

    Args:
        model (nn.Module): _description_
        epochs (int): _description_
        X (_type_): _description_
        Y (_type_): _description_
    """
    X = torch.tensor(X, dtype=torch.float32, device=constants.DEVICE)
    Y = torch.tensor(Y, dtype=torch.float32, device=constants.DEVICE)

    # Create DataLoader
    ds = TensorDataset(X, Y)
    trn_ds, val_ds = torch.utils.data.random_split(
        ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))]
    )  # 80-20 split
    trn_dl = DataLoader(trn_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_state_dict = model.state_dict()
    patience = 20
    for epoch in range(epochs):
        model.train()
        for x, y in trn_dl:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Implement early stopping based on validation loss
        model.eval()
        with torch.no_grad():
            for x, y in val_dl:
                y_hat = model(x)
                val_loss = criterion(y_hat, y).item()
                break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            patience = 20
        else:
            patience -= 1
            if patience == 0:
                break
    model.load_state_dict(best_state_dict)
    return model, {"val_loss": best_val_loss, "epoch": epoch}


class MyLinearRegressionModel:
    def __init__(self, num_parents, num_weak_parents=0, node: str = None):
        self.num_parents = num_parents
        self.num_weak_parents = num_weak_parents
        self.node = node

        self.model = LinearRegression()
        self.model.intercept_ = 0

    def init_parameters(self, weak=False):
        a = constants.NOISE_DIST["lin_model"]["a"]
        b = constants.NOISE_DIST["lin_model"]["b"]
        if weak:
            a = 0
            b = constants.WEAK_FACTOR
        self.model.coef_ = np.random.uniform(a, b, size=self.num_parents)

    def init_weak_parameters(self):
        assert self.num_weak_parents > 0, "No weak parents to initialize"
        old_coef = self.model.coef_

        a = 0
        b = constants.WEAK_FACTOR

        weak_coef = np.random.uniform(a, b, size=self.num_weak_parents)
        self.model = LinearRegression()
        self.model.coef_ = np.concatenate([weak_coef, old_coef])
        self.model.intercept_ = 0

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def _parameters(self):
        return self.model.coef_

    def clone(self):
        clone_mdl = MyLinearRegressionModel(self.num_parents, self.num_weak_parents)
        clone_mdl.model.coef_ = self.model.coef_.copy()
        return clone_mdl
