import sklearn.kernel_approximation as kernel_approx
from sklearn.ensemble import RandomForestRegressor
import scipy.linalg as linalg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


def solve_least_squares(X, y, ridge=0.0):
    if ridge <= 0:  # min norm solution
        estim_param, _resid, _rank, _s = linalg.lstsq(X, y)
    else:  # SVD implementation of ridge regression
        u, s, vh = linalg.svd(X, full_matrices=False, compute_uv=True)
        prod_aux = s / (ridge + s ** 2)  # If S = diag(s) => P = inv(S.T S + ridge * I) S.T => prod_aux = diag(P)
        estim_param = (prod_aux * (y @ u)) @ vh  # here estim_param = V P U.T
    return estim_param


def ensemble_solution_for_overparametrized(Xf, y, n_ensembles, ridge=0.0, seed=0):
    n, n_features = Xf.shape
    rng = np.random.RandomState(seed)
    estim_param = np.zeros((n_features,))
    for i in range(n_ensembles):
        idx = rng.choice(n_features, n, replace=False)
        Xs = Xf[:, idx]
        estim_param_sub = solve_least_squares(Xs, y, ridge)
        estim_param[idx] += 1 / n_ensembles * estim_param_sub
    return estim_param


class LinearInTheParameters(object):

    def __init__(self, n_features: int = 20, random_state: int = 0,
                 n_ensembles: int = 0, ridge: float = 0.0):
        self.n_features = n_features
        self.random_state = random_state
        self.estim_param = None
        self.n_ensembles = n_ensembles
        self.ridge = ridge

    def map_fit_transform(self, X):
        return NotImplementedError()

    def map_transform(self, X):
        return NotImplementedError()

    def fit(self, X, y):
        X = np.atleast_2d(X)
        Xf = self.map_fit_transform(X)
        n, n_features = Xf.shape
        if self.n_ensembles <= 1 or n >= n_features:
            self.estim_param = solve_least_squares(Xf, y, self.ridge)
        else:
            self.estim_param = ensemble_solution_for_overparametrized(Xf, y, self.n_ensembles, self.ridge, self.random_state)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        Xf = self.map_transform(X)
        return Xf @ self.estim_param

    @property
    def param_norm(self):
        return np.linalg.norm(self.estim_param)

    def __repr__(self):
        return NotImplementedError()


# --- Static models ---
class RBFSampler(LinearInTheParameters):
    def __init__(self, n_features: int = 20, gamma: float = 1.0, random_state: int = 0,
                 n_ensembles: int = 0, ridge: float = 0.0):
        self.gamma = gamma
        self.rbf_feature = None
        super(RBFSampler, self).__init__(n_features, random_state, n_ensembles, ridge)

    def map_fit_transform(self, X):
        self.rbf_feature = kernel_approx.RBFSampler(n_components=self.n_features, gamma=self.gamma,
                                                    random_state=self.random_state)
        return self.rbf_feature.fit_transform(X)

    def map_transform(self, X):
        return self.rbf_feature.transform(X)

    def __repr__(self):
        return '{}({},{},{},{},{})'.format(type(self).__name__, self.n_features, self.gamma, self.random_state,
                                              self.n_ensembles, self.ridge)


class RBFNet(LinearInTheParameters):
    def __init__(self, n_features: int = 20,  gamma: float = 1.0, spread: float = 1.0, random_state: int = 0,
                 n_ensembles: int = 0, ridge: float = 0.0):
        self.gamma = gamma
        self.spread = spread
        self.centers = None
        super(RBFNet, self).__init__(n_features, random_state, n_ensembles, ridge)

    def map_transform(self, X):
        n, d = X.shape
        aux = X.reshape((n, 1, d)) - self.centers.reshape((1, -1, d))
        features = np.exp(-self.gamma * (aux**2).sum(axis=-1))
        return features

    def map_fit_transform(self, X):
        X = np.atleast_2d(X)
        rng = np.random.RandomState(self.random_state)
        self.centers = self.spread * rng.randn(self.n_features, X.shape[1])
        return self.map_transform(X)

    def __repr__(self):
        return '{}({},{},{},{},{},{})'.format(type(self).__name__, self.n_features, self.gamma, self.spread,
                                                 self.random_state, self.n_ensembles, self.ridge)


class RandomForest(object):
    def __init__(self, n_features: int = 20,  bootstrap: bool = False, random_state: int = 0):
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.forest = None

    def fit(self, X, y):
        X = np.atleast_2d(X)
        num_samples, n_in = X.shape
        if self.n_features <= num_samples:
            max_nodes = self.n_features
            n_estimators = 1
        else:
            n_estimators = int(np.ceil(self.n_features / num_samples))
            max_nodes = num_samples

        self.forest = RandomForestRegressor(n_estimators=n_estimators, max_leaf_nodes=int(max_nodes),
                                            bootstrap=self.bootstrap, max_features='sqrt',
                                            random_state=self.random_state)
        self.forest.fit(X, y)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.forest.predict(X)

    def __repr__(self):
        return '{}({},{},{})'.format(type(self).__name__, self.n_features, self.bootstrap, self.random_state)


class FullyConnectedNet(object):
    def __init__(self, n_features: int = 20,  n_interm_layers: int = 1,
                 nonlinearity: str = 'relu', lr: float = 0.001, momentum: float = 0.0,
                 nesterov: bool = False, epochs: int = 10000, batch_size: int = 5000,
                 total_decay: float = 1000, grad_clipping: float = 1.0,
                 initialization: str = 'xavier', random_state: int = 0,
                 cpu_only: bool = False, verbose: bool = False):
        self.n_features = n_features
        self.n_interm_layers = n_interm_layers
        use_cuda = torch.cuda.is_available() and not cpu_only
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.nonlinearity = nonlinearity
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.total_decay = total_decay
        self.decay_rate = np.exp(-np.log(total_decay)/epochs) if total_decay > 0 else 1.0
        self.grad_clipping = grad_clipping
        self.initialization = initialization
        self.net = None

    def reuse_weights_from_mdl(self, mdl):
        self.net = mdl.net

    @staticmethod
    def get_nn(n_inputs, n_hidden, n_iterm_layers, nonlinearity, initialization, prev_net=None):
        layers = []
        # Get nonlinerity
        if nonlinearity.lower() == 'relu':
            nl = nn.ReLU(True)
        elif nonlinearity.lower() == 'tanh':
            nl = nn.Tanh()
        else:
            raise ValueError('invalid nonlinearity {}'.format(nonlinearity))
        layers += [nn.Linear(n_inputs, n_hidden), nl]
        for i in range(n_iterm_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nl]
        layers += [nn.Linear(n_hidden, 1)]
        net = nn.Sequential(*layers)
        # Initialize modules
        for m in net.modules():
            if isinstance(m, nn.Linear):
                if initialization == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity.lower())
                elif initialization == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Reuse weights from previous model when it is the case
        if prev_net is not None:
            FullyConnectedNet.reuse_weights(net, prev_net)
        return net

    @staticmethod
    def reuse_weights(net, net_with_weights):
        for m, mw in zip(net.modules(), net_with_weights.modules()):
            if isinstance(m, nn.Linear) and isinstance(mw, nn.Linear):
                p, q = m.weight.shape
                pl, ql = mw.weight.shape
                if p < pl or q < ql:
                    raise ValueError("mdl.shape ({}, {}) < mdl_with_weights.shape ({}, {})".format(p, q, pl, ql))
                with torch.no_grad():
                    m.weight[:pl, :ql] = mw.weight
                    m.bias[:pl] = mw.bias

    @staticmethod
    def _train(ep, net, optimizer, loader, n_total, grad_clipping, device, verbose=True):
        net = net.train()
        total_loss = 0
        n_entries = 0
        desc = "Epoch {:2d}: train - Loss: {:.6f}"
        if verbose:
            pbar = tqdm(initial=0, leave=True, total=n_total,
                        desc=desc.format(ep, 0), position=0)
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, outputs = data
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            predictions = net(inputs)
            loss = nn.functional.mse_loss(predictions.flatten(), outputs.flatten())
            loss.backward()
            if grad_clipping > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clipping)
            optimizer.step()
            # Update
            bs = len(outputs)
            total_loss += loss.detach().cpu().numpy() * bs
            n_entries += bs
            # Update train bar
            if verbose:
                pbar.desc = desc.format(ep, total_loss / n_entries)
                pbar.update(bs)
        if verbose:
            pbar.close()
        return total_loss / n_entries

    @staticmethod
    def _eval(net, loader, n_total, device, verbose=True):
        net.eval()
        n_entries = 0
        predictions_list = []
        if verbose:
            pbar = tqdm(initial=0, leave=True, total=n_total,
                         position=0)
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, = data
            inputs = inputs.to(device)
            with torch.no_grad():
                predictions = net(inputs)
            # Update
            predictions_list.append(predictions)
            bs = len(predictions)
            n_entries += bs
            # Update train bar
            if verbose:
                pbar.update(bs)
        if verbose:
            pbar.close()
        return torch.cat(predictions_list).detach().cpu().flatten().numpy()

    def fit(self, X, y):
        X = np.atleast_2d(X)
        n_total, n_in = X.shape
        torch.manual_seed(self.random_state)
        net = self.get_nn(n_in, self.n_features, self.n_interm_layers, self.nonlinearity, self.initialization, self.net)
        net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, nesterov=self.nesterov)
        if self.decay_rate < 1.0:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.decay_rate)
        X = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        y = torch.from_numpy(y).to(self.device, dtype=torch.float32)
        dset = torch.utils.data.TensorDataset(X, y)
        loader = DataLoader(dset, batch_size=32, shuffle=True)
        for ep in range(self.epochs):
            _loss = self._train(ep, net, optimizer, loader, n_total, self.grad_clipping, self.device, self.verbose)
            if self.verbose:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                tqdm.write('Train loss : {:.6f},  Lr: {:.6f}'.format(_loss, current_lr))
            if self.decay_rate < 1.0:
                scheduler.step()
        self.net = net
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        n_total, n_features = X.shape
        X = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        if n_total < self.batch_size:
            y = self.net(X).detach().cpu().flatten().numpy()
        else:
            dset = torch.utils.data.TensorDataset(X)
            loader = DataLoader(dset, batch_size=self.batch_size, shuffle=False)
            y = self._eval(self.net, loader, n_total, self.device, self.verbose)
        return y

    def __repr__(self):
        return '{}({},{},{},{},{},{},{},{},{},{},{},{})'.format(
            type(self).__name__, self.n_features, self.n_interm_layers,
            self.nonlinearity, self.lr, self.momentum, self.nesterov, self.epochs, self.batch_size,
            self.total_decay, self.grad_clipping, self.initialization, self.random_state)


class Linear(object):
    def __init__(self):
        self.estim_param = None

    def fit(self, X, y):
        self.estim_param, _resid, _rank, _s = linalg.lstsq(X, y)
        return self

    def predict(self, X):
        return X @ self.estim_param

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
