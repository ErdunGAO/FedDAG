import logging
import random
import numpy as np
import networkx as nx
from scipy.special import expit as sigmoid

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class DAG(object):
    '''
    A class for simulating random (causal) DAG, where any DAG generator
    method would return the weighed/binary adjacency matrix of a DAG.
    Besides, we recommend using the python package "NetworkX"
    to create more structures types.
    '''

    @staticmethod
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    @staticmethod
    def _random_acyclic_orientation(B_und):
        B = np.tril(DAG._random_permutation(B_und), k=-1)
        B_perm = DAG._random_permutation(B)
        return B_perm

    @staticmethod
    def _graph_to_adjmat(G):
        return nx.to_numpy_matrix(G)

    @staticmethod
    def _BtoW_2(B, d, w_range, num):
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[num, d, d])
        U[np.random.rand(num, d, d) < 0.5] *= -1
        W = (B != 0).astype(float) * U
        return W

    @staticmethod
    def er_graph(n_nodes, n_edges, weight_range=None, seed=None, num_client=10):

        assert n_nodes > 0
        set_random_seed(seed)
        # Erdos-Renyi
        creation_prob = (2 * n_edges) / (n_nodes ** 2)
        G_und = nx.erdos_renyi_graph(n=n_nodes, p=creation_prob, seed=seed)
        B_und = DAG._graph_to_adjmat(G_und)
        B = DAG._random_acyclic_orientation(B_und)
        if weight_range is None:
            return B
        else:
            W_2 = DAG._BtoW_2(B, n_nodes, weight_range, num_client)
        return B, W_2

    @staticmethod
    def sf_graph(n_nodes, n_edges, weight_range=None, seed=None, num_client=10):

        assert (n_nodes > 0 and n_edges >= n_nodes and n_edges < n_nodes * n_nodes)
        set_random_seed(seed)
        # Scale-free, Barabasi-Albert
        m = int(round(n_edges / n_nodes))
        G_und = nx.barabasi_albert_graph(n=n_nodes, m=m)
        B_und = DAG._graph_to_adjmat(G_und)
        B = DAG._random_acyclic_orientation(B_und)
        if weight_range is None:
            return B
        else:
            W_2 = DAG._BtoW_2(B, n_nodes, weight_range, num_client)
        return B, W_2


class IIDSimulation(object):
    '''
    Simulate IID datasets for causal structure learning.

    Parameters
    ----------
    W: np.ndarray
        Weighted adjacency matrix for the target causal graph.
    n: int
        Number of samples for standard trainning dataset.
    method: str, (linear or nonlinear), default='linear'
        Distribution for standard trainning dataset.
    sem_type: str
        gauss, exp, gumbel, uniform, logistic (linear); 
        mlp, mim, gp, gp-add, quadratic (nonlinear).
    noise_scale: float
        Scale parameter of noise distribution in linear SEM.
    '''

    def __init__(self, W, n=1000, method='linear', 
                 sem_type='gp', noise_scale=1.0, seed=2022):

        self.B = (W != 0).astype(int)
        set_random_seed(seed)
        if method == 'linear':
            self.X = IIDSimulation._simulate_linear_sem(
                    W, n, noise_type='gauss', noise_scale=1)
        elif method == 'nonlinear':
            self.X = IIDSimulation._simulate_nonlinear_sem(
                    W, n, sem_type, noise_scale)
        logging.info('Finished synthetic dataset')

    @staticmethod
    def _simulate_linear_sem(W, n, noise_type, noise_scale):
        """
        Simulate samples from linear SEM with specified type of noise.
        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

        Parameters
        ----------
        W: np.ndarray
            [d, d] weighted adj matrix of DAG.
        n: int
            Number of samples, n=inf mimics population risk.
        noise_type: str 
            gauss, exp, gumbel, uniform, logistic.
        noise_scale: float 
            Scale parameter of noise distribution in linear SEM.
        
        Return
        ------
        X: np.ndarray
            [n, d] sample matrix, [d, d] if n=inf
        """
        def _simulate_single_equation(X, w, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if noise_type == 'gauss':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif noise_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif noise_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif noise_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif noise_type == 'logistic':
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            else:
                raise ValueError('Unknown noise type. In a linear model, \
                                 the options are as follows: gauss, exp, \
                                 gumbel, uniform, logistic.')
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx =  nx.from_numpy_matrix(W, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')
        if np.isinf(n):  # population risk for linear gauss SEM
            if noise_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            else:
                raise ValueError('population risk not available')
        # empirical risk
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        return X

    @staticmethod
    def _simulate_nonlinear_sem(W, n, sem_type, noise_scale):
        """
        Simulate samples from nonlinear SEM.

        Parameters
        ----------
        B: np.ndarray
            [d, d] binary adj matrix of DAG.
        n: int
            Number of samples.
        sem_type: str
            mlp, mim, gp, gp-add, or quadratic.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.

        Return
        ------
        X: np.ndarray
            [n, d] sample matrix
        """
        if sem_type == 'quadratic':
            return IIDSimulation._simulate_quad_sem(W, n, noise_scale)

        def _simulate_single_equation(X, scale):
            """X: [n, num of parents], x: [n]"""
            z = np.random.normal(scale=scale, size=n)
            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            if sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                        for i in range(X.shape[1])]) + z
            else:
                raise ValueError('Unknown sem type. In a nonlinear model, \
                                 the options are as follows: mlp, mim, \
                                 gp, gp-add, or quadratic.')
            return x

        B = (W != 0).astype(int)
        d = B.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        X = np.zeros([n, d])
        G_nx =  nx.from_numpy_matrix(B, create_using=nx.DiGraph)
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
        return X

def property_generation(num_client):
    """
        Simulate the random data property for each client.

        Parameters
        ----------
        num_client: Number of the clients

        Explainations
        ----------
        linearity: Decide if it is linear model.
        nonlinear_type: Decide the SEM type for each client.
        noise_scale: Decide the noise scale.
        Please refer to our experimental results section.

        Return
        ------
        X: np.ndarray
            [num_client, 3] sample matrix
        """

    linearity = np.random.randint(0, 2, (num_client, 1))
    nonlinear_type = np.random.randint(0, 4, (num_client, 1))
    noise_scale = np.random.randint(0, 2, (num_client, 1))

    return np.hstack([linearity, nonlinear_type, noise_scale])

def NonIID_Simulation(W_true,
                      dp,
                      n,
                      seed):

    num_client, d = W_true.shape[0], W_true.shape[1]
    dataset = []
    data_all = np.zeros([n, d])
    for i in range(num_client):
        #choice of noise scale
        noise_scale = 0.8 if dp[i,2] == 0 else 1
        #choice of linear/non-linear
        if dp[i,0] == 0:
            method, sem_type = 'linear', 'Gauss'
        else:
            method = 'nonlinear'
            if dp[i,1] == 0:
                sem_type = 'mlp'
            elif dp[i,1] == 1:
                sem_type = 'gp'
            elif dp[i,1] == 2:
                sem_type = 'mim'
            else:
                sem_type = 'gp-add'

        data_part = IIDSimulation(W_true[i],
                                  n=n,
                                  method=method,
                                  sem_type=sem_type,
                                  noise_scale=noise_scale,
                                  seed=seed)

        data_all = np.vstack((data_all, data_part.X))
        dataset.append(data_part.X)

    data_all = data_all[n:,:]
    return dataset, data_all

def Multi_IID_Simulation(W_true,
                         sem_type,
                         n,
                         method,
                         seed):

    num_client, _ = W_true.shape[0], W_true.shape[1]
    dataset = []
    data = IIDSimulation(W_true[0],
                         n=n*num_client,
                         method=method,
                         sem_type=sem_type,
                         noise_scale=1.0,
                         seed=seed)

    for i in range(num_client):
        data_part = data.X[i*n:(i+1)*n, :]
        dataset.append(data_part)

    return dataset, data.X