from . import DAG
from .simulation import NonIID_Simulation, Multi_IID_Simulation

def data_gen(graph_type,
             node,
             edge,
             seed,
             num_client,
             gen_method,
             n,
             sem_type,
             dataset_property=None,
             method='nonlinear'):
    
    """
    Simulate the random data property for each client.

    Parameters
    ----------
    graph_type       : the type of graph, choose from ['er', 'sf'].
    node             : number of nodes.
    edge             : number of edges.
    seed             : seed.
    num_client       : number of the clients.
    gen_method       : data generation tyep, choose from ['multiiid', 'noniid'].
    n                : number of observations on each client.
    sem_type         : the sem_type for iid data.
    dataset_property : the property of the data.
    method           : linear or nonlinear data, choose from ['linear', 'nonlinear'].

    Return
    ------
    B_true           : the binary DAG graph (matrix).
    W_true           : the weight matrixs for each client.
    dataset          : the generated dataset.
    data_all         : put all data together.

    """

    if graph_type == 'er':
        B_true, W_true = DAG.er_graph(n_nodes=node,
                                      n_edges=edge,
                                      weight_range=(0.5, 2.0),
                                      seed=seed,
                                      num_client=num_client)
    elif graph_type == 'sf':
        B_true, W_true = DAG.sf_graph(n_nodes=node,
                                      n_edges=edge,
                                      weight_range=(0.5, 2.0),
                                      seed=seed,
                                      num_client=num_client)
    else:
        assert False, "invalid graph type {}".format(graph_type)

    if gen_method == 'noniid':
        dataset, data_all = NonIID_Simulation(W_true,
                                              dataset_property,
                                              n,
                                              seed
                                              )
    elif gen_method == 'multiiid':
        dataset, data_all = Multi_IID_Simulation(W_true,
                                                 sem_type,
                                                 n,
                                                 method,
                                                 seed)

    else:
        assert False, "invalid gen_method {}".format(gen_method)

    return B_true, W_true, dataset, data_all