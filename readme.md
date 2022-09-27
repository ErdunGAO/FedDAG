# FedDAG: Federated DAG Structure Learning

This repository contains an implementation of the federated DAG structure learning methods described in ["FedDAG: Federated DAG Structure Learning"](https://arxiv.org/abs/2112.03555).

If you find it useful, please consider citing:
```bibtex
@article{gao2021federated,
  title={Federated Causal Discovery},
  author={Gao, Erdun and Chen, Junjia and Shen, Li and Liu, Tongliang and Gong, Mingming and Bondell, Howard},
  journal={arXiv preprint arXiv:2112.03555},
  year={2021}
}
```

## Requirements
- CUDA 10.0
- cuDNN 7.4
- Anaconda

Use `conda env create -f environment.yml` to create a `feddag_test_env` conda environment.

Use [10.0-cudnn7-devel-ubuntu18.04](https://hub.docker.com/layers/nvidia/cuda/10.0-cudnn7-devel-ubuntu18.04/images/sha256-3780926c6209d27d62b2a0fb057b8b02f621fc02b0e3d6a724c1089885864202?context=explore) to create a container if running from docker.

## Examples
After creating a new conda environment, you can run `bash test_run.sh` to test the codes. Our logging results of the `test_run.sh` is recorded in [log_example.txt](https://github.com/ErdunGAO/FedDAG/blob/main/log_example.txt). If you want to test our method with other scales of graphs, change the hyper-parameters according to settings provided in the Appendix.

## Acknowledgments
- Our implementation is highly based on the existing Tool-chain named gcastle [pip link](https://pypi.org/project/gcastle/) and [paper link](https://arxiv.org/abs/2111.15155), which includes many gradient-based DAG structure learning methods.
- Our implementation is also highly based on [NOTEARS-tensorflow](https://github.com/ignavierng/notears-tensorflow) and [MCSL](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle/castle/algorithms/gradient/mcsl/torch).

## Recommendations
- Notice that [NOTEARS-ADMM](https://arxiv.org/abs/2110.09356) is a concurrent and interesting work that also considers the same problem with our FedDAG. In NOTEARS-ADMM, ADMM is leveraged to jointly learn the graph. 
- The baseline method of our FedDAG is [MCSL](https://arxiv.org/abs/1910.08527). Please read this paper if you have concerns about the basic modules of FedDAG.

You are highly recommended to read these two papers and to cite them.

```bibtex
@inproceedings{Ng2022federated,
  author = {Ng, Ignavier and Zhang, Kun},
  title = {Towards Federated Bayesian Network Structure Learning with Continuous Optimization},
  booktitle = {International Conference on Artificial Intelligence and Statistics},
  year = {2022},
}

@inproceedings{ng2022masked,
  title={Masked gradient-based causal structure learning},
  author={Ng, Ignavier and Zhu, Shengyu and Fang, Zhuangyan and Li, Haoyang and Chen, Zhitang and Wang, Jun},
  booktitle={Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
  pages={424--432},
  year={2022},
  organization={SIAM}
}
```
