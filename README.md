# PYDRSOM: A PyTorch version for DRSOM



`pydrsom` is a [PyTorch](https://pytorch.org/docs/stable/optim.html) implementation of DRSOM (Dimension-Reduced Second-Order Method).

- See [quickstart](pydrsom/pydrsom/quickstart.py) for an example.
- The `demo` included in this repo with a test on CIFAR10 is due to [adabound](https://github.com/Luolc/AdaBound)

# Getting started

- PYDRSOM is developed in `Python 3.8 (torch=1.11.0)`. It is easy to setup, see [requirements](requirements.txt) for dependencies.
- The DRSOM optimizer class provides a couple parameters, see the docstring for details [drsom.py](pydrsom/drsom.py)
  - generally, you only have to choose which type of trust-region to use by arg `option_tr`:
  ```python
  # option of trust-region, I or G?
  #  - if 'a'; G = eye(2)
  #  - if 'p'; G = [-g d]'[-g d]
  ```

## Fashion-MNIST

For usage, start at the root of this repo:

```bash
python quickstart.py -h
```

For example,
- use a simple network to train Fashion-MNIST

```bash
python quickstart.py --optim drsom
```

- or choose a CNN model
```bash
python quickstart.py --optim drsom --model cnn
```

## Adjust verbosity

If you want to see **very** detailed logs for DRSOM (which by default is turned off), try:

```bash
export DRSOM_VERBOSE=1; python quickstart.py --optim drsom --model cnn
```

Then you can see the inner interation information for each "mini-batch", e.g.,

```bash
+----+-----+-------------------+----------+-------+--------+-------+-------+-------+-------+---------+-------+-------+------+--------+------+
|    |   𝜆 | Q/c/G             | a        |   ghg |   ghg- |    dQ |    df |   rho |   acc |   acc-𝜆 |     𝛄 |    𝛄- |    f |      k |   k0 |
+====+=====+===================+==========+=======+========+=======+=======+=======+=======+=========+=======+=======+======+========+======+
|  0 |   0 | [[ 3.046  0.   ]  | [[0.58]  |  3.05 |   3.05 | 0.512 | 0.498 | 0.973 |     1 |       1 | 1e-12 | 1e-06 | 2.31 |     +0 |    1 |
|    |     |  [ 0.     0.   ]  |  [0.  ]] |       |        |       |       |       |       |         |       |       |      |        |      |
|    |     |  [-1.766  0.   ]  |          |       |        |       |       |       |       |         |       |       |      |        |      |
|    |     |  [ 1.766  0.   ]  |          |       |        |       |       |       |       |         |       |       |      |        |      |
|    |     |  [ 0.     0.   ]] |          |       |        |       |       |       |       |         |       |       |      |        |      |
+----+-----+-------------------+----------+-------+--------+-------+-------+-------+-------+---------+-------+-------+------+--------+------+
```

Some description:

- $\lambda, Q, c, G, a (\alpha), f, rho (\rho)$ correspond to the definition in the paper.
- $k, k0$ are total iteration # and inner iteration (trust-region) #, respectively.
- $dQ, df$ are model reduction and actual reduction, respectively. then you can find the value for rho $(\rho)$
- $\gamma, \gamma-$ are current and last value for $\gamma_k$, respectively. 

## CIFAR10
We also provide a preliminary script for CIFAR10. Please refer to the code: `demos/cifar10/main.py`. This script is based on the training script of [adabound](https://github.com/Luolc/AdaBound).

For usage, start at the root of this repo:

```bash
python -u -m demos.cifar10.main -h
```
A example run:

```bash
python -u -m demos.cifar10.main \
  --model resnet18 --optim drsom --epoch 50 --option_tr p --gamma_power 1e3
```

## Known issues
`DRSOM.jl` is still under active development. Please add issues on GitHub.

## License
pydrsom is licensed under the MIT License. Check `LICENSE` for more details

## Acknowledgment

- Special thanks go to the COPT team and Tianyi Lin [(Darren)](https://tydlin.github.io/) for helpful suggestions.
## Reference 
You are welcome to cite our paper :), [see](https://arxiv.org/abs/2208.00208)
```bibtex
@misc{zhang2022drsom,
      title={DRSOM: A Dimension Reduced Second-Order Method and Preliminary Analyses}, 
      author={Chuwen Zhang and Dongdong Ge and Bo Jiang and Yinyu Ye},
      year={2022},
      eprint={2208.00208},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```