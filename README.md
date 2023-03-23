# dtSNE and JEDI algorithms
dtSNE and JEDI algorithms for Skoltech's ML 2023 course

## Installation
To install this package, inside repository run this:
```python
python -m pip install .
```
or this from PyPi:
```python
python -m pip install dtsnejedi
```

We hope that after that you can just import it via `import dtsnejedi`

## dtSNE
### Reference
Based on these papers:

* Jonas Fischer, Rebekka Burkholz, Jilles Vreeken, *Preserving local densities in low-dimensional embeddings*, https://arxiv.org/pdf/2301.13732.pdf

* L.J.P. van der Maaten and G.E. Hinton. *Visualizing High-Dimensional Data Using t-SNE*. Journal of Machine Learning Research 9(Nov):2579-2605, 2008. https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf

Code is based on original implementation found [here](https://lvdmaaten.github.io/tsne/code/tsne_python.zip)

### Running
To run dtSNE with 300 optimization iterations and 30 perplexity on toy-data, use this:
```shell
python test.py --algo dtsne --n_iter 300 --perp 30
```
To use original tSNE implementation, change `algo` flag to `tsne`: `--algo tsne`.

For into about other flags, see `test.py` file.

## JEDI
### Reference
Based on these papers:

* Edith Heiter, Jonas Fischer, Jilles Vreeken, *Factoring out prior knowledge from low-dimensional embeddings*, https://arxiv.org/pdf/2103.01828.pdf

Code is based on original tSNE implementation found [here](https://lvdmaaten.github.io/tsne/code/tsne_python.zip).

### Running
To run JEDI with 300 optimization iterations and 30 perplexity on toy-data, use this:
```shell
python test.py --algo jedi --n_iter 300 --perp 30
```

For into about other flags, see `test.py` file.

## Notes
* Seems like, this tSNE and dtSNE algorimth returns matrix rotated by 180 degrees compared to other tSNE implementations (like in openTSNE and sklearn.manifold). It can be fixed by multiplying 180 degree rotation matrix.
* All of these algorithms are extremely slow. Maybe in the future, we can do similar optimizations found in openTSNE.