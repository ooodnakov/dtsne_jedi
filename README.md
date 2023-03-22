# dtSNE and JEDI algorithms
dtSNE and JEDI algorithms for Skoltech's ML 2023 course

## dtSNE
### Reference
Based on these papers:

Jonas Fischer, Rebekka Burkholz, Jilles Vreeken, *Preserving local densities in low-dimensional embeddings*, https://arxiv.org/pdf/2301.13732.pdf

L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008. https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf

Code is based on original implementation found [here][https://lvdmaaten.github.io/tsne/code/tsne_python.zip]

### Running
To run dtSNE with 300 optimizer iteratiion and 30 perplexity use this:
```shell
python tsne.py --dens 1 --iter 300 --perp 30
```
To use original tSNE implementation, change `dens` flag to `0`: `--dens 0`.






[https://lvdmaaten.github.io/tsne/code/tsne_python.zip]: [https://](https://lvdmaaten.github.io/tsne/code/tsne_python.zip)