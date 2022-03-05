# etic-experiments

This repository contains the code to reproduce the experiments 
in this [paper](https://arxiv.org/abs/2112.15265) at AISTATS 2022.
The paper introduces the entropy regularized optimal transport independence criterion and apply it to test for independence.

## Dependencies
The code is written in Python and the dependencies are:
- python >= 3.6
- ipykernel >= 6.4.1
- matplotlib >= 3.5.0
- numpy >= 1.19.1
- pathos >= 0.2.8
- pot >= 0.6.0
- scikit-learn >= 0.22.1
- scipy >= 1.6.2
- seaborn >= 0.11.2

**Conda Environment**:
We recommend using a [conda environment](https://docs.conda.io/en/latest/miniconda.html).
To setup the environment, run
```bash
conda env create --file environment.yml
# activate the environment
conda activate etic
python -m ipykernel install --user --name etic
```

## Files

* `ind_tests.py`: code implementing independence tests.
* `utils.py` and `file_utils.py`: utility functions.
* `experiments.ipynb`: Jupyter notebook to run experiments and produce plots.


## Citation
If you find this repository useful, or you use it in your research, please cite:
```
@inproceedings{liu2022entropy,
  title={{Entropy Regularized Optimal Transport Independence Criterion}},
  author={Liu, Lang and Pal, Soumik and Harchaoui, Zaid},
  booktitle={AISTATS},
  year={2022}
}
```
    
## Acknowledgements
This work was supported by NSF DMS-2023166, NSF CCF-2019844, NSF DMS-2052239, PIMS CRG (PIHOT), NSF DMS-2134012, CIFAR-LMB, and faculty research awards.
