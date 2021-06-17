# Significant DBSCAN
(This is the new **Python** version for data with arbitrary dimensions; previous [matlab version](https://github.com/yqthanks/Significant-DBSCAN-matlab) is optimized for 2D spatial data)

**Code for paper** ([Best Paper Award](http://sstd2019.org/program.html) @ SSTD 2019):

*Xie, Y. and Shekhar, S., 2019, August. Significant DBSCAN towards Statistically Robust Clustering. In Proceedings of the 16th International Symposium on Spatial and Temporal Databases (pp. 31-40).* [ACM link](https://dl.acm.org/doi/abs/10.1145/3340964.3340968)

```
@inproceedings{xie2019significant,
  title={Significant DBSCAN towards Statistically Robust Clustering},
  author={Xie, Yiqun and Shekhar, Shashi},
  booktitle={Proceedings of the 16th International Symposium on Spatial and Temporal Databases},
  pages={31--40},
  year={2019}
}
```

We appreciate citing of this paper if the code/data is used for result comparison, etc. More versions (python version added)/data will be added soon.

Another related survey on statistically-robust clustering is available at [arXiv](https://arxiv.org/pdf/2103.12019.pdf)

**Sharing code to support reproducible research.**

## Description
This work aims to address a major limitation of traditional density-based clustering approach -- the lack of statistical rigor. This makes approaches such as DBSCAN tend to return many spurious clusters. For example, according to the [HDBSCAN](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14) paper: 
>"small clusters of objects that may be highly similar to each other just by chance, that is, as a consequence of the natural randomness associated with the use of a finite data sample"

The code implements significant DBSCAN to automatically remove spurious clusters through point-process based statistical tests.

Significant DBSCAN works particularly well when data has high volume of noise.

## Example
The following figure shows an example of results comparing DBSCAN and significant DBSCAN.

<!--![Example](https://github.com/yqthanks/significantDBSCAN/blob/master/example_results.png)-->
<img src="https://github.com/yqthanks/significantDBSCAN/blob/master/example_results.png" width="600">

## Usage
[demo.py](https://github.com/yqthanks/significantDBSCAN-python/blob/master/demo.py) includes a demo on how to run Significant DBSCAN.
[SignificantDBSCAN_share.ipynb](https://github.com/yqthanks/significantDBSCAN-python/blob/master/SignificantDBSCAN_share.ipynb) has everything together in one notebook.

## Data
Example datasets are given in the sample_data/demo folder.
More datasets and scripts for synthetic data generation are provided in the eariler [matlab-version's repo](https://github.com/yqthanks/Significant-DBSCAN-matlab/tree/master/synthetic_data)

## Example comparison with other clustering techniques

In the following, the first figure shows results on complete random data where there is no true cluster, and any cluster detected is spurious (happen only due to natural randomness). In contrast, the second figure shows detections on truly clustered data (governed by clustered point process; please see [paper](https://dl.acm.org/doi/abs/10.1145/3340964.3340968) for more details).

**Complete random data (no true cluster exists)**

<!-- ![randomdata](https://github.com/yqthanks/significantDBSCAN/blob/master/example_data_and_results/comparison1_random_data.png) -->
<img src="https://github.com/yqthanks/significantDBSCAN/blob/master/example_data_and_results/comparison1_random_data.png" width="600">


**Clustered data with noise in background (four shapes are true clusters; background are noises generated with homogeneous probability density)**

<!-- ![clustereddata](https://github.com/yqthanks/significantDBSCAN/blob/master/example_data_and_results/comparison2_clustered_data.png) -->
<img src="https://github.com/yqthanks/significantDBSCAN/blob/master/example_data_and_results/comparison2_clustered_data.png" width="600">

## License

The MIT License: AS IS (please see license file)
