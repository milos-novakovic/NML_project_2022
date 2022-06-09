# Network Machine Learning final project at EPFL

Authors

Mateja Ilić ( ilic.mateja@epfl.ch )

Miloš Novaković ( milos.novakovic@epfl.ch ) 

Group: 9

## Dataset [AmazonProducts](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/amazon_products.html)

The Amazon dataset from the ["GraphSAINT: Graph Sampling Based Inductive Learning Method"](https://arxiv.org/abs/1907.04931) paper, containing products and its categories.

The Amazon dataset is processed from the bipartite user-item graph from Amazon.

The problem here is to predict an Amazon product category based on the text of the reviews. 

Details for the Amazon dataset are as follows:

* Node: One node in the Amazon Dataset represents one product listed on the Amazon website.
* Edge: We add an edge between two nodes A and B if the buyers of product A and the buyers of B overlaps.
* Node feature: The node feature contains information of all the reviews on that product. The raw features of each user are provided as a sparse vector of character 4-grams from the reviews. Each non-zero element in the vector represents the count of the 4-gram. Since the raw feature vectors are too long (length-100k for each node), we use SVD to reduce the dimensionality down to length-200 to serve as the GCN inputs.
* Node label: The label of one node represents the categories of the product (e.g., books, movie, shoes). In the original dataset, there exists a lot of "rare categories" that correspond to very few number of nodes. We eliminate those categories and keep only the most frequently occuring 107 categories. Note: as a result, a small percent of nodes may now have no label because they do not belong to any of the 107 categories. We leave them as they are because we think this just reflects the nature of the dataset. 
* Train/Val/Test split: The Train/Val/Test nodes are randomly split to 0.85/0.05/0.10. The training adjacency matrix is generated by the induced sub-graph of the training nodes.

Raw and proceseed data used for

1. the **GraphSAINT** model is located [here](https://drive.google.com/drive/folders/19qW5aq0C17Zqvv9sDmZn1sxlv3MPdYm1?usp=sharing).
2. the **MLP** and **GraphSAGE** models is located [here](https://drive.google.com/drive/folders/1KPE50zojEd0jMAY58ypHdF6sqfu71pb8?usp=sharing).



> Keep the README fresh! It's the first thing people see and will make the initial impression.


```
bash -c "$(curl -s https://raw.githubusercontent.com/CFPB/development/main/open-source-template.sh)"
```

----


## Dependencies
Required packages 

* torch
* torch_geometric
* torchmetrics
* torch-scatter
* torch-sparse
* torch-cluster
* torch-geometric
* matplotlib.pyplot
* numpy
* dgl
* sklearn
* class-resolver


## Results

Results are provided in different locations for different models

1. ***GraphSAINT*** model training and evaluation output values are stored in two places, i.e., for the [node-sampler-output-file](GraphSAINT/node_sampler_training_evaluation_test_cleared.txt) and for the [random-walk-sampler-output-file](GraphSAINT/random_walk_sampler_training_evaluation_test_cleared.txt).
2. ***MLP*** model training and evaluation output values are stored in the corresponding [ipynb-file](MLP/Project_exploitation_MLP.ipynb).
3. ***GraphSAGE*** model training and evaluation output values are stored in the corresponding [ipynb-file](GraphSAGE/Project_exploitation_GraphSAGE.ipynb).

## How to test the software

If the software includes automated tests, detail how to run those tests.

## Known issues

Document any known significant shortcomings with the software.

## Getting help

Instruct users how to get help with this software; this might include links to an issue tracker, wiki, mailing list, etc.

**Example**

If you have questions, concerns, bug reports, etc, please file an issue in this repository's Issue Tracker.

## Getting involved




----

## Open source licensing info
1. [LICENSE](LICENSE)


----

## Credits and references

1. Projects that inspired you
2. Related projects
3. Books, papers, talks, or other sources that have meaningful impact or influence on this project
