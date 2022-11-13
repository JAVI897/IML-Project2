# Work 2 Dimensionality reduction exercise

### Dependencies

- numpy
- matplotlib
- seaborn
- pandas
- sklearn

### Installation 

This repository was written and tested for python 3.7. To install the virtual environment and the required libraries on windows, run:

```bash
python -m venv group5_work2
group5_work2\Scripts\activate.bat
pip install -r requirements.txt
```

On Unix or MacOS, run:

```bash
python -m venv group5_work2
source group5_work2/bin/activate
pip install -r requirements.txt
```


### Usage

You can run the models with:

```python
python main.py --dataset <dataset>
               --dimReduction <dimReduction>
               --tsne <tsne>
               --perplexity_analysis <perplexity_analysis>
               --num_dimensions <num_dimensions>
               --clusteringAlg <clusteringAlg>
               --max_num_clusters <max_num_clusters> 
               --visualize_results <visualize_results>
               --plot_scores_colored_by_cluster <plot_scores_colored_by_cluster>
               --affinity <affinity>
               --linkage <linkage>
```
Specifying the parameters  according to the following table:

| Parameter           | Description                                                                                                                                                                                                                                                                  |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **dataset**         | Dataset to use. If set to 'vote', the vote dataset is used. If set to 'hyp', the Hypothyroid dataset will be used. If set to 'vehi', the vehicle dataset will be used.                                                                                                       |
| **dimReduction**    | Dimensionality reduction algorithm to use. If set to 'pca', the PCA algorithm is used. If set to 'fa', the FA algorithm is used. If set to 'pca_sk', sklearn's PCA implementation will be used. If set to 'ipca', Incremental PCA from sklearn will be used.                 |
| **compute_tsne**            | If set to True, the t-SNE technique will be computed to visualize the chosen dataset.                                                                                                                                                                                        |
| **perplexity_analysis** | If set to True, the t-SNE technique will be computed for different perplexity values to find visually the best parameter for the chosen dataset.                                                                                                                             |
| **num_dimensions**  | Number of dimensions used in the corresponding dimensionality reduction algorithm.                                                                                                                                                                                           |                   
| **clusteringAlg**   | Clustering algorithm to use. If set to 'km', the k-means algorithm is used. If set to 'agg', the agglomerative clustering algorithm will be used.                                                                                                                            
| **max_num_clusters** | The max_num_cluster parameter will be used to evaluate the algorithms with the specified number of clusters. Results will be saved in a csv file in the results folder.                                                                                                      |
| **visualize_results** | If set to True, different plots will be generated to evaluate the DBI, SC and CH metrics which will be saved in the plot folder of the corresponding dataset.                                                                                                                |  
| **plot_scores_colored_by_cluster** | If set to True, PCA and t-SNE plots are coloured by cluster.                                                                                                                                                                                                                 
| **affinity**        | This parameter is only used if the clusteringAlg parameter is set to 'agg'. Denotes the affinity distance to use. Possible choices: ['euclidean', 'cosine'].                                                                                                                 |
| **linkage**         | This parameter is only used if the clusteringAlg parameter is set to 'agg'. Denotes the kind of linkage to use. Possible choices: ['ward', 'complete', 'average', 'single'].                                                                                                 |
| **analyze_fa_components**         | If set to True, the Silhouette Coefficient will be computed for K-means and Agglomerative Clustering for different number of components in the Feature Agglomeration dimensionality reduction method. Plots will be saved in the corresponding folder of the plot's dataset. |