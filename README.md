# Work 2 Dimensionality reduction exercise

### Dependencies

- numpy
- matplotlib
- seaborn
- statsmodels
- pandas
- pyclustertend
- sklearn

### Usage

You can run the clustering models with:

```python
python main.py --dataset <dataset>
               --dimReduction <dimReduction>
               --tsne <tsne>
               --num_dimensions <num_dimensions>
               --clusteringAlg <clusteringAlg>
               --max_num_clusters <max_num_clusters> 
               --visualize_results <visualize_results>
               --plot_scores_colored_by_cluster <plot_scores_colored_by_cluster>
               --affinity <affinity>
               --linkage <linkage>
```
Specifying the parameters  according to the following table:

| Parameter           | Description                                                                                                                                                                                                                                                                                                                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **dataset**         | Dataset to use. If set to 'vote', the vote dataset is used. If set to 'hyp', the Hypothyroid dataset will be used. If set to 'vehi', the vehicle dataset will be used.                                                                                                                                                                    |
| **dimReduction**    | Dimensionality reduction algorithm to use. If set to 'pca', the PCA algorithm is used. If set to 'fa', the FA algorithm is used.                                                                                                                                                                                                                       |
| **tsne**            | If set to True, the t-SNE technique will be computed to visualize the chosen dataset.                                                                                                                                                                                                                                                                                                        |
| **num_dimensions**  | Number of dimensions to find on the dimensionality reduction.
| **clusteringAlg**   | Clustering algorithm to use. If set to 'km', the k-means algorithm is used. If set to 'bkm', the bisecting k-means algorithm will be used. If set to 'ms', the mean-shift algorithm will be used. If set to 'agg', the agglomerative clustering algorithm will be used. If set to 'kmed', the k-medoids algorithm will be used. If set to 'fuzzy', the fuzzy c-means algorithm will be used. |
| **max_num_clusters** | If num_clusters is not specified, the max_num_cluster parameter will be used to evaluate the algorithms with different number of clusters. Results will be saved in a csv file in the results folder.                                                                                                                                                                                        |
| **visualize_results** | If set to True, different plots will be generated to evaluate the DBI, SC and CH metrics which will be saved in the plot folder of the corresponding dataset.                                                                                                                                                                                                                                |  
| **plot_scores_colored_by_cluster** | If set to True, PCA and t-SNE plots are coloured by cluster.
| **affinity**        | This parameter is only used if the clusteringAlg parameter is set to 'agg'. Denotes the affinity distance to use. Possible choices: ['euclidean', 'cosine'].                                                                                                                                                                                                                                 |
| **linkage**         | This parameter is only used if the clusteringAlg parameter is set to 'agg'. Denotes the kind of linkage to use. Possible choices: ['ward', 'complete', 'average', 'single'].                                                                                                                                                                                                                 |

Results for the fuzzy c-means clustering are saved in a subfolder called c_means for each dataset plot folder.
