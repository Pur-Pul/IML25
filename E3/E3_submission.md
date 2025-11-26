# Problem 17
## Task a
1. **For what kinds of tasks can we use the k-means algorithm?**
- K-means algorithm can be used for clustering, which is useful in unsupervised learning where observations do not fall under any predefined class. K-means group similar observations together into K number of clusters. 
2. **What are the algorithm’s inputs and outputs?**
- K-means takes a number K and a dataset with n observations with p features. The only requirement is that $K >= n$. The algorithm outputs cluster assignments for the dataset and the centroids for each cluster. Each observation is assigned to one of K clusters.
3. **How should you interpret the results?**
The clusters assignments are groups of observations that are close to eachother, or in other words, have similar features. The centroids are vectors with components that represent the average values for each feature within the cluster.

## Task b

K-means aims to minimize the within-cluster variation. Instead of calculating the variation by taking the difference between each observation in the cluster, the algorithm takes the distance from the observations to the cluster centroids. The cluster centroids contain the feature means of the observations in the cluster.

The mean for feature $j$ in cluster $C_k$
$$
\frac{1}{C_k} \sum_{i \in C_k}x_{ij}
$$

The euclidian distance between observation $x$ and the centroid of cluster $C_k$
$$
\sqrt{\sum^p_{j=1} ( x_{ij}-\frac{1}{C_k} \sum_{i \in C_k}x_{ij} )^2 }
$$

The mean of the distances for each observation in the cluster to the cluster centroid are calculated. This gives a metric of how similar the observations in the cluster are to eachother.

$$
\frac{1}{|C_k|} \sum_{i \in C_k}\sqrt{\sum^p_{j=1} ( x_{ij}-\frac{1}{C_k} \sum_{i \in C_k}x_{ij} )^2 }
$$

Summing the afformentioned means becomes the objective function, which needs to be minimized

$$
\sum^K_{k=1} \frac{1}{|C_k|} \sum_{i \in C_k}\sqrt{\sum^p_{j=1} ( x_{ij}-\frac{1}{C_k} \sum_{i \in C_k}x_{ij} )^2 }
$$

## Task c
toy data = 
| i | x | y |
|---|---|---|
| 1 | 0 | 1 | 
| 2 | 1 | 2 | 
| 3 | 4 | 5 |
| 4 | 5 | 3 |
| 5 | 5 | 4 |
### Iteration 1
$\mu_1={1,4}$
$\mu_2={4,1}$


| i | $\mu$ | Calculation |
| - | - | - |
| 1 | 1 | $\sqrt{(0 - 1)^2 + (1 - 4)^2} = \sqrt{1 + 9} = 3.1622...$ |
| 1 | 2 | $\sqrt{(0 - 4)^2 + (1 - 1)^2} = \sqrt{16 + 0} = 4$ |
| 2 | 1 | $\sqrt{(1 - 1)^2 + (2 - 4)^2} = \sqrt{0 + 4} = 2$ |
| 2 | 2 | $\sqrt{(1 - 4)^2 + (2 - 1)^2} = \sqrt{9 + 1} = 3.1622...$ |
| 3 | 1 | $\sqrt{(4 - 1)^2 + (5 - 4)^2} = \sqrt{9 + 1} = 3.1622...$ |
| 3 | 2 | $\sqrt{(4 - 4)^2 + (5 - 1)^2} = \sqrt{0 + 16} = 4$ |
| 4 | 1 | $\sqrt{(5 - 1)^2 + (3 - 4)^2} = \sqrt{16 + 1} = 4.1231...$ |
| 4 | 2 | $\sqrt{(5 - 4)^2 + (3 - 1)^2} = \sqrt{1 + 4} = 2.2360$ |
| 5 | 1 | $\sqrt{(5 - 1)^2 + (4 - 4)^2} = \sqrt{16 + 0} = 4$ |
| 5 | 2 | $\sqrt{(5 - 4)^2 + (4 - 1)^2} = \sqrt{1 + 9} = 3.1622...$ |

#### Cluster assignments
| Cluster | Observations | mean vector |
| - | - | - |
| 1 | 1, 2, 3 | $\frac{(0, 1) + (1, 2) + (4, 5)}{3} = (\frac{5}{3},\frac{8}{3})$ |
| 2 | 4, 5 | $\frac{(5,3) + (5,4)}{2} = (5,\frac{7}{2})$ |

### Iteration 2
$\mu_1 = (\frac{5}{3},\frac{8}{3})$
$\mu_2 = (5,\frac{7}{2})$

| i | $\mu$ | Calculation |
| - | - | - |
| 1 | 1 | $\sqrt{(0 - \frac{5}{3})^2 + (1 - \frac{8}{3})^2} = \sqrt{\frac{5}{3}^2 + \frac{5}{3}^2} = 2.3570...$ |
| 1 | 2 | $\sqrt{(0 - 5)^2 + (1 - \frac{7}{2})^2} = \sqrt{25 + 2.5^2} = 5.5901...$ |
| 2 | 1 | $\sqrt{(1 - \frac{5}{3})^2 + (2 - \frac{8}{3})^2} = \sqrt{\frac{2}{3}^2 + \frac{2}{3}^2} = 0.9428...$ |
| 2 | 2 | $\sqrt{(1 - 5)^2 + (2 - \frac{7}{2})^2} = \sqrt{16 + 1.5^2} = 4.2720...$ |
| 3 | 1 | $\sqrt{(4 - \frac{5}{3})^2 + (5 - \frac{8}{3})^2} = \sqrt{\frac{7}{3}^2 + \frac{7}{3}^2} = 3.2998...$ |
| 3 | 2 | $\sqrt{(4 - 5)^2 + (5 - \frac{7}{2})^2} = \sqrt{1 + 1.5^2} = 1.8027...$ |
| 4 | 1 | $\sqrt{(5 - \frac{5}{3})^2 + (3 - \frac{8}{3})^2} = \sqrt{\frac{10}{3}^2 + \frac{1}{3}^2} = 3.3499...$ |
| 4 | 2 | $\sqrt{(5 - 5)^2 + (3 - \frac{7}{2})^2} = \sqrt{0 + 0.5^2} = 0.5$ |
| 5 | 1 | $\sqrt{(5 - \frac{5}{3})^2 + (4 - \frac{8}{3})^2} = \sqrt{\frac{10}{3}^2 + \frac{4}{3}^2} = 3.5901...$ |
| 5 | 2 | $\sqrt{(5 - 5)^2 + (4 - \frac{7}{2})^2} = \sqrt{0 + 0.5^2} = 0.5$ |

#### Cluster assignments
| Cluster | Observations | mean vector |
| - | - | - |
| 1 | 1, 2 | $\frac{(0, 1) + (1, 2)}{2} = (0.5,1.5)$ |
| 2 | 3, 4, 5 | $\frac{(4, 5) + (5,3) + (5,4)}{3} = (\frac{14}{3},4)$ |

### Iteration 3
$\mu_1 = (0.5,1.5)$
$\mu_2 = (\frac{14}{3},4)$

| i | $\mu$ | Calculation |
| - | - | - |
| 1 | 1 | $\sqrt{(0 - 0.5)^2 + (1 - 1.5)^2} = \sqrt{0.5^2 + 0.5^2} = 0.7071...$ |
| 1 | 2 | $\sqrt{(0 - \frac{14}{3})^2 + (1 - 4)^2} = \sqrt{\frac{14}{3}^2 + 9} = 5.5477...$ |
| 2 | 1 | $\sqrt{(1 - 0.5)^2 + (2 - 1.5)^2} = \sqrt{0.5^2 + 0.5^2} = 0.7071...$ |
| 2 | 2 | $\sqrt{(1 - \frac{14}{3})^2 + (2 - 4)^2} = \sqrt{\frac{11}{3}^2 + 4} = 4.1766...$ |
| 3 | 1 | $\sqrt{(4 - 0.5)^2 + (5 - 1.5)^2} = \sqrt{3.5^2 + 3.5^2} = 4.9497...$ |
| 3 | 2 | $\sqrt{(4 - \frac{14}{3})^2 + (5 - 4)^2} = \sqrt{\frac{2}{3}^2 + 1} = 1.2018...$ |
| 4 | 1 | $\sqrt{(5 - 0.5)^2 + (3 - 1.5)^2} = \sqrt{4.5^2 + 1.5^2} = 4.7434...$ |
| 4 | 2 | $\sqrt{(5 - \frac{14}{3})^2 + (3 - 4)^2} = \sqrt{\frac{1}{3}^2 + 1} = 1.0540...$ |
| 5 | 1 | $\sqrt{(5 - 0.5)^2 + (4 - 1.5)^2} = \sqrt{4.5^2 + 2.5^2} = 5.1478...$ |
| 5 | 2 | $\sqrt{(5 - \frac{14}{3})^2 + (4 - 4)^2} = \sqrt{\frac{2}{3}^2 + 0} = 0.4444...$ |

#### Cluster assignments
No changes to the cluster assignments, therefore the run stops here.
| Cluster | Observations | mean vector |
| - | - | - |
| 1 | 1, 2 | (0.5,1.5)$ |
| 2 | 3, 4, 5 | (\frac{14}{3},4)$ |