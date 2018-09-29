# Overview

This project uses the clustering-based Ensemble Prediction to predict the car parking occupancy of Melbourne.

The data is available [here](https://data.melbourne.vic.gov.au/Transport-Movement/On-street-Car-Parking-Sensor-Data-2016/dj7e-rdx9)

This program uses the values in a moving window (e.g., 8am-10am) as training data to predict the next 30 mins/1hours/2 hours, etc. Different values of window width are used for training and test data, e.g. use 2 hours to predict the next 1 hour, use 4 hours to predict the next 2 hours, use 2 hours to predict the next 2 hours, etc.

The clustering results are produced by our proposed subspace clustering algorithm which is used to cluster high dimensional data. The details of the subspace clustering algorithm is not presented in this repository.

Two different models are built for comparison:

1. An ensemble prediction model built on results of the proposed subspace clustering algorithm (model 1).
2. An ensemble prediction model built on randomly generated 'clusters' (model 2).

The empirical results should show that model 1 outperforms model 2. Other ensemble prediction models can be built on top of clustering results of other algorithms (k-means, DBSCAN, etc.) for more comparison

# Usage
1. Each submodel is constructed on a clusters (that group parking bays with similar patterns). The sample format of the
clusters are as follows:
    ```
    1     2     3     4     6     
    5     8     9    11    13    15    17
    58    77    93   106   107   108   117   195   198   200   225   243   303
    41    60    80    83    87    90   101  
    ``` 
    Each line corresponds to a clusters. To interprete the above: Cluster 1 consists of parking bays ID `1, 2, 3, 4, 6`
Cluster 2 consists of parking bays ID `5, 8, 9, 11, 13, 15, 17`

2. Each prediction submodel is built on top of each cluster. The ensemble model is cross-validated

3. Predict

Check `regression_parking.py` for an end-to-end example of how to call and evaluate the model. 