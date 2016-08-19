import os
import numpy as np

DATA_LOC='C:\\PhD\\data\\parking\\extract\\parking_usage_12P\\';
DISCARDED_LOCS=np.array([2,6,23,24,29,38,44,46,47,48,56,64,65,66,71,79,81,83,93,109,110,113,115,116,118,129,141,148,149])-1


def ensemble_predict(X,no_of_locs,cluster_locs,cluster_intervals,basic_model):
    ''' The clusters are defined by locs (rows) and intervals (cols).
        Locs is a list of location list. e.g. if locs is [[1,3,5],[6,7,8]], and there are 120 locs in total
            - The data of cluster 1 consist of rows [1, 3, 5, 121, 123, 125, 241, 243, 245...]
            - The data of cluster 2 consist of rows [6, 7, 8, 126, 127, 128...]
        Fit a model into each cluster
    '''
    no_of_records=X.shape[0]
    no_of_days=no_of_records//no_of_locs

    clusters=[]
    for i,cluster_loc in enumerate(cluster_locs):
        all_locs_this_cluster=np.array([])
        for day in range(no_of_days):
            if len(all_locs_this_cluster)==0:
                all_locs_this_cluster=cluster_loc+no_of_locs*day
            else:
                all_locs_this_cluster=np.hstack((all_locs_this_cluster,cluster_loc+no_of_locs*day))

        #clusters.append(X[all_locs_this_cluster,:][:,cluster_intervals[i]])
        clusters.append(X[all_locs_this_cluster,:][:,np.linspace(31,75,num=75-31+1,dtype=int)])

    return clusters

def init_data(data_loc):
    parking_data=np.array([])
    for file in os.listdir(DATA_LOC):
        if file.endswith('.csv'):
            try:
                #locs x 96 intervals
                data=np.loadtxt(DATA_LOC+file,delimiter=',',dtype=float)
                data=np.delete(data,DISCARDED_LOCS,axis=0)
                #slice 7:30-19:30
                #data=data*100

                if parking_data.shape[0]==0:
                    parking_data=data
                else:
                    parking_data=np.vstack((parking_data,data))
            except Exception:
                print('Something wrong with '+file)
    return parking_data


