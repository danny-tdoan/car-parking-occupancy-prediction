from EnsemblePrediction import *

INTERVAL_OFFSET=31

parking_data=init_data(DATA_LOC)
cluster_loc1=np.array([1,2,3,5])
cluster_loc2=np.array([4,7])

#FIRST STEP: just use all the intervals
# cluster_int1=np.array([1,2,3,4,5,6])+INTERVAL_OFFSET
# cluster_int2=np.array([9,10])+INTERVAL_OFFSET


clusters=ensemble_predict(parking_data,120,[cluster_loc1,cluster_loc2],[cluster_int1,cluster_int2],None)