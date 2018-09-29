import os
import math
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score,make_scorer
from sklearn.tree import DecisionTreeRegressor

#load the parking occupancy data
DATA_LOC='C:\\PhD\\data\\parking\\extract\\parking_usage_12P\\';
DISCARDED_LOCS=np.array([2,6,23,24,29,38,44,46,47,48,56,64,65,66,71,79,81,83,93,109,110,113,115,116,118,129,141,148,149])-1

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

parking_data

#use WINDOW_WIDTH lookbacks to predict the next occupancy
WINDOW_WIDTH=8
NO_OF_LOCS=120


#test with one  8pm-11pm to predict 11.15, 11.30
#input_hours=32
input_hours=32
for input_hours in range(32,33):
    predicted_hour_1=input_hours+WINDOW_WIDTH+1
    predicted_hour_2=input_hours+WINDOW_WIDTH+2
    predicted_hour_3=input_hours+WINDOW_WIDTH+3
    predicted_hour_4=input_hours+WINDOW_WIDTH+4
    predicted_hour_5=input_hours+WINDOW_WIDTH+5
    predicted_hour_6=input_hours+WINDOW_WIDTH+6
    predicted_hour_7=input_hours+WINDOW_WIDTH+7
    predicted_hour_8=input_hours+WINDOW_WIDTH+8


    observations=np.linspace(0,NO_OF_LOCS*30-1,NO_OF_LOCS*30,dtype=int)
    observations_to_predict=np.linspace(NO_OF_LOCS*30-1,parking_data.shape[0]-1,num=parking_data.shape[0]-NO_OF_LOCS*30+1,dtype=int)


    X=parking_data[:,np.linspace(input_hours,input_hours+WINDOW_WIDTH,WINDOW_WIDTH,dtype=int)][observations,:]

    Y1=parking_data[observations,predicted_hour_1]
    Y2=parking_data[observations,predicted_hour_2]
    Y3=parking_data[observations,predicted_hour_3]
    Y4=parking_data[observations,predicted_hour_4]
    Y5=parking_data[observations,predicted_hour_5]
    Y6=parking_data[observations,predicted_hour_6]
    Y7=parking_data[observations,predicted_hour_7]
    Y8=parking_data[observations,predicted_hour_8]

    regressor = DecisionTreeRegressor(random_state=0)
    res=[]
    # res.append(cross_val_score(regressor, X, Y1, cv=10, scoring='r2'))
    # res.append(cross_val_score(regressor, X, Y2, cv=10, scoring='r2'))
    # res.append(cross_val_score(regressor, X, Y3, cv=10, scoring='r2'))
    # res.append(cross_val_score(regressor, X, Y4, cv=10, scoring='r2'))
    # res.append(cross_val_score(regressor, X, Y5, cv=10, scoring='r2'))
    # res.append(cross_val_score(regressor, X, Y6, cv=10, scoring='r2'))
    # res.append(cross_val_score(regressor, X, Y7, cv=10, scoring='r2'))
    # res.append(cross_val_score(regressor, X, Y8, cv=10, scoring='r2'))

    #res=cross_val_score(regressor,X,np.transpose(np.vstack((Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8))),cv=10,scoring=r2_score(multioutput='uniform_average'))
    res=cross_val_score(regressor,X,np.transpose(np.vstack((Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8))),cv=10,scoring=make_scorer(r2_score,multioutput='uniform_average'))

    #print([np.mean(res[i]) for i in range(8)])
    print(res)


#train the model
regressor.fit(X,np.transpose(np.vstack((Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8))))

#data for prediction
X_test=parking_data[:,np.linspace(input_hours,input_hours+WINDOW_WIDTH,WINDOW_WIDTH,dtype=int)][observations_to_predict,:]

Y_predicted=regressor.predict(X_test)

#verify the result
Y_actual=parking_data[observations_to_predict,:][:,np.linspace(predicted_hour_1,predicted_hour_1+7,num=8,dtype=int)]

r2_score(Y_actual,Y_predicted,multioutput='uniform_average')