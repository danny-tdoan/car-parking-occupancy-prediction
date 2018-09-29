import os
import numpy as np
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score, make_scorer, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot

DATA_LOC = 'parking_usage_12P\\';
STARTING_TIME = 31  # 7:30 am = midnight + 15 mins * (31 - 1)
ENDING_TIME = 75  # 6:30 pm = midnight + 15 mins * (75 - 1)


def extract_clusters_for_modelling_has_exact_index(X, cluster_idx):
    clusters = []
    for i, c_idx in enumerate(cluster_idx):
        clusters.append(X[c_idx, :])

    return clusters


def extract_clusters_for_modelling(X, no_of_locs, cluster_locs, cluster_intervals):
    ''' The clusters are defined by locs (rows) and intervals (cols).
        Locs is a list of location list. e.g. if locs is [[1,3,5],[6,7,8]], and there are 120 locs in total
            - The data of cluster 1 consist of rows [1, 3, 5, 121, 123, 125, 241, 243, 245...]
            - The data of cluster 2 consist of rows [6, 7, 8, 126, 127, 128...]
        Fit a model into each cluster
    '''
    no_of_records = X.shape[0]
    no_of_days = no_of_records // no_of_locs

    clusters = []
    for i, cluster_loc in enumerate(cluster_locs):
        all_locs_this_cluster = np.array([])
        for day in range(no_of_days):
            if len(all_locs_this_cluster) == 0:
                all_locs_this_cluster = np.array(cluster_loc) + no_of_locs * day
            else:
                all_locs_this_cluster = np.hstack((all_locs_this_cluster, np.array(cluster_loc) + no_of_locs * day))

        clusters.append(X[all_locs_this_cluster, :])

    return clusters


def init_data(data_input_list):
    parking_data = np.array([])
    for file in open(data_input_list).read().split('\n'):
        if file.endswith('.csv'):
            try:
                # locs x 96 intervals
                data = np.loadtxt(DATA_LOC + file, delimiter=',', dtype=float)
                # data=np.delete(data,DISCARDED_LOCS,axis=0)
                # slice 7:30-19:30
                # data=data*100

                if parking_data.shape[0] == 0:
                    parking_data = data
                else:
                    parking_data = np.vstack((parking_data, data))
            except Exception:
                print('Something wrong with ' + file)
    return parking_data


def fit_regression_tree_model(train_clusters, test_clusters, start, end, prediction_length, m_depth=5, export_tree=True,
                              range_prediction=True, show_sub=False):
    all_models = []
    X_train_all = np.array([])
    Y_train_all = np.array([])

    for train_cluster in train_clusters:
        model = DecisionTreeRegressor(random_state=1, max_depth=m_depth, min_samples_leaf=5)

        X = train_cluster[:, np.linspace(start, end, end - start + 1, dtype=int)]
        if range_prediction:
            Y = train_cluster[:, np.linspace(end + 1, end + prediction_length, prediction_length, dtype=int)]
        else:
            Y = train_cluster[:, end + prediction_length]

        # append the data to use with one model later - for comparison
        if len(X_train_all) == 0:
            X_train_all = X
        else:
            X_train_all = np.vstack((X_train_all, X))

        if len(Y_train_all) == 0:
            Y_train_all = Y
        else:
            if range_prediction:
                Y_train_all = np.vstack((Y_train_all, Y))
            else:
                Y_train_all = np.append(Y_train_all, Y)

        model.fit(X, Y)
        cv_object = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.3, random_state=1)
        # score=cross_val_score(model,X,Y,cv=cv_object,scoring=make_scorer(r2_score,multioutput='uniform_average'))
        score = cross_val_score(model, X, Y, cv=cv_object,
                                scoring=make_scorer(mean_absolute_error, multioutput='uniform_average'))
        print(score)
        all_models.append(model)

    print("Ensemble scores "+str(score.mean()))

    X_values = np.array([])
    predicted_values = np.array([])
    actual_values = np.array([])

    sub_model_scores = []

    for i, test_cluster in enumerate(test_clusters):
        X_test = test_cluster[:, np.linspace(start, end, end - start + 1, dtype=int)]
        if range_prediction:
            Y_actual = test_cluster[:, np.linspace(end + 1, end + prediction_length, prediction_length, dtype=int)]
        else:
            Y_actual = test_cluster[:, end + prediction_length]

        Y_predicted = all_models[i].predict(X_test)

        s = mean_absolute_error(Y_actual, Y_predicted)
        sub_model_scores.append((len(Y_predicted), s))

        if show_sub:
            # plot here to check
            import matplotlib.pyplot as plt
            plt.figure(export_tree + 10000 + i)
            bp_train = plt.boxplot(train_clusters[i][:, np.linspace(start, end, end - start + 1, dtype=int)])
            plt.setp(bp_train['boxes'], color='blue')
            plt.setp(bp_train['whiskers'], color='blue')
            plt.setp(bp_train['fliers'], color='blue')

            bp_test = plt.boxplot(X_test)
            plt.setp(bp_train['boxes'], color='green')
            plt.setp(bp_train['whiskers'], color='green')
            plt.setp(bp_train['fliers'], color='green')

            plt.title(
                'Box plot for sub-model data {} - MAE {:.3f}'.format(i, mean_absolute_error(Y_actual, Y_predicted)))

            plt.figure(export_tree + 20000 + i)
            plt.plot(Y_predicted, label='predicted')
            plt.plot(Y_actual, label='actual')
            plt.title('Sub-model {}'.format(i))

        if len(X_values) == 0:
            X_values = X_test
        else:
            X_values = np.vstack((X_values, X_test))

        if len(predicted_values) == 0:
            predicted_values = Y_predicted
        else:
            if range_prediction:
                predicted_values = np.vstack((predicted_values, Y_predicted))
            else:
                predicted_values = np.append(predicted_values, Y_predicted)

        if len(actual_values) == 0:
            actual_values = Y_actual
        else:
            if range_prediction:
                actual_values = np.vstack((actual_values, Y_actual))
            else:
                actual_values = np.append(actual_values, Y_actual)

        print(r2_score(Y_actual,Y_predicted,multioutput='uniform_average'))

    # test with the entire input, output
    # this is to reconfirm the correctnest of the data slicing
    one_model = DecisionTreeRegressor(random_state=1, max_depth=m_depth)
    one_model.fit(X_train_all, Y_train_all)
    predicted_values_one_model = one_model.predict(X_values)

    cv_object = ShuffleSplit(X_train_all.shape[0], n_iter=10, test_size=0.3, random_state=1)
    scores = cross_val_score(one_model, X_train_all, Y_train_all, cv=cv_object,
                             scoring=make_scorer(mean_absolute_error, multioutput='uniform_average'))
    print(scores)

    one_model_score_1 = mean_absolute_error(actual_values, predicted_values_one_model, multioutput='uniform_average')
    one_model_score_2 = r2_score(actual_values, predicted_values_one_model, multioutput='uniform_average')

    # take care of some admin stuff
    if export_tree:
        # export the one_model tree
        dot_data = StringIO()
        tree.export_graphviz(one_model, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_png("output\\graphviz\\one_model_" + str(start) + "_" + str(end) + ".png")

        for i, model in enumerate(all_models):
            dot_data = StringIO()
            tree.export_graphviz(model, out_file=dot_data)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(
                "output\\graphviz\\ensemble_model_" + str(export_tree) + "_" + str(start) + "_" + str(end) + "_" + str(
                    i) + ".png")

    # final_scores=r2_score(actual_values,predicted_values,multioutput='uniform_average')
    final_score_1 = mean_absolute_error(actual_values, predicted_values, multioutput='uniform_average')
    final_score_2 = r2_score(actual_values, predicted_values, multioutput='uniform_average')

    return (predicted_values, actual_values, (final_score_1, final_score_2), (one_model_score_1, one_model_score_2),
            sub_model_scores)
