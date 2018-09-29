"""This regresssion_parking use the clustering-based Ensemble Prediction to predict the car parking occupancy of Melbourne.
The program uses the values in a moving window (e.g., 8am-10am) as training data to predict the next
30 mins/1hours/2 hours, etc.

Different values of window width are used for training and test data, e.g. use 2 hours to predict the next 1 hour,
use 4 hours to predict the next 2 hours, etc.

Two different models are built for comparison:
    1. An ensemble prediction model built on results of the proposed subspace clustering algorithm
    2. An ensemble prediction model built on randomly generated 'clusters'
"""

from ensemble_prediction import *
import matplotlib.pyplot as plt

INTERVAL_OFFSET = 31

starting_month = 'jun'
parking_data = init_data(DATA_LOC + "list_of_inputs_starting_{}.txt".format(starting_month))

cluster_data = open('clusters/final_cluster_presentation_{}.txt'.format(starting_month)).read().split('\n')
clusters = []
for c in cluster_data:
    tmp_cluster = []
    for item in c.split():
        # 1 is the offset for difference in matlab and python indexing
        tmp_cluster.append(int(item) - 1)

    clusters.append(tmp_cluster)

# TO NOT WASTE ANOTHER 2 HOURS AGAIN: BE VERY VERY CAREFUL WITH THE INDEX. EVEN AN INCORRECT OFFSET OF 1 CAN SCREW UP THE ENTIRE RESULT
# E.G.: TRAIN DATA ARE EXTRACTED ON ONE SET OF LOCATIONS, TEST DATA (TO BE PREDICTED USING THE SAME MODEL) ARE EXTRACTED
# FROM ANOTHER SET
NUM_OF_LOCATION = 149
TRAIN_DATA_START = NUM_OF_LOCATION * 0
TRAIN_DATA_END = NUM_OF_LOCATION * 90 - 1

TEST_DATA_START = TRAIN_DATA_END + 1
TEST_DATA_END = NUM_OF_LOCATION * 100 - 1
START_TIME = 32

TRAIN_WINDOW = 8
RANGE_PREDICTION = False
PREDICT_AHEAD = 2

random_clusters = []
size_of_group = NUM_OF_LOCATION // 7
for i in range(0, NUM_OF_LOCATION - size_of_group - 1, size_of_group):
    random_clusters.append([i + m for m in range(size_of_group)])

# append the remaining locations to the last cluster
for i in range(NUM_OF_LOCATION - size_of_group - 1, NUM_OF_LOCATION):
    random_clusters[-1].append(i)

print(random_clusters)
print(clusters)

cnt = 0

for predicting_hour in [2, 4, 8, 12, 16]:

    one_model_scores = []
    ensemble_scores = []
    random_ensemble_scores = []
    for i in range(16):
        start_time = START_TIME + i
        print()
        print('=================================')
        print(start_time)

        important_hour = list(
            np.linspace(start_time, start_time + 7 + predicting_hour, 7 + predicting_hour + 1, dtype=int))
        important_hour = important_hour[:8] + important_hour[-1:]

        # print("Using 12 observations before "+str((start_time)/4)+" to predict the next 2 hours")
        # print("-----------------------------------------------------")
        train_data = extract_clusters_for_modelling(parking_data[np.linspace(TRAIN_DATA_START, TRAIN_DATA_END,
                                                                             TRAIN_DATA_END - TRAIN_DATA_START + 1,
                                                                             dtype=int), :], NUM_OF_LOCATION, clusters,
                                                    None)
        test_data = extract_clusters_for_modelling(
            parking_data[np.linspace(TEST_DATA_START, TEST_DATA_END, TEST_DATA_END - TEST_DATA_START + 1, dtype=int),
            :], NUM_OF_LOCATION, clusters, None)

        for m_i, c in enumerate(train_data):
            print()
            print("Column stats for cluster", m_i)
            refined_cluster = c[:, important_hour]
            refined_cluster_test = test_data[m_i][:, important_hour]
            for col in range(refined_cluster.shape[1]):
                print("Column ", col)
                print(np.mean(refined_cluster[:, col]), np.std(refined_cluster[:, col]),
                      np.mean(refined_cluster_test[:, col]), np.std(refined_cluster_test[:, col]))

        (predict, actual, ensemble_score, one_model_score) = fit_regression_tree_model(train_data, test_data,
                                                                                       start_time,
                                                                                       start_time + TRAIN_WINDOW - 1,
                                                                                       predicting_hour, m_depth=3,
                                                                                       export_tree=10,
                                                                                       range_prediction=RANGE_PREDICTION)

        one_model_scores.append(one_model_score)
        ensemble_scores.append(ensemble_score)

        # FOR BENCHMARK: build another ensemble prediction model but use randomly-generated clusters
        # Ideally the results should be much worse than the properly-clustered ensemble model
        train_data_random_c = extract_clusters_for_modelling(parking_data[
                                                             np.linspace(TRAIN_DATA_START, TRAIN_DATA_END - 1,
                                                                         TRAIN_DATA_END - TRAIN_DATA_START, dtype=int),
                                                             :], NUM_OF_LOCATION, random_clusters, None)
        test_data_random_c = extract_clusters_for_modelling(
            parking_data[np.linspace(TEST_DATA_START, TEST_DATA_END - 1, TEST_DATA_END - TEST_DATA_START, dtype=int),
            :], NUM_OF_LOCATION, random_clusters, None)
        (predict_random_c, actual_random_c, r2_random_c, r2_one_model_random_c) = fit_regression_tree_model(
            train_data_random_c, test_data_random_c, start_time, start_time + TRAIN_WINDOW - 1, predicting_hour,
            m_depth=3, export_tree=100, range_prediction=RANGE_PREDICTION)
        random_ensemble_scores.append(r2_random_c)

        print("R2 Score one model: {}".format(r2_one_model_random_c))
        print("RANDOM R2 Score ensemble: {}".format(r2_random_c))

    print(one_model_scores)
    print(ensemble_scores)
    print(random_ensemble_scores)

    plt.figure(predicting_hour)

    plt.plot(one_model_scores, label='one_model_r2')
    plt.plot(ensemble_scores, label='ensemble_model_r2')
    plt.plot(random_ensemble_scores, label='random_split_r2')
    plt.legend(loc=3)
    # plt.show()
    plt.savefig('jun_sep_' + str(
        predicting_hour) + '_hours_ahead_MAE.png')
    plt.clf()
