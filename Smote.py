import random
import numpy as np


class Smote:
    def __init__(self, k=5):
        """
        This class could use for balancing unbalance data set , it create artificial data from existing data
        :param k:  k of knn
        """
        self.k = k

    def do_smote(self, n, sample_class):
        """ smote method main function
        :param n: number of new sample
        :param sample_class : all minority sample
        :return:
        """

        # for tracking number of generated synthetic samples
        newindex = 0
        synthetic = []
        while n != 0:
            i = random.randint(1, len(sample_class)) - 1
            self.populate(i, self.get_nearest_neighbor(sample_class[i], self.k, sample_class), self.k, synthetic,
                          newindex,
                          sample_class)
            n = n - 1

        return synthetic

    def populate(self, i, nn_array, k, synthetic, new_index, sample_class):
        """
        :param i: index of current sample
        :param nn_array: index of items that got from nearest neighbor method
        :param k : k of knn
        :param synthetic : list of synthetic objects
        :param new_index : number of generated object
        :param sample_class : all minority samples
        :return:
        """

        # while n != 0:
        nn = random.randint(1, k) - 1
        temp = []
        for feature_position in range(0, len(sample_class[0])):
            dif = sample_class[nn_array[nn]][feature_position] - sample_class[i][feature_position]
            gap = random.random()
            temp.insert(feature_position, sample_class[i][feature_position] + gap * dif)

        synthetic.insert(new_index, temp)

        new_index += 1
        # n = n - 1

        return

    def get_nearest_neighbor(self, x_test, k, sample_class):
        """ calculate k nearest neighbor and return their indices
        :param x_test: the current data we want to find its neighbor
        :param k: k of knn
        :param sample_class : all minority samples
        :return: k nearest neighbor indices
        """
        # create list for distances and targets
        distances = []
        targets_index = []

        for i in range(len(sample_class)):
            if sample_class[i][:] != x_test:
                # first we compute the euclidean distance
                distance = np.sqrt(np.sum(np.square(np.array(x_test) - np.array(sample_class[i][:]))))
                # add it to list of distances
                distances.append([distance, i])

        # sort the list
        distances = sorted(distances)

        # make a list of the k neighbors' targets
        for i in range(k):
            targets_index.append(distances[i][1])

        # return most common target
        return targets_index
