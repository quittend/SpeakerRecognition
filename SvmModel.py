from matplotlib import pyplot
import bob.learn.libsvm
import bob.measure
import numpy as np
import os
import bob.io.base
from TimitManager import TimitManager
from FeaturesManager import FeaturesManager
import logging
from sklearn import preprocessing


class SvmModel(object):
    def __init__(self,
                 subset_of_ubm=1.0):
        self.subset_of_ubm = subset_of_ubm
        self.ubm = []
        self.speakers = {}
        self.negatives_scores = []
        self.positives_scores = []

    # data should be a list of features, where each feature is an array i.e 237x60
    def train_ubm(self, data):
        print 'Training UBM'
        subset, __ = TimitManager.random_split(1, self.subset_of_ubm, data)
        stack_scaled_data = preprocessing.scale(np.vstack(subset))
        self.ubm = stack_scaled_data

    # data should be a dictionary, where value is a list of features, where each feature is an array i.e 237x60
    def adapt_speakers(self, data):
        i = 1
        n = len(data)
        for name, list_features in data.iteritems():

            print 'Learning %d/%d model\r' % (i, n)
            features_stacked_scaled = preprocessing.scale(np.vstack(list_features))
            svm_trainer = bob.learn.libsvm.Trainer(probability=True)
            train_data = [features_stacked_scaled, self.ubm]
            svm_machine = svm_trainer.train(train_data)
            self.speakers[name] = svm_machine
            i += 1

    # data should be a dictionary, where value is a list of features, where each feature is an array i.e 237x60 (created from file)
    def calculate_scores(self, data):
        positives = []
        negatives = []
        for verification_name, features_list in data.iteritems():
            for model_name, model in self.speakers.iteritems():
                for features in features_list:
                    features_scaled = preprocessing.scale(features)
                    __, probability = model.predict_class_and_probabilities(features_scaled)
                    score = np.mean(probability, axis=0)[0]
                    if verification_name == model_name:
                        positives.append(score)
                    else:
                        negatives.append(score)

        self.positives_scores = positives
        self.negatives_scores = negatives

    def calculate_eer(self):
        T = bob.measure.eer_threshold(self.negatives_scores, self.positives_scores)
        FAR, FRR = bob.measure.farfrr(self.negatives_scores, self.positives_scores, T)
        print 'FAR = %.3f\nFRR = %.3f' % (FAR, FRR)

    def plot_det(self, label='Test', color=(0, 0, 0)):
        n_points = len(self.speakers)
        bob.measure.plot.det(self.negatives_scores, self.positives_scores, n_points,
                             color=color, linestyle='-', label=label)
        bob.measure.plot.det_axis([0.01, 100, 0.01, 100])
        pyplot.xlabel('FAR (%)')
        pyplot.ylabel('FRR (%)')
        pyplot.grid(True)
        pyplot.show()

    # def save(self, dirname):
    #     if not os.path.exists(dirname):
    #         os.mkdir(dirname)
    #
    #     path = os.path.join(dirname, 'ubm.hdf5')
    #     out_file = bob.io.base.HDF5File(path, 'w')
    #     self.ubm.save(out_file)
    #     out_file.close()
    #
    #     for name, model in self.speakers.iteritems():
    #         path = os.path.join(dirname, '%s.hdf5' % name)
    #         out_file = bob.io.base.HDF5File(path, 'w')
    #         model.save(out_file)
    #         out_file.close()
    #
    #     path = os.path.join(dirname, 'cfg.hdf5')
    #     out_file = bob.io.base.HDF5File(path, 'w')
    #     out_file.set('gaussians', self.gaussians)
    #     out_file.set('kmeans_iterations', self.kmeans_iterations)
    #     out_file.set('gmm_iterations', self.gmm_iterations)
    #     out_file.set('training_threshold', self.training_threshold)
    #     out_file.set('variance_threshold', self.variance_threshold)
    #     out_file.set('update_weights', self.update_weights)
    #     out_file.set('update_means', self.update_means)
    #     out_file.set('update_variances', self.update_variances)
    #     out_file.set('relevance_factor', self.relevance_factor)
    #     out_file.set('gmm_enrollment_iterations', self.gmm_enrollment_iterations)
    #     out_file.set('speakers', ";".join(self.speakers.keys()))
    #     out_file.close()

    # @staticmethod
    # def load(dirname):
    #     path = os.path.join(dirname, 'cfg.hdf5')
    #     in_file = bob.io.base.HDF5File(path, 'r')
    #     temp_model = GaussianMixtureModel(in_file.read('gaussians'),
    #                                       in_file.read('kmeans_iterations'),
    #                                       in_file.read('gmm_iterations'),
    #                                       in_file.read('gmm_enrollment_iterations'),
    #                                       in_file.read('training_threshold'),
    #                                       in_file.read('variance_threshold'),
    #                                       in_file.read('update_weights'),
    #                                       in_file.read('update_means'),
    #                                       in_file.read('update_variances'),
    #                                       in_file.read('relevance_factor')
    #                                       )
    #     speakers = in_file.read('speakers').split(';')
    #     in_file.close()
    #
    #     for speaker in speakers:
    #         path = os.path.join(dirname, '%s.hdf5' % speaker)
    #         in_file = bob.io.base.HDF5File(path, 'r')
    #         temp_model.speakers[speaker] = bob.learn.em.GMMMachine(in_file)
    #         in_file.close()
    #
    #     path = os.path.join(dirname, 'ubm.hdf5')
    #     in_file = bob.io.base.HDF5File(path, 'r')
    #     temp_model.ubm = bob.learn.em.GMMMachine(in_file)
    #     in_file.close()
    #
    #     return temp_model


if __name__ == "__main__":
    # data_ubm = [
    #     np.array([[1.4,2.2,3.1], [-1,-3,6]], dtype='float64'),
    #     np.array([[7,-8,9], [10,-11,12]], dtype='float64'),
    #     np.array([[13,-14,15], [16,-17,18]], dtype='float64')
    # ]
    #
    # data_speakers = {
    #     'user1': [
    #         np.array([[1,2,3], [4,5,6]], dtype='float64'),
    #         np.array([[7,8,9], [10,11,12]], dtype='float64'),
    #         np.array([[13,14,15], [16,17,18]], dtype='float64')
    #     ],
    #     'user2': [
    #         np.array([[-1,-2,-3], [-4,-5,-6]], dtype='float64'),
    #         np.array([[-7,-8,-9], [-10,-11,-12]], dtype='float64'),
    #         np.array([[-13,-14,-15], [-16,-17,-18]], dtype='float64')
    #     ]
    # }
    #
    # data_verification = {
    #     'user1': [
    #         np.array([[1,2,3], [4,5,6]], dtype='float64')
    #     ],
    #     'user2': [
    #         np.array([[-1,-2,-3], [-4,-5,-6]], dtype='float64'),
    #         np.array([[-7,-8,-9], [-10,-11,-12]], dtype='float64')
    #     ]
    # }
    #
    # svm_model = SvmModel()
    # svm_model.train_ubm(data_ubm)
    # svm_model.adapt_speakers(data_speakers)
    # svm_model.calculate_scores(data_verification)
    # svm_model.plot_det()


    from timeit import default_timer as timer
    start = timer()

    timit = TimitManager.load('/home/mateusz/SpeakerRecognition/timit_manager.pickle')
    features_path = '/home/mateusz/features1'
    features = FeaturesManager(features_path)
    features.load_features(timit)

    data_ubm2 = [feature for speaker in features.ubm.values() for feature in speaker]
    data_speakers2 = features.adaptation
    data_verification2 = features.verification

    svm_model = SvmModel(0.3)
    svm_model.train_ubm(data_ubm2)
    svm_model.adapt_speakers(data_speakers2)
    svm_model.calculate_scores(data_verification2)
    svm_model.plot_det()

    elapsed_time = timer() - start  # in seconds
    print elapsed_time
