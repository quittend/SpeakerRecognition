from matplotlib import pyplot
import bob.learn.em
import bob.measure
import numpy as np
import bob.io.base
from GaussianMixturesModel import GaussianMixtureModel
from Model import Model
from enum import Enum


class IvectorsModel(Model):
    def __init__(self,
                 ubm,
                 tv_dim=400,
                 tv_iterations=5,
                 tv_update_sigma=True,
                 tv_variance_threshold=5e-4,
                 ):

        Model.__init__(self)
        self.ubm = ubm
        self.tv_dim = tv_dim
        self.tv_iterations = tv_iterations
        self.tv_update_sigma = tv_update_sigma
        self.tv_variance_threshold = tv_variance_threshold

    @staticmethod
    def __get_stats(gmm_machine, features):
        stats = []
        for feature in features:
            gmm_stats = bob.learn.em.GMMStats(gmm_machine.shape[0], gmm_machine.shape[1])
            gmm_machine.acc_statistics(feature, gmm_stats)
            stats.append(gmm_stats)
        return stats

    def __get_ivector(self, list_features):
        stats = self.__get_stats(self.ubm, list_features)
        ivectors = [self.model(stat) for stat in stats]
        return np.mean(ivectors, axis=0).tolist()

    def train_model(self, data):
        ivector_machine = bob.learn.em.IVectorMachine(self.ubm, self.tv_dim, self.tv_variance_threshold)
        ivector_trainer = bob.learn.em.IVectorTrainer(update_sigma=self.tv_update_sigma)
        training_ubm_stats = self.__get_stats(self.ubm, data)
        bob.learn.em.train(ivector_trainer, ivector_machine, training_ubm_stats, self.tv_iterations)
        self.model = ivector_machine

    def enroll_speakers(self, data):
        for name, list_features in data.iteritems():
            self.speakers[name] = self.__get_ivector(list_features)

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
    #
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
    #     np.array([[1.4, 2.2, 3.1], [-1, -3, 6], [6, 8, 9]], dtype='float64'),
    #     np.array([[7, -8, 9], [10, -11, 12], [6, 7, 2]], dtype='float64'),
    #     np.array([[13, -14, 15], [16, -17, 18]], dtype='float64')
    # ]
    #
    # data_speakers = {
    #     'user1': [
    #         np.array([[1, 2, 3], [4, 5, 6]], dtype='float64'),
    #         np.array([[7, 8, 9], [10, 11, 12], [6, 7, 2]], dtype='float64'),
    #         np.array([[13, 14, 15], [16, 17, 18]], dtype='float64')
    #     ],
    #     'user2': [
    #         np.array([[-1, -2, -3], [-4, -5, -6]], dtype='float64'),
    #         np.array([[-7, -8, -9], [-10, -11, -12]], dtype='float64'),
    #         np.array([[-13, -14, -15], [-16, -17, -18]], dtype='float64')
    #     ]
    # }
    #
    # data_verification = {
    #     'user1': [
    #         np.array([[1, 2, 3], [4, 5, 6]], dtype='float64')
    #     ],
    #     'user2': [
    #         np.array([[-1, -2, -3], [-4, -5, -6]], dtype='float64'),
    #         np.array([[-7, -8, -9], [-10, -11, -12]], dtype='float64')
    #     ]
    # }

    # from TimitManager import TimitManager
    # from FeaturesManager import FeaturesManager
    #
    # timit = TimitManager.load('/home/mateusz/SpeakerRecognition/timit_manager.pickle')
    # features_path = '/home/mateusz/features1'
    # features = FeaturesManager(features_path)
    # features.load_features(timit)
    #
    # data_ubm = [feature for speaker in features.ubm.values() for feature in speaker]
    # data_speakers = features.adaptation
    # data_verification = features.verification
    #
    # gmm_model = GaussianMixtureModel.load('model')
    # ivector_model = IvectorsModel(gmm_model.ubm, tv_dim=400)
    # ivector_model.train_model(data_ubm)
    # ivector_model.enroll_speakers(data_speakers)
    # ivector_model.calculate_scores(data_verification)
    # ivector_model.plot_det()
    # far, frr = ivector_model.calculate_farfrr()
    # print far, frr
