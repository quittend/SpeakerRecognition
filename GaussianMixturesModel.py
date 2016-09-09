import bob.learn.em
import bob.measure
import numpy as np
import bob.io.base
import logging
from timeit import default_timer as timer
from Model import Model

logging.basicConfig(filename='gmm.log', level=logging.INFO)


class GaussianMixtureModel(Model):
    """
    A class that contains GMM-UBM model, adapted GMM speakers' models, procedures for training universal model and
    enrolling speaker and measuring results as well. It can be saved or loaded anytime.
    """

    def __init__(self,
                 n_gaussians,
                 kmeans_iterations=10,
                 gmm_iterations=10,
                 gmm_enrollment_iterations=10,
                 training_threshold=5e-4,
                 variance_threshold=5e-4,
                 update_weights=True,
                 update_means=True,
                 update_variances=True,
                 relevance_factor=4):

        Model.__init__(self)
        self.gaussians = n_gaussians
        self.kmeans_iterations = kmeans_iterations
        self.gmm_iterations = gmm_iterations
        self.training_threshold = training_threshold
        self.variance_threshold = variance_threshold
        self.update_weights = update_weights
        self.update_means = update_means
        self.update_variances = update_variances
        self.relevance_factor = relevance_factor
        self.gmm_enrollment_iterations = gmm_enrollment_iterations

    def train_model(self, data):
        stack_data = np.vstack(data)
        dim = stack_data.shape[1]

        kmeans_machine = bob.learn.em.KMeansMachine(self.gaussians, dim)
        kmeans_trainer = bob.learn.em.KMeansTrainer()

        start = timer()
        logging.info('Started training K-means machine.')
        bob.learn.em.train(kmeans_trainer, kmeans_machine, stack_data,
                           max_iterations=self.kmeans_iterations,
                           convergence_threshold=self.training_threshold)
        logging.info('Stopped training K-means machine. Running for %d seconds.', timer() - start)

        variances, weights = kmeans_machine.get_variances_and_weights_for_each_cluster(stack_data)
        gmm_machine = bob.learn.em.GMMMachine(self.gaussians, dim)
        gmm_trainer = bob.learn.em.ML_GMMTrainer(self.update_means, self.update_variances, self.update_weights)
        gmm_machine.means = kmeans_machine.means
        gmm_machine.variances = variances
        gmm_machine.weights = weights
        gmm_machine.set_variance_thresholds(self.variance_threshold)

        start = timer()
        logging.info('Started training GMM-UBM machine.')
        bob.learn.em.train(gmm_trainer, gmm_machine, stack_data,
                           max_iterations=self.gmm_iterations,
                           convergence_threshold=self.training_threshold)
        logging.info('Stopped training GMM-UBM. Running for %d seconds.', timer() - start)
        self.model = gmm_machine

    def enroll_speakers(self, data):
        i = 1
        n = len(data)
        start = timer()
        logging.info('Started training speakers models.')
        for name, list_features in data.iteritems():
            logging.info('Training %d/%d speaker model.' % (i, n))
            features = np.vstack(list_features)
            gmm_adapted = bob.learn.em.GMMMachine(self.model)
            gmm_trainer = bob.learn.em.MAP_GMMTrainer(self.model,
                                                      relevance_factor=self.relevance_factor,
                                                      update_variances=False,
                                                      update_weights=False)

            bob.learn.em.train(gmm_trainer, gmm_adapted, features,
                               max_iterations=self.gmm_enrollment_iterations,
                               convergence_threshold=self.training_threshold)

            self.speakers[name] = gmm_adapted
            i += 1
        logging.info('Stopped training speaker models. Running for %d seconds.', timer() - start)

    def get_scores(self, features):
        scores = {}
        for name, model in self.speakers.iteritems():
            score = np.mean(
                np.apply_along_axis(
                    lambda row: model.log_likelihood(row) - self.model.log_likelihood(row), 1, features
                ))
            scores[name] = score
        return scores

    def save(self, filename):
        f = bob.io.base.HDF5File(filename, "w")
        f.set_attribute('gaussians', self.gaussians)
        f.set_attribute('kmeans_iterations', self.kmeans_iterations)
        f.set_attribute('gmm_iterations', self.gmm_iterations)
        f.set_attribute('training_threshold', self.training_threshold)
        f.set_attribute('variance_threshold', self.variance_threshold)
        f.set_attribute('update_weights', self.update_weights)
        f.set_attribute('update_means', self.update_means)
        f.set_attribute('update_variances', self.update_variances)
        f.set_attribute('relevance_factor', self.relevance_factor)
        f.set_attribute('gmm_enrollment_iterations', self.gmm_enrollment_iterations)
        f.set_attribute('speakers', ";".join(self.speakers.keys()))

        f.create_group('ubm')
        f.cd('ubm')
        bob.io.base.save(self.model, f)
        self.model.save(f)
        f.cd('/')

        f.create_group('speakers')
        f.cd('speakers')
        for name, model in self.speakers.iteritems():
            f.create_group(name)
            f.cd(name)
            model.save(f)
            f.cd('../')
        f.cd('/')

        f.create_group('scores')
        f.set('/scores/positives', np.array(self.positives))
        f.set('/scores/negatives', np.array(self.negatives))
        f.close()

    @staticmethod
    def load(filename):
        f = bob.io.base.HDF5File(filename, "r")
        temp_model = GaussianMixtureModel(f.get_attribute(name='gaussians'),
                                          f.get_attribute(name='kmeans_iterations'),
                                          f.get_attribute(name='gmm_iterations'),
                                          f.get_attribute(name='gmm_enrollment_iterations'),
                                          f.get_attribute(name='training_threshold'),
                                          f.get_attribute(name='variance_threshold'),
                                          f.get_attribute(name='update_weights'),
                                          f.get_attribute(name='update_means'),
                                          f.get_attribute(name='update_variances'),
                                          f.get_attribute(name='relevance_factor')
                                          )
        f.cd('ubm')
        temp_model.ubm = bob.learn.em.GMMMachine(f)
        f.cd('../')

        speakers = f.get_attribute(name='speakers').split(';')
        f.cd('speakers')
        for speaker in speakers:
            f.cd(speaker)
            temp_model.speakers[speaker] = bob.learn.em.GMMMachine(f)
            f.cd('../')
        f.cd('../')

        temp_model.positives = f.get('/scores/positives')
        temp_model.negatives = f.get('/scores/negatives')

        return temp_model


if __name__ == "__main__":
    data_ubm = [
        np.array([[1.4,2.2,3.1], [-1,-3,6], [6,8,9]], dtype='float64'),
        np.array([[7,-8,9], [10,-11,12], [6,7,2]], dtype='float64'),
        np.array([[13,-14,15], [16,-17,18]], dtype='float64')
    ]

    data_speakers = {
        'user1': [
            np.array([[1,2,3], [4,5,6]], dtype='float64'),
            np.array([[7,8,9], [10,11,12], [6,7,2]], dtype='float64'),
            np.array([[13,14,15], [16,17,18]], dtype='float64')
        ],
        'user2': [
            np.array([[-1,-2,-3], [-4,-5,-6]], dtype='float64'),
            np.array([[-7,-8,-9], [-10,-11,-12]], dtype='float64'),
            np.array([[-13,-14,-15], [-16,-17,-18]], dtype='float64')
        ]
    }

    data_verification = {
        'user1': [
            np.array([[1,2,3], [4,5,6]], dtype='float64')
        ],
        'user2': [
            np.array([[-1,-2,-3], [-4,-5,-6]], dtype='float64'),
            np.array([[-7,-8,-9], [-10,-11,-12]], dtype='float64')
        ]
    }

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    gmm_model = GaussianMixtureModel(2)
    gmm_model.train_model(data_ubm)
    gmm_model.enroll_speakers(data_speakers)
    gmm_model.calculate_scores(data_verification)
    gmm_model.save('test.hdf5')
    new_model = GaussianMixtureModel.load('test.hdf5')
    # -----------------------------------------------------------

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
    # gmm_model = GaussianMixtureModel(256, kmeans_iterations=10, gmm_iterations=10, gmm_enrollment_iterations=10)
    # gmm_model.train_ubm(data_ubm)
    # gmm_model.enroll_speakers(data_speakers)
    # gmm_model.calculate_scores(data_verification)
    # gmm_model.save('model')

    # -----------------------------------------------------------
    # gmm_model = GaussianMixtureModel.load('model')
    # gmm_model.plot_det()
    # far, frr = gmm_model.calculate_farfrr()
    # print far, frr

