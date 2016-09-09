from abc import ABCMeta, abstractmethod
import logging
from timeit import default_timer as timer
import bob.measure
from matplotlib import pyplot


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None
        self.speakers = {}
        self.negatives = []
        self.positives = []

    @abstractmethod
    def train_model(self, data):
        """
        A method that creates an GMM-UBM model from data. UBM model is stored as an object's field.

        :return: It returns nothing since the result is saved into the internal structure self.ubm.
        :param data: A list of features, where each feature is a 2-D numpy array (cols describe feature-dimensions).
        """
        pass

    @abstractmethod
    def enroll_speakers(self, data):
        """
        A method that creates a GMM speaker model from data. Model is created using MAP adaptation from a GMM-UBM
        model trained previously.

        :param data: A dictionary with key representing a speaker name and value a list of features, where each feature is a 2-D numpy array (cols describe feature-dimensions).
        :return: It returns nothing since the result is saved into the internal structure self.speakers.
        """
        pass

    @abstractmethod
    def get_scores(self, features):
        """
        A method gets all scores when comparing feature with speaker models.

        :param features: 2-D numpy array (cols describe feature-dimensions).
        :returns: It returns a dict with key representing speaker and a value a score value
        """
        pass

    @abstractmethod
    def save(self, filename):
        """
        A method saves a class to HDF5 file with models and calculated results.
        """
        pass

    @staticmethod
    @abstractmethod
    def load(filename):
        """
        A method loads a class object and calculated results from to HDF5 file.

        :return Returns a GaussianMixturesModel object.
        """
        pass

    def calculate_scores(self, data):
        """
        A method calculates all scores during the verification process and shuffle them into positives and
        negatives ones.

        :param data: A dictionary with key representing a speaker name and value a list of features, where each feature is a 2-D numpy array (cols describe feature-dimensions).
        :return: It returns nothing since the result is saved into the internal structure self.positives and self.negatives.
        """
        start = timer()
        logging.info('Started calculating scores for a verification data.')
        self.positives = []
        self.negatives = []
        for verification_name, features_list in data.iteritems():
            for features in features_list:
                scores = self.get_scores(features)
                for speaker_name, score in scores.iteritems():
                    if speaker_name == verification_name:
                        self.positives.append(score)
                    else:
                        self.negatives.append(score)
        logging.info('Stopped calculating scores for a verification data. Running for %d seconds.', timer() - start)

    def calculate_farfrr(self):
        """
        A method calculates FAR and FRR for an EER threshold.

        :return It returns FAR and FRR respectively.
        """

        threshold = bob.measure.eer_threshold(self.negatives, self.positives)
        eer = bob.measure.farfrr(self.negatives, self.positives, threshold)
        return eer

    def plot_det(self, label='Test', color=(0, 0, 0)):
        """
        A method plots a DET diagram for a full verification process.

        :param label: Determines the label as a string
        :param color: Determines the color line in a following format (0, 0, 0).
        """
        n_points = len(self.speakers)
        bob.measure.plot.det(self.negatives, self.positives, n_points,
                             color=color, linestyle='-', label=label)
        bob.measure.plot.det_axis([0.01, 40, 0.01, 40])
        pyplot.xlabel('FAR (%)')
        pyplot.ylabel('FRR (%)')
        pyplot.grid(True)
        pyplot.show()
