import bob.bio.spear.extractor
import bob.bio.base.preprocessor.Preprocessor
import bob.io.base
from scikits.audiolab import Sndfile
from TimitManager import TimitManager
import os
import cPickle
import numpy as np


class FeaturesManager(object):
    def __init__(self, features_rootpath, file_ext='.HTK'):
        self.features_rootpath = features_rootpath
        self.file_ext = file_ext
        self.ubm = {}
        self.adaptation = {}
        self.verification = {}

    def create_features(self, timit_manager, mode='mfcc'):
        def create_per_dict(speaker_name, speaker_path, dict):
            if speaker_name in dict.keys():
                only_files = [f for f in os.listdir(speaker_path) if os.path.isfile(os.path.join(speaker_path, f))]
                speaker_files = filter(lambda f: f in dict[speaker_name], only_files)
                for speaker_file in speaker_files:
                    path = os.path.join(speaker_path, speaker_file)
                    feature_filename = speaker_file.replace(timit_manager.file_ext, self.file_ext)
                    self.__create_feature(path, speaker_name, feature_filename, mode)

        if not os.path.exists(self.features_rootpath):
            os.mkdir(self.features_rootpath)

        for curr_path, dirs, __ in os.walk(timit_manager.wav_rootpath):
            for speaker_name in dirs:
                speaker_path = os.path.join(curr_path, speaker_name)
                create_per_dict(speaker_name, speaker_path, timit_manager.ubm)
                create_per_dict(speaker_name, speaker_path, timit_manager.adaptation)
                create_per_dict(speaker_name, speaker_path, timit_manager.verification)

    def __create_feature(self, input_path, speaker_name, feature_filename, mode):
        speaker_featurepath = os.path.join(self.features_rootpath, speaker_name)
        if not os.path.exists(speaker_featurepath):
            os.mkdir(speaker_featurepath)

        output_path = os.path.join(speaker_featurepath, feature_filename)
        f = Sndfile(input_path)
        n = f.nframes
        rate = f.samplerate
        data = f.read_frames(n)
        original_data = data * pow(2, 15)

        extractor = bob.bio.spear.extractor.Cepstral(win_length_ms=25,
                                                     n_filters=27,
                                                     n_ceps=13,
                                                     with_energy=False,
                                                     mel_scale=True,
                                                     features_mask=np.arange(0, 39))
        preprocessor = bob.bio.spear.preprocessor.Energy_Thr()
        __, __, labels = preprocessor((rate, original_data))
        feature = extractor([rate, original_data, labels])

        out_file = bob.io.base.HDF5File(output_path, 'w')
        extractor.write_feature(feature, out_file)
        out_file.close()

    def load_features(self, timit_manager):
        def load_per_dict(speaker_name, speaker_path, dict, feature_dict):
            if speaker_name in dict.keys():
                only_files = [f for f in os.listdir(speaker_path) if os.path.isfile(os.path.join(speaker_path, f))]
                speaker_files = filter(lambda f: f.replace(self.file_ext, timit_manager.file_ext) in dict[speaker_name],
                                       only_files)
                for speaker_file in speaker_files:
                    read_path = os.path.join(speaker_path, speaker_file)
                    in_file = bob.io.base.HDF5File(read_path, 'r')
                    feature = bob.bio.base.load(in_file)
                    in_file.close()

                    if speaker_name in feature_dict.keys():
                        feature_dict[speaker_name].append(feature)
                    else:
                        feature_dict[speaker_name] = [feature]

        for curr_path, dirs, __ in os.walk(self.features_rootpath):
            for speaker_name in dirs:
                speaker_path = os.path.join(curr_path, speaker_name)
                load_per_dict(speaker_name, speaker_path, timit_manager.ubm, self.ubm)
                load_per_dict(speaker_name, speaker_path, timit_manager.adaptation, self.adaptation)
                load_per_dict(speaker_name, speaker_path, timit_manager.verification, self.verification)

    # def __call__(self, timit_manager, mode='mfcc'):
    #     self.create_features(timit_manager, mode)
    #     self.load_features(timit_manager, mode)


if __name__ == "__main__":
    timit = TimitManager.load('/home/mateusz/SpeakerRecognition/timit_manager.pickle')
    features_path = '/home/mateusz/features1'
    features = FeaturesManager(features_path)
    features.create_features(timit)
    # features.load_features(timit)
    print 'x'
