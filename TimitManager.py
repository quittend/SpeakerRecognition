from random import sample, seed
import os
import cPickle


class TimitManager(object):
    def __init__(self, wav_rootpath, file_ext='.WAV'):
        self.wav_rootpath = wav_rootpath
        self.file_ext = file_ext
        self.ubm = {}
        self.adaptation = {}
        self.verification = {}

    @staticmethod
    def random_split(seed_value, fraction, data):
        # type: (object, object, object) -> object
        n1 = len(data)
        n2 = int(n1 * fraction)
        seed(seed_value)
        idxs1 = sample(range(0, n1), n2)
        idxs2 = [i for i in range(0, n1) if i not in idxs1]
        data1 = [data[idx] for idx in idxs1]
        data2 = [data[idx] for idx in idxs2]
        return data1, data2

    # TODO: add modes = TRAININGTEST/TRAINING/CUSTOM
    def __call__(self, seed, frac_ubm=0.8, frac_adaptation=0.9):
        dialects_paths = [os.path.join(self.wav_rootpath, 'DR%d' % i) for i in range(1, 9)]
        speakers_paths = [os.path.join(dialect_path, speaker)
                          for dialect_path in dialects_paths
                          for speaker in os.listdir(dialect_path)]

        ubm_speakers, adaptation_speakers = self.random_split(seed, frac_ubm, speakers_paths)

        for speaker_path in ubm_speakers:
            __, speaker_name = os.path.split(speaker_path)
            speaker_files = filter(lambda f: f.endswith(self.file_ext), os.listdir(speaker_path))
            self.ubm[speaker_name] = speaker_files

        for speaker_path in adaptation_speakers:
            __, speaker_name = os.path.split(speaker_path)
            speaker_files = filter(lambda f: f.endswith(self.file_ext), os.listdir(speaker_path))
            model_files, verification_files = self.random_split(seed, frac_adaptation, speaker_files)

            self.adaptation[speaker_name] = model_files
            self.verification[speaker_name] = verification_files

    def save(self, filename):
        with open(filename, 'wb') as handle:
            cPickle.dump(self, handle)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as handle:
            return cPickle.load(handle)

if __name__ == "__main__":
    wav_path = '/media/mateusz/Data/Magisterka/Datasets/TIMIT/Raw'
    timit = TimitManager(wav_path, file_ext='.WAV')
    timit(1, 0.8, 0.9)
    timit.save('/home/mateusz/SpeakerRecognition/timit_manager.pickle')
    timit2 = timit.load('/home/mateusz/SpeakerRecognition/timit_manager.pickle')
    print 'x'




