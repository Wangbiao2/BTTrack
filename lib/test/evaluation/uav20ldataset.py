import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAV20LDataset(BaseDataset):
    """
    UAV20L test set consisting of 20 videos.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx

    - base_path
        - anno
        - bike1
        - bird1
        - car1
        - ...
        - uav1
    20250107 wangbiao301@buaa.edu.cn
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav20l_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name[:-1]

        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'uav20l', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['bike1',
                         'bird1',
                         'car1',
                         'car3',
                         'car6',
                         'car8',
                         'car9',
                         'car16',
                         'group1',
                         'group2',
                         'group3',
                         'person2',
                         'person4',
                         'person5',
                         'person7',
                         'person14',
                         'person17',
                         'person19',
                         'person20',
                         'uav1']
        return sequence_list
