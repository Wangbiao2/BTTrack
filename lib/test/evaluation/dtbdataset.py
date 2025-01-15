import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class DTB70Dataset(BaseDataset):
    """
    DTB70 test set consisting of 70 videos.

    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.dtb_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name[:-1] if ('0' <= sequence_name[-1] <= '9') else sequence_name
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'dtb', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['Animal1',
                         'Animal2',
                         'Animal3',
                         'Animal4',
                         'Basketball',
                         'BMX2',
                         'BMX3',
                         'BMX4',
                         'BMX5',
                         'Car2',
                         'Car4',
                         'Car5',
                         'Car6',
                         'Car8',
                         'ChasingDrones',
                         'Girl1',
                         'Girl2',
                         'Gull1',
                         'Gull2',
                         'Horse1',
                         'Horse2',
                         'Kiting',
                         'ManRunning1',
                         'ManRunning2',
                         'Motor1',
                         'Motor2',
                         'MountainBike1',
                         'MountainBike5',
                         'MountainBike6',
                         'Paragliding3',
                         'Paragliding5',
                         'RaceCar',
                         'RaceCar1',
                         'RcCar3',
                         'RcCar4',
                         'RcCar5',
                         'RcCar6',
                         'RcCar7',
                         'RcCar8',
                         'RcCar9',
                         'Sheep1',
                         'Sheep2',
                         'SkateBoarding4',
                         'Skiing1',
                         'Skiing2',
                         'SnowBoarding2',
                         'SnowBoarding4',
                         'SnowBoarding6',
                         'Soccer1',
                         'Soccer2',
                         'SpeedCar2',
                         'SpeedCar4',
                         'StreetBasketball1',
                         'StreetBasketball2',
                         'StreetBasketball3',
                         'SUP2',
                         'SUP4',
                         'SUP5',
                         'Surfing03',
                         'Surfing04',
                         'Surfing06',
                         'Surfing10',
                         'Surfing12',
                         'Vaulting',
                         'Wakeboarding1',
                         'Wakeboarding2',
                         'Walking',
                         'Yacht2',
                         'Yacht4',
                         'Zebra']
        return sequence_list
