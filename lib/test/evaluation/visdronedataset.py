import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class VisDroneDataset(BaseDataset):
    """
    VisDrone test set consisting of 35 videos

    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.visdrone_path
        self.sequence_list = self._get_sequence_list()


    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/annotations/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/sequences/{}/'.format(self.base_path, sequence_name)

        frames_list = ['{}/img{:07d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'visdrone', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['uav0000011_00000_s',
                         'uav0000021_00000_s',
                         'uav0000069_00576_s',
                         'uav0000074_01656_s',
                         'uav0000074_04320_s',
                         'uav0000074_04992_s',
                         'uav0000074_05712_s',
                         'uav0000074_06312_s',
                         'uav0000074_11915_s',
                         'uav0000079_02568_s',
                         'uav0000088_00000_s',
                         'uav0000093_00000_s',
                         'uav0000093_01817_s',
                         'uav0000116_00503_s',
                         'uav0000151_00000_s',
                         'uav0000155_01201_s',
                         'uav0000164_00000_s',
                         'uav0000180_00050_s',
                         'uav0000184_00625_s',
                         'uav0000207_00675_s',
                         'uav0000208_00000_s',
                         'uav0000241_00001_s',
                         'uav0000242_02327_s',
                         'uav0000242_05160_s',
                         'uav0000294_00000_s',
                         'uav0000294_00069_s',
                         'uav0000294_01449_s',
                         'uav0000324_00069_s',
                         'uav0000340_01356_s',
                         'uav0000353_00001_s',
                         'uav0000353_01127_s',
                         'uav0000367_02761_s',
                         'uav0000367_04137_s',
                         'uav0000368_03312_s',
                         'uav0000368_03612_s']
        return sequence_list
