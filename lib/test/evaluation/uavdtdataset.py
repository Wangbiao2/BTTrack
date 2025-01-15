import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVDTDataset(BaseDataset):
    """
    UAVDT test set consisting of 50 videos

    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uavdt_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):

        anno_path = '{}/anno/{}_gt.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/'.format(self.base_path, sequence_name)

        frames_list = ['{}/img{:06d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'uavdt', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['S0101','S0102','S0103','S0201','S0301','S0302','S0303','S0304','S0305','S0306','S0307','S0308','S0309','S0310','S0401','S0402','S0501','S0601','S0602','S0701','S0801','S0901','S1001','S1101','S1201','S1202','S1301','S1302','S1303','S1304','S1305','S1306','S1307','S1308','S1309','S1310','S1311','S1312','S1313','S1401','S1501','S1601','S1602','S1603','S1604','S1605','S1606','S1607','S1701','S1702']
        return sequence_list
