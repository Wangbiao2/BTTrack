import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVTrack112Dataset(BaseDataset):
    """UAVTrack112 test set consisting of 112 videos 

    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uavtrack112_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/data_seq/{}/'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'uavtrack112', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['human2','air conditioning box2','courier2','human1','car3','basketball player1','sand truck-n','car7_1','car18','car9_1','tree','football player1_3','football player2_2','pot bunker','car15','hiker2','car4','football player1_2','football player2_1','tennis player1_1','basketball player2','dark car1-n','football player1_1','uav5','uav2','car17','bike2','car9_2','bike7_1','bike5','group3_2','car8','car6_2','duck1_2','bike9_1','car6_1','car1','duck3','basketball player2-n','duck2','container','car7_2','tricycle1_1','courier1','truck','uav3_1','motor2','electric box','jogging1','car12','building1_1','building1_2','island','tricycle1_2','runner1','group1','bike1','truck_night','bike6','runner2','uav1','car7_3','excavator','hiker1','car5','bike9_2','human4','bike7_2','basketball player1_2-n','car16_1','bike4_1','uav4','basketball player4-n','uav3_2','tower crane','motor1','bike3','group3_3','duck1_1','car16_3','car11','car16_2','bus2-n','basketball player1_1-n','car1-n','basketball player3','bike4_2','group3_1','group4','group4_2','air conditioning box1','bell tower','parterre1','parterre2','human5','human','car10','jogging2','group4_1','dark car2-n','human3','swan','bus1-n','tennis player1_2','bike8','car14','group2','car2-n','couple','basketball player3-n','car13','car2']
        return sequence_list
