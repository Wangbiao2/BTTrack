import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAV10fpsDataset(BaseDataset):
    """ UAV123 dataset.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav10fps_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav10fps', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "uav10fps_bike1", "path": "data_seq/UAV123_10fps/bike1", "startFrame": 1, "endFrame": 1029, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/bike1.txt", "object_class": "vehicle"},
            {"name": "uav10fps_bike2", "path": "data_seq/UAV123_10fps/bike2", "startFrame": 1, "endFrame": 185, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/bike2.txt", "object_class": "vehicle"},
            {"name": "uav10fps_bike3", "path": "data_seq/UAV123_10fps/bike3", "startFrame": 1, "endFrame": 145, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/bike3.txt", "object_class": "vehicle"},
            {"name": "uav10fps_bird1_1", "path": "data_seq/UAV123_10fps/bird1", "startFrame": 1, "endFrame": 85, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/bird1_1.txt", "object_class": "bird"},
            {"name": "uav10fps_bird1_2", "path": "data_seq/UAV123_10fps/bird1", "startFrame": 259, "endFrame": 493, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/bird1_2.txt", "object_class": "bird"},
            {"name": "uav10fps_bird1_3", "path": "data_seq/UAV123_10fps/bird1", "startFrame": 525, "endFrame": 813, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/bird1_3.txt", "object_class": "bird"},
            {"name": "uav10fps_boat1", "path": "data_seq/UAV123_10fps/boat1", "startFrame": 1, "endFrame": 301, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat1.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat2", "path": "data_seq/UAV123_10fps/boat2", "startFrame": 1, "endFrame": 267, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat2.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat3", "path": "data_seq/UAV123_10fps/boat3", "startFrame": 1, "endFrame": 301, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat3.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat4", "path": "data_seq/UAV123_10fps/boat4", "startFrame": 1, "endFrame": 185, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat4.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat5", "path": "data_seq/UAV123_10fps/boat5", "startFrame": 1, "endFrame": 169, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat5.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat6", "path": "data_seq/UAV123_10fps/boat6", "startFrame": 1, "endFrame": 269, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat6.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat7", "path": "data_seq/UAV123_10fps/boat7", "startFrame": 1, "endFrame": 179, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat7.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat8", "path": "data_seq/UAV123_10fps/boat8", "startFrame": 1, "endFrame": 229, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat8.txt", "object_class": "vessel"},
            {"name": "uav10fps_boat9", "path": "data_seq/UAV123_10fps/boat9", "startFrame": 1, "endFrame": 467, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/boat9.txt", "object_class": "vessel"},
            {"name": "uav10fps_building1", "path": "data_seq/UAV123_10fps/building1", "startFrame": 1, "endFrame": 157, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/building1.txt", "object_class": "other"},
            {"name": "uav10fps_building2", "path": "data_seq/UAV123_10fps/building2", "startFrame": 1, "endFrame": 193, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/building2.txt", "object_class": "other"},
            {"name": "uav10fps_building3", "path": "data_seq/UAV123_10fps/building3", "startFrame": 1, "endFrame": 277, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/building3.txt", "object_class": "other"},
            {"name": "uav10fps_building4", "path": "data_seq/UAV123_10fps/building4", "startFrame": 1, "endFrame": 263, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/building4.txt", "object_class": "other"},
            {"name": "uav10fps_building5", "path": "data_seq/UAV123_10fps/building5", "startFrame": 1, "endFrame": 161, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/building5.txt", "object_class": "other"},
            {"name": "uav10fps_car1_1", "path": "data_seq/UAV123_10fps/car1", "startFrame": 1, "endFrame": 251, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car1_1.txt", "object_class": "car"},
            {"name": "uav10fps_car1_2", "path": "data_seq/UAV123_10fps/car1", "startFrame": 251, "endFrame": 543, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car1_2.txt", "object_class": "car"},
            {"name": "uav10fps_car1_3", "path": "data_seq/UAV123_10fps/car1", "startFrame": 543, "endFrame": 877, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car1_3.txt", "object_class": "car"},
            {"name": "uav10fps_car10", "path": "data_seq/UAV123_10fps/car10", "startFrame": 1, "endFrame": 469, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car10.txt", "object_class": "car"},
            {"name": "uav10fps_car11", "path": "data_seq/UAV123_10fps/car11", "startFrame": 1, "endFrame": 113, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car11.txt", "object_class": "car"},
            {"name": "uav10fps_car12", "path": "data_seq/UAV123_10fps/car12", "startFrame": 1, "endFrame": 167, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car12.txt", "object_class": "car"},
            {"name": "uav10fps_car13", "path": "data_seq/UAV123_10fps/car13", "startFrame": 1, "endFrame": 139, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car13.txt", "object_class": "car"},
            {"name": "uav10fps_car14", "path": "data_seq/UAV123_10fps/car14", "startFrame": 1, "endFrame": 443, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car14.txt", "object_class": "car"},
            {"name": "uav10fps_car15", "path": "data_seq/UAV123_10fps/car15", "startFrame": 1, "endFrame": 157, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car15.txt", "object_class": "car"},
            {"name": "uav10fps_car16_1", "path": "data_seq/UAV123_10fps/car16", "startFrame": 1, "endFrame": 139, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car16_1.txt", "object_class": "car"},
            {"name": "uav10fps_car16_2", "path": "data_seq/UAV123_10fps/car16", "startFrame": 139, "endFrame": 665, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car16_2.txt", "object_class": "car"},
            {"name": "uav10fps_car17", "path": "data_seq/UAV123_10fps/car17", "startFrame": 1, "endFrame": 353, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car17.txt", "object_class": "car"},
            {"name": "uav10fps_car18", "path": "data_seq/UAV123_10fps/car18", "startFrame": 1, "endFrame": 403, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car18.txt", "object_class": "car"},
            {"name": "uav10fps_car1_s", "path": "data_seq/UAV123_10fps/car1_s", "startFrame": 1, "endFrame": 492, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car1_s.txt", "object_class": "car"},
            {"name": "uav10fps_car2", "path": "data_seq/UAV123_10fps/car2", "startFrame": 1, "endFrame": 441, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car2.txt", "object_class": "car"},
            {"name": "uav10fps_car2_s", "path": "data_seq/UAV123_10fps/car2_s", "startFrame": 1, "endFrame": 107, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car2_s.txt", "object_class": "car"},
            {"name": "uav10fps_car3", "path": "data_seq/UAV123_10fps/car3", "startFrame": 1, "endFrame": 573, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car3.txt", "object_class": "car"},
            {"name": "uav10fps_car3_s", "path": "data_seq/UAV123_10fps/car3_s", "startFrame": 1, "endFrame": 434, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car3_s.txt", "object_class": "car"},
            {"name": "uav10fps_car4", "path": "data_seq/UAV123_10fps/car4", "startFrame": 1, "endFrame": 449, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car4.txt", "object_class": "car"},
            {"name": "uav10fps_car4_s", "path": "data_seq/UAV123_10fps/car4_s", "startFrame": 1, "endFrame": 277, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car4_s.txt", "object_class": "car"},
            {"name": "uav10fps_car5", "path": "data_seq/UAV123_10fps/car5", "startFrame": 1, "endFrame": 249, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car5.txt", "object_class": "car"},
            {"name": "uav10fps_car6_1", "path": "data_seq/UAV123_10fps/car6", "startFrame": 1, "endFrame": 163, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car6_1.txt", "object_class": "car"},
            {"name": "uav10fps_car6_2", "path": "data_seq/UAV123_10fps/car6", "startFrame": 163, "endFrame": 603, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car6_2.txt", "object_class": "car"},
            {"name": "uav10fps_car6_3", "path": "data_seq/UAV123_10fps/car6", "startFrame": 603, "endFrame": 985, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car6_3.txt", "object_class": "car"},
            {"name": "uav10fps_car6_4", "path": "data_seq/UAV123_10fps/car6", "startFrame": 985, "endFrame": 1309, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car6_4.txt", "object_class": "car"},
            {"name": "uav10fps_car6_5", "path": "data_seq/UAV123_10fps/car6", "startFrame": 1309, "endFrame": 1621, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car6_5.txt", "object_class": "car"},
            {"name": "uav10fps_car7", "path": "data_seq/UAV123_10fps/car7", "startFrame": 1, "endFrame": 345, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car7.txt", "object_class": "car"},
            {"name": "uav10fps_car8_1", "path": "data_seq/UAV123_10fps/car8", "startFrame": 1, "endFrame": 453, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car8_1.txt", "object_class": "car"},
            {"name": "uav10fps_car8_2", "path": "data_seq/UAV123_10fps/car8", "startFrame": 453, "endFrame": 859, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car8_2.txt", "object_class": "car"},
            {"name": "uav10fps_car9", "path": "data_seq/UAV123_10fps/car9", "startFrame": 1, "endFrame": 627, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/car9.txt", "object_class": "car"},
            {"name": "uav10fps_group1_1", "path": "data_seq/UAV123_10fps/group1", "startFrame": 1, "endFrame": 445, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group1_1.txt", "object_class": "person"},
            {"name": "uav10fps_group1_2", "path": "data_seq/UAV123_10fps/group1", "startFrame": 445, "endFrame": 839, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group1_2.txt", "object_class": "person"},
            {"name": "uav10fps_group1_3", "path": "data_seq/UAV123_10fps/group1", "startFrame": 839, "endFrame": 1309, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group1_3.txt", "object_class": "person"},
            {"name": "uav10fps_group1_4", "path": "data_seq/UAV123_10fps/group1", "startFrame": 1309, "endFrame": 1625, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group1_4.txt", "object_class": "person"},
            {"name": "uav10fps_group2_1", "path": "data_seq/UAV123_10fps/group2", "startFrame": 1, "endFrame": 303, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group2_1.txt", "object_class": "person"},
            {"name": "uav10fps_group2_2", "path": "data_seq/UAV123_10fps/group2", "startFrame": 303, "endFrame": 591, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group2_2.txt", "object_class": "person"},
            {"name": "uav10fps_group2_3", "path": "data_seq/UAV123_10fps/group2", "startFrame": 591, "endFrame": 895, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group2_3.txt", "object_class": "person"},
            {"name": "uav10fps_group3_1", "path": "data_seq/UAV123_10fps/group3", "startFrame": 1, "endFrame": 523, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group3_1.txt", "object_class": "person"},
            {"name": "uav10fps_group3_2", "path": "data_seq/UAV123_10fps/group3", "startFrame": 523, "endFrame": 943, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group3_2.txt", "object_class": "person"},
            {"name": "uav10fps_group3_3", "path": "data_seq/UAV123_10fps/group3", "startFrame": 943, "endFrame": 1457, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group3_3.txt", "object_class": "person"},
            {"name": "uav10fps_group3_4", "path": "data_seq/UAV123_10fps/group3", "startFrame": 1457, "endFrame": 1843, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/group3_4.txt", "object_class": "person"},
            {"name": "uav10fps_person1", "path": "data_seq/UAV123_10fps/person1", "startFrame": 1, "endFrame": 267, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person1.txt", "object_class": "person"},
            {"name": "uav10fps_person10", "path": "data_seq/UAV123_10fps/person10", "startFrame": 1, "endFrame": 341, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person10.txt", "object_class": "person"},
            {"name": "uav10fps_person11", "path": "data_seq/UAV123_10fps/person11", "startFrame": 1, "endFrame": 241, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person11.txt", "object_class": "person"},
            {"name": "uav10fps_person12_1", "path": "data_seq/UAV123_10fps/person12", "startFrame": 1, "endFrame": 201, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person12_1.txt", "object_class": "person"},
            {"name": "uav10fps_person12_2", "path": "data_seq/UAV123_10fps/person12", "startFrame": 201, "endFrame": 541, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person12_2.txt", "object_class": "person"},
            {"name": "uav10fps_person13", "path": "data_seq/UAV123_10fps/person13", "startFrame": 1, "endFrame": 295, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person13.txt", "object_class": "person"},
            {"name": "uav10fps_person14_1", "path": "data_seq/UAV123_10fps/person14", "startFrame": 1, "endFrame": 283, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person14_1.txt", "object_class": "person"},
            {"name": "uav10fps_person14_2", "path": "data_seq/UAV123_10fps/person14", "startFrame": 283, "endFrame": 605, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person14_2.txt", "object_class": "person"},
            {"name": "uav10fps_person14_3", "path": "data_seq/UAV123_10fps/person14", "startFrame": 605, "endFrame": 975,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123_10fps/person14_3.txt", "object_class": "person"},
            {"name": "uav10fps_person15", "path": "data_seq/UAV123_10fps/person15", "startFrame": 1, "endFrame": 447, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person15.txt", "object_class": "person"},
            {"name": "uav10fps_person16", "path": "data_seq/UAV123_10fps/person16", "startFrame": 1, "endFrame": 383, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person16.txt", "object_class": "person"},
            {"name": "uav10fps_person17_1", "path": "data_seq/UAV123_10fps/person17", "startFrame": 1, "endFrame": 501, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person17_1.txt", "object_class": "person"},
            {"name": "uav10fps_person17_2", "path": "data_seq/UAV123_10fps/person17", "startFrame": 501, "endFrame": 783,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123_10fps/person17_2.txt", "object_class": "person"},
            {"name": "uav10fps_person18", "path": "data_seq/UAV123_10fps/person18", "startFrame": 1, "endFrame": 465, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person18.txt", "object_class": "person"},
            {"name": "uav10fps_person19_1", "path": "data_seq/UAV123_10fps/person19", "startFrame": 1, "endFrame": 415, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person19_1.txt", "object_class": "person"},
            {"name": "uav10fps_person19_2", "path": "data_seq/UAV123_10fps/person19", "startFrame": 415, "endFrame": 931,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123_10fps/person19_2.txt", "object_class": "person"},
            {"name": "uav10fps_person19_3", "path": "data_seq/UAV123_10fps/person19", "startFrame": 931, "endFrame": 1453,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123_10fps/person19_3.txt", "object_class": "person"},
            {"name": "uav10fps_person1_s", "path": "data_seq/UAV123_10fps/person1_s", "startFrame": 1, "endFrame": 534, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person1_s.txt", "object_class": "person"},
            {"name": "uav10fps_person2_1", "path": "data_seq/UAV123_10fps/person2", "startFrame": 1, "endFrame": 397, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person2_1.txt", "object_class": "person"},
            {"name": "uav10fps_person2_2", "path": "data_seq/UAV123_10fps/person2", "startFrame": 397, "endFrame": 875, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person2_2.txt", "object_class": "person"},
            {"name": "uav10fps_person20", "path": "data_seq/UAV123_10fps/person20", "startFrame": 1, "endFrame": 595, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person20.txt", "object_class": "person"},
            {"name": "uav10fps_person21", "path": "data_seq/UAV123_10fps/person21", "startFrame": 1, "endFrame": 163, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person21.txt", "object_class": "person"},
            {"name": "uav10fps_person22", "path": "data_seq/UAV123_10fps/person22", "startFrame": 1, "endFrame": 67, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person22.txt", "object_class": "person"},
            {"name": "uav10fps_person23", "path": "data_seq/UAV123_10fps/person23", "startFrame": 1, "endFrame": 133, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person23.txt", "object_class": "person"},
            {"name": "uav10fps_person2_s", "path": "data_seq/UAV123_10fps/person2_s", "startFrame": 1, "endFrame": 84, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person2_s.txt", "object_class": "person"},
            {"name": "uav10fps_person3", "path": "data_seq/UAV123_10fps/person3", "startFrame": 1, "endFrame": 215, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person3.txt", "object_class": "person"},
            {"name": "uav10fps_person3_s", "path": "data_seq/UAV123_10fps/person3_s", "startFrame": 1, "endFrame": 169, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person3_s.txt", "object_class": "person"},
            {"name": "uav10fps_person4_1", "path": "data_seq/UAV123_10fps/person4", "startFrame": 1, "endFrame": 501, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person4_1.txt", "object_class": "person"},
            {"name": "uav10fps_person4_2", "path": "data_seq/UAV123_10fps/person4", "startFrame": 501, "endFrame": 915, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person4_2.txt", "object_class": "person"},
            {"name": "uav10fps_person5_1", "path": "data_seq/UAV123_10fps/person5", "startFrame": 1, "endFrame": 293, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person5_1.txt", "object_class": "person"},
            {"name": "uav10fps_person5_2", "path": "data_seq/UAV123_10fps/person5", "startFrame": 293, "endFrame": 701, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person5_2.txt", "object_class": "person"},
            {"name": "uav10fps_person6", "path": "data_seq/UAV123_10fps/person6", "startFrame": 1, "endFrame": 301, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person6.txt", "object_class": "person"},
            {"name": "uav10fps_person7_1", "path": "data_seq/UAV123_10fps/person7", "startFrame": 1, "endFrame": 417, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person7_1.txt", "object_class": "person"},
            {"name": "uav10fps_person7_2", "path": "data_seq/UAV123_10fps/person7", "startFrame": 417, "endFrame": 689, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person7_2.txt", "object_class": "person"},
            {"name": "uav10fps_person8_1", "path": "data_seq/UAV123_10fps/person8", "startFrame": 1, "endFrame": 359, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person8_1.txt", "object_class": "person"},
            {"name": "uav10fps_person8_2", "path": "data_seq/UAV123_10fps/person8", "startFrame": 359, "endFrame": 509, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person8_2.txt", "object_class": "person"},
            {"name": "uav10fps_person9", "path": "data_seq/UAV123_10fps/person9", "startFrame": 1, "endFrame": 221, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/person9.txt", "object_class": "person"},
            {"name": "uav10fps_truck1", "path": "data_seq/UAV123_10fps/truck1", "startFrame": 1, "endFrame": 155, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/truck1.txt", "object_class": "truck"},
            {"name": "uav10fps_truck2", "path": "data_seq/UAV123_10fps/truck2", "startFrame": 1, "endFrame": 129, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/truck2.txt", "object_class": "truck"},
            {"name": "uav10fps_truck3", "path": "data_seq/UAV123_10fps/truck3", "startFrame": 1, "endFrame": 179, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/truck3.txt", "object_class": "truck"},
            {"name": "uav10fps_truck4_1", "path": "data_seq/UAV123_10fps/truck4", "startFrame": 1, "endFrame": 193, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/truck4_1.txt", "object_class": "truck"},
            {"name": "uav10fps_truck4_2", "path": "data_seq/UAV123_10fps/truck4", "startFrame": 193, "endFrame": 421, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/truck4_2.txt", "object_class": "truck"},
            {"name": "uav10fps_uav1_1", "path": "data_seq/UAV123_10fps/uav1", "startFrame": 1, "endFrame": 519, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav1_1.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav1_2", "path": "data_seq/UAV123_10fps/uav1", "startFrame": 519, "endFrame": 793, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav1_2.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav1_3", "path": "data_seq/UAV123_10fps/uav1", "startFrame": 825, "endFrame": 1157, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav1_3.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav2", "path": "data_seq/UAV123_10fps/uav2", "startFrame": 1, "endFrame": 45, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav2.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav3", "path": "data_seq/UAV123_10fps/uav3", "startFrame": 1, "endFrame": 89, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav3.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav4", "path": "data_seq/UAV123_10fps/uav4", "startFrame": 1, "endFrame": 53, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav4.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav5", "path": "data_seq/UAV123_10fps/uav5", "startFrame": 1, "endFrame": 47, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav5.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav6", "path": "data_seq/UAV123_10fps/uav6", "startFrame": 1, "endFrame": 37, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav6.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav7", "path": "data_seq/UAV123_10fps/uav7", "startFrame": 1, "endFrame": 125, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav7.txt", "object_class": "aircraft"},
            {"name": "uav10fps_uav8", "path": "data_seq/UAV123_10fps/uav8", "startFrame": 1, "endFrame": 101, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/uav8.txt", "object_class": "aircraft"},
            {"name": "uav10fps_wakeboard1", "path": "data_seq/UAV123_10fps/wakeboard1", "startFrame": 1, "endFrame": 141, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard1.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard10", "path": "data_seq/UAV123_10fps/wakeboard10", "startFrame": 1, "endFrame": 157,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard10.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard2", "path": "data_seq/UAV123_10fps/wakeboard2", "startFrame": 1, "endFrame": 245, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard2.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard3", "path": "data_seq/UAV123_10fps/wakeboard3", "startFrame": 1, "endFrame": 275, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard3.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard4", "path": "data_seq/UAV123_10fps/wakeboard4", "startFrame": 1, "endFrame": 233, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard4.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard5", "path": "data_seq/UAV123_10fps/wakeboard5", "startFrame": 1, "endFrame": 559, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard5.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard6", "path": "data_seq/UAV123_10fps/wakeboard6", "startFrame": 1, "endFrame": 389, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard6.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard7", "path": "data_seq/UAV123_10fps/wakeboard7", "startFrame": 1, "endFrame": 67, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard7.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard8", "path": "data_seq/UAV123_10fps/wakeboard8", "startFrame": 1, "endFrame": 515, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard8.txt", "object_class": "person"},
            {"name": "uav10fps_wakeboard9", "path": "data_seq/UAV123_10fps/wakeboard9", "startFrame": 1, "endFrame": 119, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123_10fps/wakeboard9.txt", "object_class": "person"}
        ]

        return sequence_info_list

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAV10fpsDatasetV2(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav10fps_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def get_jpg_paths(self, dir):
        from glob import glob
        import os
        jpg_files = glob(os.path.join(dir, '*.jpg')) + glob(os.path.join(dir, '*.JPG'))
        sorted_jpg_files = sorted(os.path.basename(f) for f in jpg_files)
        return sorted_jpg_files

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/data_seq/{}/'.format(self.base_path, sequence_name)

        jpgs_name_list = self.get_jpg_paths(frames_path)

        frames_list = ['{}/{}'.format(frames_path, jpg_name) for jpg_name in jpgs_name_list]

        if len(frames_list) != len(ground_truth_rect):
            raise ValueError

        return Sequence(sequence_name, frames_list, 'uav10fps', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['bike1','bike2','bike3','bird1_1','bird1_2','bird1_3','boat1','boat2','boat3','boat4','boat5','boat6','boat7','boat8','boat9','building1','building2','building3','building4','building5','car10','car1_1','car11','car1_2','car12','car1_3','car13','car14','car15','car16_1','car16_2','car17','car18','car1_s','car2','car2_s','car3','car3_s','car4','car4_s','car5','car6_1','car6_2','car6_3','car6_4','car6_5','car7','car8_1','car8_2','car9','group1_1','group1_2','group1_3','group1_4','group2_1','group2_2','group2_3','group3_1','group3_2','group3_3','group3_4','person1','person10','person11','person12_1','person12_2','person13','person14_1','person14_2','person14_3','person15','person16','person17_1','person17_2','person18','person19_1','person19_2','person19_3','person1_s','person20','person2_1','person21','person2_2','person22','person23','person2_s','person3','person3_s','person4_1','person4_2','person5_1','person5_2','person6','person7_1','person7_2','person8_1','person8_2','person9','truck1','truck2','truck3','truck4_1','truck4_2','uav1_1','uav1_2','uav1_3','uav2','uav3','uav4','uav5','uav6','uav7','uav8','wakeboard1','wakeboard10','wakeboard2','wakeboard3','wakeboard4','wakeboard5','wakeboard6','wakeboard7','wakeboard8','wakeboard9']
        return sequence_list
