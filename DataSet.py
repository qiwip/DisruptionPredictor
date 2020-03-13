import os
import numpy as np
import configparser
import tensorflow as tf


class DataSet:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'DataSetConfig.ini'))
        self.frame_size = int(config['Diagnosis']['frame_size'])
        self.shots = int(config['DataSet']['shots'])
        self.train = float(config['DataSet']['train'])
        self.test = float(config['DataSet']['test'])
        self.npy_path = config['path']['npy']
        if not os.path.exists(self.npy_path):
            raise NotADirectoryError('Path {} don\'t exist.'.format(self.npy_path))

    def load(self):
        """
        加载npy数据到tf.data.DataSet
        :return: training set, test set
        """
        examples_und = list()
        examples_dis = list()
        labels_und = list()
        labels_dis = list()
        shot_nums = list()
        file_names = os.listdir(self.npy_path)
        print(len(file_names))
        file_names = [i for i in file_names if 'x' in i]
        print(len(file_names))
        for file in file_names:
            shot = file.split('_')[1]
            if shot not in shot_nums:
                shot_nums.append(shot)
                print(shot)
                if len(shot_nums) >= self.shots:
                    break
            x = np.load(os.path.join(self.npy_path, file))
            y = np.load(os.path.join(self.npy_path, file.replace('x', 'y')))
            if y[-1] > 0:
                examples_dis.append(x)
                labels_dis.append(y[-1])
            else:
                examples_und.append(x)
                labels_und.append(y[-1])
        print('Length un_disruption: ', len(labels_und), '\nLength disruption: ', len(labels_dis))
        dataset_und = tf.data.Dataset.from_tensor_slices((examples_und, labels_und))
        dataset_dis = tf.data.Dataset.from_tensor_slices((examples_dis, labels_dis))

        split_point_und = (int(len(labels_und)*self.train), int(len(labels_und)*self.test))
        split_point_dis = (int(len(labels_dis)*self.train), int(len(labels_dis)*self.test))

        train_dataset_und = dataset_und.take(split_point_und[0])
        test_dataset_und = dataset_und.skip(split_point_und[0]).take(split_point_und[1])
        train_dataset_dis = dataset_dis.take(split_point_dis[0])
        test_dataset_dis = dataset_dis.skip(split_point_dis[0]).take(split_point_dis[1])

        train_dataset_dis = train_dataset_dis.repeat(int(split_point_und[0]/split_point_dis[0]))

        train_dataset = train_dataset_und.concatenate(train_dataset_dis)
        test_dataset = test_dataset_und.concatenate(test_dataset_dis)

        return train_dataset, test_dataset


if __name__ == '__main__':
    ds = DataSet()
    ds.load()
