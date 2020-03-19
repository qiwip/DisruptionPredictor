import os
import numpy as np
import configparser
import tensorflow as tf
from DDB.Service import Query


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
        ddb = Query()
        shots = list()
        # my_query = {'IsValidShot': True, 'IsDisrupt': False}
        # for shot in ddb.query(my_query):
        #     if os.path.exists(os.path.join(self.npy_path, '{}'.format(shot))):
        #         shots.append(shot)
        #         if len(shots) >= self.shots/2:
        #             break

        my_query = {'IsValidShot': True, 'IsDisrupt': True, 'CqTime': {"$gte": 0.15}, 'IpFlat': {'$gte': 110}}
        for shot in ddb.query(my_query):
            if os.path.exists(os.path.join(self.npy_path, '{}'.format(shot))):
                shots.append(shot)
                if len(shots) >= self.shots:
                    break
        if not os.path.exists('log'):
            os.mkdir('log')
        with open(os.path.join('log', 'ShotsUsed4Training.txt'), 'w') as f:
            for shot in shots:
                print(shot, file=f)

        for shot in shots:
            file_names = [i for i in os.listdir(os.path.join(self.npy_path, '{}'.format(shot))) if 'x' in i]
            for file in file_names:
                x = np.load(os.path.join(self.npy_path, '{}'.format(shot), file))
                y = np.load(os.path.join(self.npy_path, '{}'.format(shot), file.replace('x', 'y')))
                if y[-1] > 0:
                    examples_dis.append(x)
                    labels_dis.append(y[-1])
                else:
                    examples_und.append(x)
                    labels_und.append(y[-1])
        len_und = len(labels_und)
        len_dis = len(labels_dis)
        print('Length un_disruption: ', len_und, '\nLength disruption: ', len_dis)
        # --------------------------------------------------------------------------------------
        # 均衡策略1:扩大disruption, un_disruption不变
        # --------------------------------------------------------------------------------------
        # dataset_und = tf.data.Dataset.from_tensor_slices((examples_und, labels_und))
        # dataset_dis = tf.data.Dataset.from_tensor_slices((examples_dis, labels_dis))
        #
        # split_point_und = (int(len(labels_und)*self.train), int(len(labels_und)*self.test))
        # split_point_dis = (int(len(labels_dis)*self.train), int(len(labels_dis)*self.test))
        #
        # train_dataset_und = dataset_und.take(split_point_und[0])
        # test_dataset_und = dataset_und.skip(split_point_und[0]).take(split_point_und[1])
        # train_dataset_dis = dataset_dis.take(split_point_dis[0])
        # test_dataset_dis = dataset_dis.skip(split_point_dis[0]).take(split_point_dis[1])
        #
        # train_dataset_dis = train_dataset_dis.repeat(int(split_point_und[0]/split_point_dis[0]))
        #
        # train_dataset = train_dataset_und.concatenate(train_dataset_dis)
        # test_dataset = test_dataset_und.concatenate(test_dataset_dis)
        # --------------------------------------------------------------------------------------
        # 均衡策略2:disruption扩大2倍, 随机抽取un_disruption, 比例为und/dis = 6/4
        # --------------------------------------------------------------------------------------
        dataset_und = tf.data.Dataset.from_tensor_slices((examples_und, labels_und))
        dataset_dis = tf.data.Dataset.from_tensor_slices((examples_dis, labels_dis))
        dataset_und = dataset_und.shuffle(buffer_size=len_und).take(3*len_dis)
        dataset_dis = dataset_dis.repeat(2)
        dataset = dataset_und.concatenate(dataset_dis)
        dataset = dataset.shuffle(5*len_dis)
        train_dataset = dataset.take(int(5*len_dis*self.train))
        test_dataset = dataset.skip(int(5*len_dis*self.train)).take(int(5*len_dis*self.test))

        return train_dataset, test_dataset


if __name__ == '__main__':
    ds = DataSet()
    ds.load()
