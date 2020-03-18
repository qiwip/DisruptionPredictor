import os
import numpy as np
import configparser
import tensorflow as tf
from DDB.Service import Query


class DataSetShots:
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

    def get(self):
        """
        加载npy数据到tf.data.DataSet
        :return: training set, test set
        """
        train_test_shots = list()
        with open(os.path.join('log', 'ShotsInDataset.txt'), 'r') as f:
            for shot in f.readlines():
                train_test_shots.append(int(shot))
        train_test_shots.sort(reverse=False)
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
        shots.sort(reverse=False)

        with open(os.path.join('log', 'IsDisruptShots.txt'), 'w') as f:
            for shot in shots:
                if shot in train_test_shots:
                    print('{} 1'.format(shot), file=f)
                else:
                    print('{} 0'.format(shot), file=f)

        my_query = {'IsValidShot': True, 'IsDisrupt': False, 'IpFlat': {'$gte': 110}}
        for shot in ddb.query(my_query):
            if os.path.exists(os.path.join(self.npy_path, '{}'.format(shot))):
                shots.append(shot)

        shots.sort(reverse=False)

        with open(os.path.join('log', 'UnDisruptShots.txt'), 'w') as f:
            for shot in shots:
                if shot < train_test_shots[0] or shot > train_test_shots[-1]:
                    print('{} 0'.format(shot), file=f)
                else:
                    print('{} 1'.format(shot), file=f)


if __name__ == '__main__':
    ds = DataSetShots()
    ds.get()
