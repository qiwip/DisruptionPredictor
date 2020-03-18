from DDB.Data import Reader
from DDB.Service import Query
import os
import ast
from scipy import signal
import traceback
import numpy as np
import configparser
import tensorflow as tf
"""
算法思路：首先把各个诊断都按时间截取一致，然后得到时间长度，根据要求的输出采样率和时间算出长度，然后每个再resample成相同长度的，之后再截
取y生成，输入是时间长度和采样率，计算生成诊断等长的序列，截取的时候按步长截和x一样
时间×采样率=采样点

数据集划分思路：因为最后的结果是按炮的，而数据集是按切片的，不知道多少炮是有完整数据的，所以先不划分，先切片，切片的时候文件名标注炮号
序列和y，在生成数据集的时候去查找切片就行了，按炮号分训练集测试集和验证集
归一化：归一化参数也存储在数据库里，可以按需归一化
"""


def y(length, sample_rate, disruptive):
    """
    生成训练目标，既破裂概率，非破裂全0
    :param length:总的序列长度
    :param sample_rate:采样率
    :param disruptive:是否是破裂
    :return:
    """
    # 这里是破裂前15ms是1, 0和1之间用15ms的sigmoid
    if disruptive:
        if length < 2*30*sample_rate:
            return np.ones([length])
        y_ = np.zeros([length-2*15*sample_rate])
        x = np.linspace(-10, 0, 15*sample_rate)
        y_ = np.append(y_, 2 * tf.sigmoid(x))
        y_ = np.append(y_, np.ones([15*sample_rate]))
    else:
        y_ = np.zeros([length])
    return y_


class Cutter:
    def __init__(self, normalized=False):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'DataSetConfig.ini'))
        self._tags = ast.literal_eval(config['Diagnosis']['tags'])
        self._sample_rate = int(config['Diagnosis']['sample_rate'])
        self._frame_size = int(config['Diagnosis']['frame_size'])
        self._step = int(config['Diagnosis']['step'])
        self._npy_path = config['path']['npy']
        if not os.path.exists(self._npy_path):
            os.makedirs(self._npy_path)
        ddb = Query()
        self._normalized = normalized
        if normalized:
            self._normalize_param = ddb.get_normalize_parm(self._tags)
        my_query = {'IsValidShot': True, 'IsDisrupt': False}
        self._shots = ddb.query(my_query)
        my_query = {'IsValidShot': True, 'IsDisrupt': True, 'CqTime': {"$gte": 0.05}, 'IpFlat': {'$gte': 110}}
        self._shots += ddb.query(my_query)

    def run(self):
        data_reader = Reader()
        ddb = Query()
        for shot in self._shots:
            # if shot < 1065500 or shot > 1065599:
            #     continue
            print(shot)
            try:
                tags = ddb.tag(shot)
                if tags['IsDisrupt']:
                    t1 = tags['CqTime']
                else:
                    t1 = tags['RampDownTime']
                new_dig_length = int((t1 * 1000 - 50) * self._sample_rate)
                data = data_reader.read_many(shot, self._tags)
                digs = []
                for tag, (dig, time) in data.items():
                    dig = dig[(0.05 <= time) & (time <= t1)]
                    if self._normalized:
                        dig = (dig - self._normalize_param[tag]['min']) / \
                              (self._normalize_param[tag]['max'] - self._normalize_param[tag]['min'])
                    digs.append(signal.resample(dig, new_dig_length))

                digs = np.array(digs)
                y_ = y(new_dig_length, self._sample_rate, tags['IsDisrupt'])
                index = 0
                path = os.path.join(self._npy_path, '{}'.format(shot))
                if not os.path.exists(path):
                    os.makedirs(path)
                while index + self._frame_size <= new_dig_length:
                    frame = digs[:, index: index + self._frame_size]
                    y_frame = y_[index: index + self._frame_size]
                    np.save(os.path.join(path, 'x_{}.npy'.format(int(index / self._step))), frame)
                    np.save(os.path.join(path, 'y_{}.npy'.format(int(index / self._step))), y_frame)
                    index += self._step
                if index + self._frame_size - new_dig_length < self._frame_size / 2:
                    frame = digs[:, new_dig_length - self._frame_size: new_dig_length]
                    y_frame = y_[new_dig_length - self._frame_size: new_dig_length]
                    np.save(os.path.join(path, 'x_{}.npy'.format(int(index / self._step))), frame)
                    np.save(os.path.join(path, 'y_{}.npy'.format(int(index / self._step))), y_frame)
            except Exception as e:
                print(e)
                traceback.print_exc()

    def get_one(self, shot):
        data_reader = Reader()
        ddb = Query()
        try:
            tags = ddb.tag(shot)
            if tags['IsDisrupt']:
                t1 = tags['CqTime']
            else:
                t1 = tags['RampDownTime']
            new_dig_length = int((t1 * 1000 - 50) * self._sample_rate)
            data = data_reader.read_many(shot, self._tags)
            digs = []
            for tag, (dig, time) in data.items():
                dig = dig[(0.05 <= time) & (time <= t1)]
                if self._normalized:
                    dig = (dig - self._normalize_param[tag]['min']) / \
                          (self._normalize_param[tag]['max'] - self._normalize_param[tag]['min'])
                digs.append(signal.resample(dig, new_dig_length))

            digs = np.array(digs)
            y_ = y(new_dig_length, self._sample_rate, tags['IsDisrupt'])
            index = 0
            x = list()
            labels = list()
            while index + self._frame_size <= new_dig_length:
                frame = digs[:, index: index + self._frame_size]
                y_frame = y_[index: index + self._frame_size]
                # index += self.frame_size
                x.append(frame)
                labels.append(y_frame[-1])
                index += self._step
            return np.array(x), np.array(labels)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return None, None


if __name__ == '__main__':
    cutter = Cutter(normalized=True)
    x, y = cutter.get_one(1065500)
    print(x.shape, y.shape)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(y)
    # plt.show()
