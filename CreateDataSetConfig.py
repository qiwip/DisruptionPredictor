import configparser
import os


config = configparser.ConfigParser()
if(platform.system()=='Windows'):
    config['Diagnosis'] = {
        'tags': ['\\Bt', '\\MA_POL_CA01T', '\\vs_c3_aa001', '\\vs_ha_aa001'],
        'sample_rate': 100,
        'frame_size': 100,
        'step': 100
    }
    config['DataSet'] = {
        'shots': 80,
        'train': 0.8,
        'test': 0.2
    }

    config['path'] = {
        'npy': './temp/npy'
    }
else:
    config['Diagnosis'] = {
        'tags': ['\\Bt', '\\Ihfp', '\\Ivfp', '\\MA_POL_CA01T', '\\MA_POL_CA02T', '\\MA_POL_CA23T', '\\MA_POL_CA24T',
                 '\\axuv_ca_01', '\\ip', '\\sxr_cb_024', '\\sxr_cc_049', '\\vs_c3_aa001', '\\vs_ha_aa001'],
        'sample_rate': 100,
        'frame_size': 100,
        'step': 100
    }
    config['DataSet'] = {
        'shots': 300,
        'train': 0.8,
        'test': 0.2
    }

    config['path'] = {
        'npy': '/nas/wqq/npy'
    }

with open(os.path.join(os.path.dirname(__file__), 'DataSetConfig.ini'), 'w') as configfile:
    config.write(configfile)
