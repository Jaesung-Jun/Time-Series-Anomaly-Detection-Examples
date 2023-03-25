# NAB
from ts_datasets.anomaly import NAB

#NASA
from ts_datasets.anomaly import SMAP
from ts_datasets.anomaly import MSL

#CWRU
import cwru_py3 as cwru

class Dataset:
    def load_NAB(self, subset, root_dir):
        dataset = NAB(subset=subset, rootdir=root_dir)
        # .time_series
        return dataset
    
    def load_SMAP(self, root_dir):
        dataset = SMAP(rootdir=root_dir)
        # .time_series
        return dataset
        
    def load_MSL(self, root_dir):
        dataset = MSL(rootdir=root_dir)
        # .time_series
        return dataset
        
    def load_CWRU(self, rpm, length):
        """
        exp : '12DriveEndFault', '12FanEndFault', '48DriveEndFault'
        rpm : '1797', '1772', '1750', '1730'
        """
        
        _12_drive_end_data = cwru.CWRU("12DriveEndFault", rpm, length)
        _12_fan_end_data = cwru.CWRU("12FanEndFault", rpm, length)
        _48_drive_end_data = cwru.CWRU("48DriveEndFault", rpm, length)
        
        # .X_train, .y_train, .X_test, .y_test
        
        return _12_drive_end_data, _12_fan_end_data, _48_drive_end_data