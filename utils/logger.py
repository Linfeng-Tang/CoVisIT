

import logging
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import numbers


def as_time(s):
    s = int(s)
    one_day, one_hour, one_min = 3600 * 24, 3600, 60
    day, hour, min = 0, 0, 0
    
    if s >= one_day:
        day = int(s // one_day)
        s = s % one_day
    
    if s >= one_hour:
        hour = int(s // one_hour)
        s = s % one_hour
    
    if s >= one_min:
        min = int(s // one_min)
        s = s % one_min
    output_str = []
    if day > 0:
        output_str.append('{}days'.format(day))
    if hour > 0:
        output_str.append('{}hours'.format(hour))
    if min > 0:
        output_str.append('{}mins'.format(min))
    output_str.append('{}secs'.format(int(s)))
    return ' '.join(output_str)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '{} {:.2f}%({})'.format(as_time(s), 100 * percent, as_time(rs))


def get_logger(name='scepter', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(name)s [%(levelname)s] %(asctime)s '
            '[File: %(filename)s Function: %(funcName)s at line %(lineno)d]  %(message)s'
        )
        std_handler.setFormatter(formatter)
        std_handler.setLevel(level)
        logger.setLevel(level)
        logger.addHandler(std_handler)
    return logger


def init_logger(in_logger, log_file=None, dist_launcher='pytorch'):
    from modules.utils.distribute import get_dist_info
    rank, _ = get_dist_info()
    if rank == 0:
        if log_file is not None:
            from modules.utils.file_system import FS
            file_handler = FS.get_fs_client(log_file).get_logging_handler(
                log_file)
            formatter = logging.Formatter(
                '%(name)s [%(levelname)s] %(asctime)s [File: %(filename)s '
                'Function: %(funcName)s at line %(lineno)d]  %(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            in_logger.addHandler(file_handler)
            in_logger.info(f'Running task with log file: {log_file}')
        in_logger.setLevel(logging.INFO)
    else:
        if dist_launcher == 'pytorch':
            in_logger.setLevel(logging.ERROR)
        else:
            
            in_logger.setLevel(logging.INFO)


class StdMsg():
    def __init__(self, name='msg'):
        self.name = name

    def info(self, msg):
        sys.stdout.write('[Info]: ' + msg + '\n')

    def error(self, msg):
        sys.stdout.write('[Error]: ' + msg + '\n')

    def warning(self, msg):
        sys.stdout.write('[Warning]: ' + msg + '\n')


class LogAgg(object):
    def __init__(self):
        self.buffer = OrderedDict()
        self.counter = []

    def update(self, kv: dict, count=1):
        for k, v in kv.items():
            if isinstance(v, torch.Tensor):
                
                if not v.ndim == 0:
                    continue
                v = v.item()
            elif isinstance(v, np.ndarray):
                
                if not v.ndim == 0:
                    continue
            elif isinstance(v, numbers.Number):
                
                pass
            else:
                continue

            if k not in self.buffer:
                self.buffer[k] = []
            self.buffer[k].append(v)
        self.counter.append(count)

    def _aggregate(self, n=0):

        ret = OrderedDict()
        for key in self.buffer:
            values = np.array(self.buffer[key][-n:])
            nums = np.array(self.counter[-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            ret[key] = avg
        return ret

    def aggregate(self, log_interval=1):
        cur = self._aggregate(log_interval)
        all_mean = self._aggregate(0)
        ret = OrderedDict()
        for key in cur:
            ret[key] = (cur[key], all_mean[key])
        return ret

    def reset(self):
        self.buffer.clear()
        self.counter.clear()
