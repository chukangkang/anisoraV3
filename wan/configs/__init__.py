# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_14B import i2v_14B
from .wan_t2v_1_3B import t2v_1_3B
from .wan_t2v_14B import t2v_14B


def get_optimal_window_size(resolution):
    """
    根据分辨率动态计算window_size，适配480/720/1080P
    :param resolution: 字符串，如"480*832"、"720*1280"、"1080*1920"
    :return: (T, H, W) 最优窗口大小
    """
    # 解析分辨率
    w, h = map(int, resolution.split('*'))
    # 转为特征图尺寸（÷8，对应vae_stride=(4,8,8)）
    feat_w = w // 8
    feat_h = h // 8
    
    # 时间窗固定为6（24帧视频的最优时间连贯性）
    T = 6
    # 高度窗：取能整除feat_h的最大数，且≤30（显存限制）
    H = max([x for x in range(30, 0, -1) if feat_h % x == 0 or feat_h % x < 5], default=8)
    # 宽度窗：取能整除feat_w的最大数，且≤40（显存限制）
    W = max([x for x in range(40, 0, -1) if feat_w % x == 0 or feat_w % x < 5], default=16)
    
    return (T, H, W)


# the config of t2i_14B is the same as t2v_14B
t2i_14B = copy.deepcopy(t2v_14B)
t2i_14B.__name__ = 'Config: Wan T2I 14B'

WAN_CONFIGS = {
    't2v-14B': t2v_14B,
    't2v-1.3B': t2v_1_3B,
    'i2v-14B': i2v_14B,
    't2i-14B': t2i_14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
    '960*544': (960, 544),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '960*544': 960 * 544,
}

SUPPORTED_SIZES = {
    't2v-14B': ('720*1280', '1280*720', '480*832', '832*480'),
    't2v-1.3B': ('480*832', '832*480'),
    'i2v-14B': ('720*1280', '1280*720', '480*832', '832*480', '960*544'),
    't2i-14B': tuple(SIZE_CONFIGS.keys()),
}
