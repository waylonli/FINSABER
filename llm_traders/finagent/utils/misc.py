# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import warnings
from typing import Union

from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from datetime import datetime, timedelta

def generate_intervals(start_date, end_date, interval_level='year'):

    intervals = []

    if interval_level == 'year':
        current_date = start_date
        while current_date < end_date:
            next_year = current_date.replace(year=current_date.year + 1)
            if next_year > end_date:
                next_year = end_date
            interval = (current_date, next_year)
            intervals.append(interval)
            current_date = next_year
    elif interval_level == 'day':
        current_date = start_date
        while current_date < end_date:
            next_day = current_date + timedelta(days=1)
            interval = (current_date, next_day)
            intervals.append(interval)
            current_date = next_day
    elif interval_level == 'month':
        current_date = start_date

        while current_date < end_date:
            year, month = current_date.year, current_date.month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            if next_month > end_date:
                next_month = end_date
            interval = (current_date, next_month)
            intervals.append(interval)
            current_date = next_month
    else:
        return None

    return intervals

def generate_dates(start_date, end_date, interval_level='year'):

    dates = []

    if interval_level == 'year':
        current_date = start_date
        while current_date < end_date:
            next_year = current_date.replace(year=current_date.year + 1)
            if next_year > end_date:
                next_year = end_date
            date = current_date
            dates.append(date)
            current_date = next_year
    elif interval_level == 'day':
        current_date = start_date
        while current_date < end_date:
            next_day = current_date + timedelta(days=1)
            date = current_date
            dates.append(date)
            current_date = next_day
    else:
        return None

    return dates

def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path

def update_data_root(cfg, root):
    cfg.root = root
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = root

def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    """Get the test dataset pipeline from entire config.

    Args:
        cfg (str or :obj:`ConfigDict`): the entire config. Can be a config
            file or a ``ConfigDict``.

    Returns:
        :obj:`ConfigDict`: the config of test dataset.
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    def _get_test_pipeline_cfg(dataset_cfg):
        if 'pipeline' in dataset_cfg:
            return dataset_cfg.pipeline
        # handle dataset wrapper
        elif 'dataset' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        # handle dataset wrappers like ConcatDataset
        elif 'datasets' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])

        raise RuntimeError('Cannot find `pipeline` in `test_dataloader`')

    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)
