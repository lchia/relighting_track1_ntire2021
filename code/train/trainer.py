import os
from datetime import datetime

import torch.multiprocessing as mp

from utils import Logger

from .test import test


class Trainer(object):
    def __init__(self, para):
        self.para = para
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '6666'

    def run(self):
        # recoding parameters
        self.para.time = datetime.now()
        logger = Logger(self.para)
        logger.record_para()

        # test
        test(self.para, logger)
