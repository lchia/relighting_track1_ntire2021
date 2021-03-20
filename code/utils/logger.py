import os
import shutil
from datetime import datetime
from os.path import dirname, join

import torch


class Logger():
    def __init__(self, para):
        self.para = para
        # save pynet
        self.pynet_level = para.pynet_level
        self.stage = para.stage

        if not para.test_only:
            now = datetime.now() if 'time' not in vars(para) else para.time
            now = now.strftime("%Y_%m_%d_%H_%M_%S")
            self.now_str = now
            #mark = para.model + '_' + para.dataset
            mark = para.model + '_' + para.log_tag
            if para.model == 'pynet2':
                file_path = join(para.save_dir, mark + '_nc' + str(para.n_colors) + '_nf' + str(para.n_features) + '_' + para.trainer_mode, '%s_level_%d_log.txt'%(self.now_str, para.pynet_level))
            elif para.model == 'wdrn2':
                file_path = join(para.save_dir, mark + '_nc' + str(para.n_colors) + '_nf' + str(para.n_features) + '_' + para.trainer_mode, '%s_stage_%d_log.txt'%(self.now_str, para.stage))
            else:
                file_path = join(para.save_dir, now + '_' + mark + '_nc' + str(para.n_colors) + '_nf' + str(para.n_features) + '_' + para.trainer_mode, 'train_log.txt')

            self.log_file = file_path
            self.save_dir = dirname(file_path) 
            self.check_dir(file_path)
            self.logger = open(file_path, 'a+')
        else:
            ckpt_dir, ckpt_name = os.path.split(para.test_checkpoint)
            file_path = join(ckpt_dir, 'test_log.txt')
            self.log_file = file_path
            self.save_dir = dirname(file_path) 
            self.logger = open(file_path, 'w')
 
        # variable register
        self.register_dict = {}
        # tensorboard

    def record_para(self):
        self('recording parameters ...')
        for key, value in vars(self.para).items():
            self('{}: {}'.format(key, value), timestamp=False)

    def check_dir(self, file_path):
        dir = dirname(file_path)
        os.makedirs(dir, exist_ok=True)

    def __call__(self, *args, verbose=True, prefix='', timestamp=True):
        if timestamp:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d, %H:%M:%S - ")
        else:
            now = ''
        info = prefix + now
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + '\n'
        self.logger.write(info)
        if verbose:
            print(info, end='')
        self.logger.flush()

    def __del__(self):
        self.logger.close()

    # register values for each epoch, such as loss, PSNR etc.
    def register(self, name, epoch, value):
        if name in self.register_dict:
            self.register_dict[name][epoch] = value
            if value > self.register_dict[name]['max']:
                self.register_dict[name]['max'] = value
            if value < self.register_dict[name]['min']:
                self.register_dict[name]['min'] = value
        else:
            self.register_dict[name] = {}
            self.register_dict[name][epoch] = value
            self.register_dict[name]['max'] = value
            self.register_dict[name]['min'] = value

    def report(self, items, state, epoch):
        # items - [['MSE', 'min'], ['PSNR', 'max'] ... ]
        msg = '[{}] '.format(state.lower())
        state = '_' + state.lower()
        for i in range(len(items)):
            item, best = items[i]
            msg += '{} : {:.4f} (best {:.4f})'.format(
                item,
                self.register_dict[item + state][epoch],
                self.register_dict[item + state][best]
            )
            if i < len(items) - 1:
                msg += ', '
        self(msg, timestamp=False)

    def is_best(self, epoch):
        item = self.register_dict[self.para.loss + '_valid']
        return item[epoch] == item['min']

    def save(self, state, filename='checkpoint.pth.tar'):
        path = join(self.save_dir, filename)
        torch.save(state, path)
        if self.is_best(state['epoch']):
            copy_path = join(self.save_dir, 'model_best.pth.tar')
            shutil.copy(path, copy_path)

    def save_dense(self, state, filename='checkpoint.pth.tar'):
        path = join(self.save_dir, 'epoch_%04d_%s'%(state['epoch'], filename))
        torch.save(state, path)
        if self.is_best(state['epoch']):
            copy_path = join(self.save_dir, '%s'%('model_best.pth.tar',))
            shutil.copy(path, copy_path)

            # record the best epoch 
            filepath = join(self.save_dir, 'best_checkpoint.txt')
            fid = open(filepath, 'w')
            fid.write('epoch %d\n'%(state['epoch']))
            fid.close()


    def save_pynet(self, state, filename='checkpoint.pth.tar'):
        path = join(self.save_dir, '%s_level_%d_%s'%(self.now_str, self.pynet_level, filename))
        torch.save(state, path)
        if self.is_best(state['epoch']):
            copy_path = join(self.save_dir, 'level_%d_%s'%(self.pynet_level, 'model_best.pth.tar'))
            shutil.copy(path, copy_path)

    def save_wdrn2(self, state, filename='checkpoint.pth.tar'):
        path = join(self.save_dir, '%s_stage_%d_%s'%(self.now_str, self.stage, filename))
        torch.save(state, path)
        if self.is_best(state['epoch']):
            copy_path = join(self.save_dir, 'stage_%d_%s'%(self.stage, 'model_best.pth.tar'))
            shutil.copy(path, copy_path)


    def save_dense_wdrn2(self, state, filename='checkpoint.pth.tar'):
        path = join(self.save_dir, 'stage_%d_epoch_%04d_%s'%(self.stage, state['epoch'], filename))
        torch.save(state, path)
        if self.is_best(state['epoch']):
            copy_path = join(self.save_dir, 'stage_%d_%s'%(self.stage, 'model_best.pth.tar',))
            shutil.copy(path, copy_path)

            # record the best epoch 
            filepath = join(self.save_dir, 'stage_%d_best_checkpoint.txt'%(self.stage))
            fid = open(filepath, 'w')
            fid.write('epoch %d\n'%(state['epoch']))
            fid.close()
