import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name, data_name, log_path):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_path, comment=self.comment)

    def log(self, mode, loss, accuracy, iou, epoch):

        # var_class = torch.autograd.variable.Variable
        if isinstance(loss, torch.autograd.Variable):
            loss = loss.data.cpu().numpy()
        if isinstance(accuracy, torch.autograd.Variable):
            accuracy = accuracy.data.cpu().numpy()
        if isinstance(iou, torch.autograd.Variable):
            iou = iou.data.cpu().numpy()

        #step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/loss'.format(mode+'_'+self.comment), loss, epoch)
        self.writer.add_scalar(
            '{}/accuracy'.format(mode+'_'+self.comment), accuracy, epoch)
        self.writer.add_scalar(
            '{}/iou'.format(mode+'_'+self.comment), iou, epoch)


    def log_frcnn(self, mode, rpn_cls, rpn_regr, detector_cls, detector_regr, accuracy, step):

        # var_class = torch.autograd.variable.Variable
        if isinstance(rpn_cls, torch.autograd.Variable):
            rpn_cls = rpn_cls.data.cpu().numpy()
        if isinstance(rpn_regr, torch.autograd.Variable):
            rpn_regr = rpn_regr.data.cpu().numpy()
        if isinstance(detector_cls, torch.autograd.Variable):
            detector_cls = detector_cls.data.cpu().numpy()
        if isinstance(detector_regr, torch.autograd.Variable):
            detector_regr = detector_regr.data.cpu().numpy()
        if isinstance(accuracy, torch.autograd.Variable):
            accuracy = accuracy.data.cpu().numpy()


        #step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/rpn_cls'.format(mode+'_'+self.comment), rpn_cls, step)
        self.writer.add_scalar(
            '{}/rpn_regr'.format(mode+'_'+self.comment), rpn_regr, step)
        self.writer.add_scalar(
            '{}/detector_cls'.format(mode+'_'+self.comment), detector_cls, step)
        self.writer.add_scalar(
            '{}/detector_regr'.format(mode + '_' + self.comment), detector_regr, step)
        self.writer.add_scalar(
            '{}/accuracy'.format(mode + '_' + self.comment), accuracy, step)


    def log_images(self, mode, images, epoch, normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        # if type(images) == np.ndarray:
        #     images = torch.from_numpy(images)
        #     images = images.transpose(1, 3)

        step = epoch
        img_name = '{}/images{}'.format(mode+'_'+self.comment, '',step)

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        # nrows = int(np.sqrt(num_images))
        # grid = vutils.make_grid(
        #     images, nrow=nrows, normalize=True, scale_each=True)
        self.writer.add_image(tag=img_name, img_tensor=horizontal_grid, global_step=step)

        # Save plots
        # self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def log_predictions(self, mode, images, epoch):
        '''
        input images are expected in format (NCHW)
        '''

        step = epoch
        img_name = '{}/images{}'.format(mode + '_' + self.comment, '', step)

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(tensor=images,padding=8, pad_value=1)

        # Add horizontal images to tensorboard
        self.writer.add_image(tag=img_name, img_tensor=horizontal_grid, global_step=step)


    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
