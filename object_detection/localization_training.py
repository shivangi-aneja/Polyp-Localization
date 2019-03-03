import torch
import os
import datetime
import pytz
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import shutil
from object_detection.logging.logger import rootLogger
from object_detection.logging.tf_logger import Logger
from object_detection.metrics.evaluation_metrics import label_accuracy_score, get_dice_score
from metric_evaluation import get_eval_metrics
from object_detection.utils.save_predictions import save_image

TIME_ZONE = 'Europe/Berlin'


class DeepSegmentation(object):

    """Class encapsulating training of network.
        Parameters
        ----------
        autoencoder : `torch.nn.Module`
            Segmentor model.
        optim : `torch.optim`
        optim_kwargs : dict

    """

    def __init__(self, model, dataset, model_name, gpu_id, out_path, epochs, model_params=None,
                 optim=None, optim_kwargs=None, batch_size=32, lr=1e-4, loss_func=None, tf_log_path=None, log_path=None,
                size_average=False,image_path = None):

        """
        Initialize the variables related to network
        """

        self.model = model
        if model_params is None or not len(model_params):
            model_params = filter(lambda x: x.requires_grad, self.model.parameters())

        optim = optim or torch.optim.SGD
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        self.optim = optim(model_params, **optim_kwargs)
        self.batch_size = batch_size
        self.lr =lr
        self.tf_log_path = tf_log_path
        self.out_path = out_path
        self.log_path = log_path
        self.dataset = dataset
        self.model_name = model_name
        if not osp.exists(self.out_path):
            os.makedirs(self.out_path)
        self.loss_func = loss_func or nn.MSELoss()
        if gpu_id > -1:
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            self.model.cuda()
        self.size_average=size_average
        self.start_epoch = 0
        self.epochs = epochs
        self.best_mean_iu = 0
        self.image_path = image_path
        self.timestamp_start = datetime.datetime.now(pytz.timezone(TIME_ZONE))
        # Tensorboard Logging
        self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=self.tf_log_path)


    def train(self, train_loader, val_loader, n_classes):
        rootLogger.info("Mode       Epoch             Loss             Mean Acc             BG Acc              FG Acc           IOU")

        for epoch in range(self.start_epoch, self.epochs):
            self.start_epoch = epoch

            # Train the model
            self.train_epoch(train_loader=train_loader, n_classes=n_classes)

            # Validate the model
            self.validate_epoch(val_loader=val_loader, n_classes=n_classes)


    def train_epoch(self,train_loader, n_classes):

        # Change the model to training model
        self.model.train()

        acc_list, acc_cls_list, mean_iu_list, fwavacc_list = np.array([]), np.array([]), np.array([]), np.array([])
        epoch_train_loss = 0.
        for batch_idx, (data, target) in enumerate(train_loader):

            assert self.model.training

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.optim.zero_grad()
            score = self.model(data)

            loss = self.loss_func(score, target, size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            epoch_train_loss += loss_data

            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbl_true, lbl_pred, n_class=n_classes)
            acc_list = np.append(acc_list,acc)
            acc_cls_list = np.append(acc_cls_list,acc_cls)
            mean_iu_list = np.append(mean_iu_list,mean_iu)
            fwavacc_list = np.append(fwavacc_list,fwavacc)

        epoch_loss = epoch_train_loss/len(train_loader)
        rootLogger.info("Train     %d/%d           [%8.3f]           [%.4f]            [%.4f]           [%.4f]           [%.4f]" %
        (self.start_epoch+1, self.epochs, epoch_loss, np.mean(acc_list),np.mean(acc_cls_list[0]),
        np.mean(acc_cls_list[1]), np.mean(mean_iu)))

        # Log the training losses
        self.logger.log(mode='train', loss= epoch_loss, accuracy=np.mean(acc_list), iou=np.mean(mean_iu), epoch=self.start_epoch+1)


    def validate_epoch(self, val_loader, n_classes):

        # Set the model to evaluation mode
        self.model.eval()

        n_class = n_classes

        val_loss = 0.
        label_trues, label_preds = [], []
        orig_img, pred_img, gt_img = list(), list(), list()
        for batch_idx, (data, target) in enumerate(val_loader):

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            with torch.no_grad():
                score = self.model(data)

            # Calculate loss
            loss = self.loss_func(score, target,
                                  size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()

            # the predicted class is class with maximum probability
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

            # Save the predicted masks
            ctr = batch_idx*self.batch_size
            save_image(lbl_pred,image_path=self.image_path,file_num=ctr)

            # true class
            lbl_true = target.data.cpu()

            # Just log 24 images
            if len(orig_img) <= 24:
                orig_img.append(imgs[0])
                pred_img.append( torch.from_numpy(np.expand_dims(lbl_pred, axis=1)[0]).float())
                gt_img.append( torch.from_numpy(np.expand_dims(lbl_true, axis=1)[0]).float())

            # For Metric Calculation
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                label_trues.append(lt.numpy())
                label_preds.append(lp)

        metrics = label_accuracy_score(label_trues, label_preds, n_class)

        val_loss /= len(val_loader)

        rootLogger.info(
            "Val       %d/%d           [%8.3f]           [%.4f]            [%.4f]           [%.4f]           [%.4f]"%
                        (self.start_epoch+1, self.epochs, val_loss, metrics[0], metrics[1][0], metrics[1][1], metrics[2]))
        # Log the validation losses
        self.logger.log(mode='val',loss=val_loss, accuracy=metrics[0], iou=metrics[2], epoch=self.start_epoch+1)

        # Log the prediction for validation
        self.logger.log_images(mode='image', images=torch.stack(orig_img), epoch=self.start_epoch+1, normalize=True)
        self.logger.log_predictions(mode='predicted', images=torch.stack(pred_img), epoch=self.start_epoch+1)
        self.logger.log_predictions(mode='ground_truth', images=torch.stack(gt_img), epoch=self.start_epoch+1)

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.start_epoch+1,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out_path, self.model_name+'_checkpoint.pth.tar'))

        # Copy and save the best model
        if is_best:
            shutil.copy(osp.join(self.out_path, self.model_name+'_checkpoint.pth.tar'),
                        osp.join(self.out_path, self.model_name+'_model_best.pth.tar'))



    def validate(self, val_loader, n_classes):

        # Set the model to evaluation mode
        self.model.eval()

        n_class = n_classes

        val_loss = 0.
        label_trues, label_preds = [], []
        rootLogger.info("Loss             Mean Acc             BG Acc            FG Acc           IOU         Dice Score      Prec        Recall        F1")

        for batch_idx, (data, target) in enumerate(val_loader):

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            with torch.no_grad():
                score = self.model(data)

            # Calculate loss
            loss = self.loss_func(score, target,
                                  size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()

            # the predicted class is class with maximum probability
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

            # Save the predicted masks
            ctr = batch_idx*self.batch_size
            save_image(lbl_pred,image_path=self.image_path,file_num=ctr,mode='pred')
            save_image(target,image_path=self.image_path,file_num=ctr,mode='gt')

            # true class
            lbl_true = target.data.cpu()

            # For Metric Calculation
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                label_trues.append(lt.numpy())
                label_preds.append(lp)

        metrics = label_accuracy_score(label_trues, label_preds, n_class)
        prf1 = get_eval_metrics(label_trues,label_preds)
        dice_score = get_dice_score(label_trues, label_preds)
        val_loss /= len(val_loader)

        rootLogger.info(
            "[%8.3f]           [%.4f]            [%.4f]           [%.4f]           [%.4f]       [%.4f]      [%.4f]      [%.4f]      [%.4f]"%
                        (val_loss, metrics[0], metrics[1][0], metrics[1][1], metrics[2], dice_score,prf1[0],prf1[1],prf1[2]))


    def test_model(self, test_loader, n_classes):

        # Set the model to evaluation mode
        self.model.eval()

        for batch_idx, (data, target) in enumerate(test_loader):

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            with torch.no_grad():
                score = self.model(data)

            # the predicted class is class with maximum probability
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

            # Save the predicted masks
            ctr = batch_idx*self.batch_size
            save_image(lbl_pred,image_path=self.image_path,file_num=ctr,mode='pred')
            save_image(target,image_path=self.image_path,file_num=ctr,mode='gt')

