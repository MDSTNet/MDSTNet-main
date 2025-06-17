import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

from utils.dist import synchronize, is_main_process, get_rank
from tqdm import tqdm
import os

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, args, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, output_attn=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config, args)
        self.config = config
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.output_attn = output_attn
        self.c_out = args.c_out
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        
        
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        if hasattr(self.valid_data_loader.dataset, 'metedata'):
            if self.data_loader.dataset.scale:
                L, N, C = self.valid_data_loader.dataset.metedata.shape
                metedata = self.valid_data_loader.dataset.metedata.reshape(L * N, C)
                metedata = self.data_loader.dataset.mete_scaler.transform(metedata)
                self.valid_data_loader.dataset.metedata = metedata.reshape(L, N, C)
        if hasattr(self.valid_data_loader.dataset, 'AQdata'):
            if self.data_loader.dataset.scale:
                L, N, C = self.valid_data_loader.dataset.AQdata.shape
                AQdata = self.valid_data_loader.dataset.AQdata.reshape(L * N, C)
                AQdata[:, -7:] = self.data_loader.dataset.aq_scaler.transform(AQdata[:, -7:])
                self.valid_data_loader.dataset.AQdata = AQdata.reshape(L, N, C)

        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_mean = torch.Tensor(self.data_loader.dataset.aq_scaler.mean_).to(self.device)
        self.val_var = torch.sqrt(torch.Tensor(self.data_loader.dataset.aq_scaler.var_).to(self.device))
        self.loss_weight = torch.tensor([1,0.7,0.7,0.5,0.5,0.5,0.5]).to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # outputs, targets = self._valid_epoch(epoch)
        pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, (data, target) in pbar:
            
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(self.device)
            for key, value in target.items():
                if torch.is_tensor(value):
                    target[key] = value.to(self.device)
            # target = target.to(self.device)

            self.optimizer.zero_grad()
            
            output,reconstructed_out = self.model(data)
            # loss = self.criterion(output[:, :]*self.loss_weight, target['label'][:,:,:self.c_out]*self.loss_weight) + \
            #       self.criterion(reconstructed_out*self.loss_weight, target['reconstructed_label'][:,:,:self.c_out]*self.loss_weight)
            loss = self.criterion(output[:, :], target['label'][:,:,:self.c_out]) + self.criterion(reconstructed_out, target['reconstructed_label'][:,:,:self.c_out])

            loss_reduced = self.reduce_loss(loss)
            loss.backward()
            self.optimizer.step()

            if is_main_process():
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss_reduced.item())

                # if batch_idx % self.log_step == 0:
                #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                #         epoch,
                #         self._progress(batch_idx+1),
                #         loss_reduced.item()))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            pbar.set_description('Train Epoch: {} {} '.format(epoch,self._progress(batch_idx+1)))

            pbar.set_postfix(train_loss=loss_reduced.item())
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            self.valid_metrics.reset()
            outputs, targets = self._valid_epoch(epoch)
            synchronize()
            # outputs = self._accumulate_predictions_from_multiple_gpus(outputs)
            # targets = self._accumulate_predictions_from_multiple_gpus(targets)

            if is_main_process():
                # outputs = [output.to(self.device) for output in outputs]
                # targets = [target.to(self.device) for target in targets]
                # loss = self.criterion(output[:, :], target)
                #
                # loss_reduced = self.reduce_loss(loss)
                # self.valid_metrics.update('loss', loss_reduced.item())
                # outputs = torch.cat(outputs,dim=0)*self.val_mean +self.val_mean
                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(torch.cat(outputs,dim=0), torch.cat(targets,dim=0)))
                val_log = self.valid_metrics.result()
                log.update(**{'val_'+k : v for k, v in val_log.items()})

            if self.output_attn:

                if not os.path.exists('attns'):
                    os.makedirs('attns')
                print(os.path.join('attns', 'attns_of_Epoch{}.npy'))
                np.save(os.path.join('attns', 'attns_of_Epoch{}.npy'.format(epoch)), attns.cpu().numpy())

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        outputs = []
        targets = []
        attns = []
        with torch.no_grad():
            pbar = tqdm(enumerate(self.valid_data_loader),
               total=int(len(self.valid_data_loader.sampler) / self.valid_data_loader.batch_size) + 1, leave=False)
            for batch_idx, (data, target) in pbar:

                for key, value in data.items():
                    if torch.is_tensor(value):
                        data[key] = value.to(self.device)
                for key, value in target.items():
                    if torch.is_tensor(value):
                        target[key] = value.to(self.device)
            # target = target.to(self.device)


                output,reconstructed_out = self.model(data)
                output = output*self.val_var + self.val_mean
                target =  target['label']*self.val_var + self.val_mean
                # output = self.data_loader.dataset.aq_scaler.inverse_transform(output)
                # target = self.data_loader.dataset.aq_scaler.inverse_transform(target['label'])
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # outputs.append(output.clone()[:, :])
                # targets.append(target.clone())
                # attns.append(attn[0])
        pbar.set_description('Val Epoch: {} {} '.format(epoch, self._progress(batch_idx + 1)))

        return None,None #torch.cat(outputs,dim=0), torch.cat(targets,dim=0),# torch.cat(attns,dim=0)
    def invert_trans(self,data):
        B,L,C = data.size()
        return self.valid_data_loader.dataset.scaler.inverse_transform(data.reshape(B*L,C)).reshape(B,L,C)


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
