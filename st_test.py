import os
import argparse
import torch
from tqdm import tqdm
import data_loader as module_data
import criterion as module_loss
import evaluation.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
import numpy as np
from utils.dist import synchronize, get_rank, all_gather
from utils import inf_loop, MetricTracker

def main(args, config):
    logger = config.get_logger('test')

    # prepare for (multi-device) GPU testing
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        synchronize()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, args=args)

    device = torch.device("cuda")
    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if update BatchNorm stats
            # broadcast_buffers=False,
        )

    # save file flag
    save_to_disk = get_rank() == 0

    # setup data_loader instances
    data_loader = config.init_obj('train_loader', module_data, args=args)
    mean = torch.Tensor(data_loader.dataset.scaler.mean_).to(device)
    var = torch.sqrt(torch.Tensor(data_loader.dataset.scaler.var_).to(device))


    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if save_to_disk:
        logger.info(model)
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    if distributed:
        state_dict = {'module.' + k:v for k,v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint['state_dict']
    del state_dict['enc_embedding.spatial_embedding.pe']
    model.load_state_dict(state_dict, strict=False)

    # prepare model for testing
    model.eval()
    total_loss = 0.0
    outputs = []
    targets = []
    total_metrics = torch.zeros(len(metric_fns)).to(device)
    valid_metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns])

    period_conting = []
    period_weight_conting=[]
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(data_loader),
                                    total=int(len(data_loader.sampler) / data_loader.batch_size) + 1, leave=False):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            for key, value in target.items():
                if torch.is_tensor(value):
                    target[key] = value.to(device)
            
            output,reconstructed_out   = model(data)
            output = output*var + mean
            target =  target['label']*var + mean
            # outputs.append(output.clone())
            # targets.append(target.clone())
            
            for met in metric_fns:
                valid_metrics.update(met.__name__, met(output,target))
            # period_conting.append(period_list)
            # period_weight_conting.append(period_weight)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size

        # period_conting = np.array(period_conting)
        # period_weight_conting = np.array(period_weight_conting)

        # unique, count = np.unique(period_conting, return_counts=True)
        # perioddata_count = dict(zip(unique, count))

        synchronize()
        total_loss_list = all_gather(total_loss)
        # outputs_list = all_gather(outputs)
        # targets_list = all_gather(targets)
        log = {'loss': sum(total_loss_list) / len(data_loader.dataset)}
        if save_to_disk:
            # outputs = [output.to(device) for outputs in outputs_list for output in outputs]
            # targets = [target.to(device) for targets in targets_list for target in targets]
            # outputs = [output.to(device) for output in outputs]
            # targets = [target.to(device) for target in targets]
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(torch.cat(outputs,dim=0), torch.cat(targets,dim=0))
            for met in metric_fns:
                # valid_metrics.update(met.__name__, met(torch.cat(outputs,dim=0), torch.cat(targets,dim=0)))
                val_log = valid_metrics.result()
                log.update(**{'val_'+k : v for k, v in val_log.items()})

    
    # log.update({
    #     met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)
    # })
    if save_to_disk:
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-l', '--local_rank', default=0, type=int,
                      help='local rank of gpu')
    args.add_argument('-s', '--save_dir', default=None, type=str,
                      help='dir of save path')
    args.add_argument('--seed', type=int, default=None, help='Random seed.')
    args.add_argument('--deterministic', action='store_true',
        help='Whether to set deterministic options for CUDNN backend.')
    args.add_argument('--debug', action='store_true',
                      help='Whether not to evaluate the checkpoint during training.')
    args.add_argument('--seq_len', type=int, default=72, help='input sequence length')
    args.add_argument('--label_len', type=int, default=72, help='start token length')
    args.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
    args, config = ConfigParser.from_args(args, training=False)
    
    # fix random seeds for reproducibility
    if not args.seed is None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    assert not config.resume is None, "Testing mode requires model path!"
    main(args, config)
