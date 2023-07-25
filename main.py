import os
import torch
import shutil
import logging
import numpy as np
import torch.nn as nn
from argparse import ArgumentParser

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from model import myModel
from train import train_model
from tools import make_logdir, parse_configuration, init_obj
from data_load import generateDataset, DataCollatorForMultipleChoice




os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device = torch.device("cuda")
logger = logging.getLogger(__file__)


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
class CustomDistributedDataParallel(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def run():
    
    args = ArgumentParser()
    args.add_argument('-c', '--config', default='./config.json', type=str,
                    help='config file path (default: None)')
    args.add_argument('--local_rank', type=int, default=-1)
    args.add_argument('--ngpu', type=int, default=1)
    args = args.parse_args()
    config = parse_configuration(args.config)
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', 
                                         init_method='env://', 
                                         world_size=args.ngpu, 
                                         rank=args.local_rank)
    
    
    model_type=config["model"]["type"]
    
    # create save directory and tensorboard writer
    log_dir = make_logdir(model_type)
    writer = SummaryWriter(log_dir)
    shutil.copyfile('./config.json', log_dir+'/config.json')
    shutil.copyfile('./main.py', log_dir+'/main.py')
    shutil.copyfile('./model.py', log_dir+'/model.py')
    shutil.copyfile('./data_load.py', log_dir+'/dataset.py')
    shutil.copyfile('./train.py', log_dir+'/train.py')


    tokenizer = AutoTokenizer.from_pretrained(model_type)

    dataset = generateDataset('../siq2/qa/qa_train.json', '../siq2/qa/qa_val.json', '../siq2/video', tokenizer)

    tokenized_train, tokenized_val = dataset.returnDataset()


    model = myModel.from_pretrained(model_type)
    model.to(device)
    
    for param in model.videomae.parameters():
        param.requires_grad = False
        
    for param in model.wav2vec.parameters():
        param.requires_grad = False
    
    model = CustomDistributedDataParallel(model, 
                                          find_unused_parameters = True,
                                          device_ids=[args.local_rank], 
                                          output_device=args.local_rank)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(tokenized_train)
    # define dataloader
    dataloaders_dict = {'train': DataLoader(tokenized_train,
                                            batch_size=int(config["data_loader"]["args"]["batch_size"]/args.ngpu), 
                                            shuffle=False,
                                            num_workers=int((config["data_loader"]["args"]["num_workers"] + args.ngpu - 1)/args.ngpu),
                                            sampler = train_sampler,
                                            pin_memory=True,
                                            collate_fn=DataCollatorForMultipleChoice(tokenizer)),
                        'val': DataLoader(tokenized_val, 
                                          batch_size=int(config["data_loader"]["args"]["batch_size"]/args.ngpu), 
                                          shuffle=config["data_loader"]["args"]["shuffle"],
                                          num_workers=int((config["data_loader"]["args"]["num_workers"] + args.ngpu - 1)/args.ngpu),
                                          pin_memory=True,
                                          collate_fn=DataCollatorForMultipleChoice(tokenizer)) }
    
    
    optimizer = AdamW(model.parameters(), lr = config["optimizer"]["args"]["lr"], correct_bias=True)
    
    total_steps = len(dataloaders_dict['train']) * config["trainer"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = config["lr_scheduler"]["args"]["num_warmup_steps"], 
                                            num_training_steps = total_steps)
    
    model, hist = train_model(device, model, model_type, dataloaders_dict, optimizer, scheduler, writer,
                              accum_steps=config["trainer"]["accum_steps"], num_epochs=config["trainer"]["epochs"])
    
    if args.local_rank in [-1, 0]:
        np.save(log_dir + '/history.npy', hist)
        torch.save(args, log_dir + '/model_training_args.txt')
        torch.save(model.state_dict(), log_dir + '/model_{}.pth'.format(model_type))
        tokenizer.save_pretrained(log_dir)
        
        writer.add_hparams({'lr': config["optimizer"]["args"]["lr"],
                            'batch_size': config["data_loader"]["args"]["batch_size"], 
                            'model': config["model"]["type"], 
                            'loss' : 'cross_entropy_loss',
                            'optimizer': str(config["optimizer"]),
                            'lr_scheduler': str(config["lr_scheduler"]),
                            'epochs': config["trainer"]["epochs"]},
                        {'hparam/accuracy_tra': max(hist[0]), 
                            'hparam/loss_tra': min(hist[1]), 
                            'hparam/accuracy_val': max(hist[2]),
                            'hparam/loss_val': min(hist[3])})

    
    writer.flush()
    writer.close()
    
if __name__ == '__main__':
    run()