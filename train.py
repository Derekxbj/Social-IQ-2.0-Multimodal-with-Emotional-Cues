import time
import copy
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from torchmetrics.classification import Accuracy
    

def train_model(device, model, model_type, dataloaders, optimizer, scheduler, writer, accum_steps, num_epochs=25):
    since = time.time()
    
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    
    acc_metric = Accuracy(task="multiclass", num_classes=4)
    acc_metric.to(device)

    print("Starting Training of {} model".format(model_type))
    
    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
        
            
            running_loss = 0.0
            running_corrects = 0
            
            avg_loss = 0.
            counter = 0
            
            running_loss_mse = 0.0
            
            for batch in tqdm(dataloaders[phase]):
                batch = {k:v.to(device) for k,v in batch.items()}
                labels = batch["labels"]
                

                
                counter += 1
                         
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs= model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits

                    
                    _, preds = torch.max(logits, 1)
                    
                    
                    if phase == 'train':
                        loss.mean().backward()
                        # Clip the norm of the gradients to 1.0 to prevent the "exploding gradients" problem.
                        torch.nn.utils.clip_grad_norm_(model.roberta.parameters(), 1.0)
                        
                        if counter % accum_steps == 0 or counter == len(dataloaders['train']):
                            optimizer.step()
                            optimizer.zero_grad()
                         
                    
                    avg_loss += loss.mean().item()
                    if counter%100 == 0:
                        print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(dataloaders[phase]),
                                                                                                   avg_loss/counter))
                        writer.add_scalar('Training average loss', avg_loss/counter, counter)
                
                # metrics update        
                running_loss += loss.mean().item() * batch['labels'].size(0)
                running_corrects += torch.sum(preds == labels.data)
                acc_metric.update(preds, labels)
                
                del batch, labels, preds, loss, logits
                torch.cuda.empty_cache()
                                
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            
            # add metrics to tensorboard
            writer.add_scalar(phase+' loss', epoch_loss, epoch)
            # writer.add_scalar(phase+' accuracy', epoch_acc, epoch)
            epoch_acc = acc_metric.compute()
            writer.add_scalar(phase+' accuracy', epoch_acc, epoch)

            
            # reset metrics
            acc_metric.reset()
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # if phase == 'val':
            #     scheduler.step(epoch_acc)  # if use val_acc as metric for ReduceLROnPlateau scheduler
            
            if phase == 'train':
                scheduler.step()  # if use learning rate scheduler
                train_acc.append(epoch_acc.data.cpu().numpy())
                train_loss.append(epoch_loss)
            else:
                val_acc.append(epoch_acc.data.cpu().numpy())
                val_loss.append(epoch_loss)
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} in epoch: {:.0f}'.format(best_acc, best_epoch))

    # save running history to file
    history = np.array([train_acc, train_loss, val_acc, val_loss])
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history