
import torch
import numpy as np
from tqdm import tqdm
from apex import amp
import warnings
from utils import data_util


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def validation(model, loss_function, epoch_iterator_val, device):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0
    total_loss_last = 0

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            # get model output
            ids, x, y = (batch['id'], batch['x'].to(
                device), batch['y'].to(device))

            x = x.squeeze(0)  # put slided time windows as batch dimension
            y = y.expand(x.size(0))

            logit_map = model(x, device).squeeze(1)

            # calculate validation loss

            loss = loss_function(logit_map, y)
            total_loss += loss

            epoch_iterator_val.set_description(
                "Validation loss %2.5f" % (loss))

            total += 1

    avg_loss = total_loss / total
    avg_loss_last = total_loss_last / total

    return avg_loss, avg_loss_last

# def validation(model, loss_function, epoch_iterator_val, device):
#     model.eval()

#     correct_sub = 0
#     seq_acc_sum = 0
#     total = 0
#     total_loss = 0

#     with torch.no_grad():
#         for step, batch in enumerate(epoch_iterator_val):
#             # get model output
#             ids, x, y = (batch['id'], batch['x'].to(
#                 device), batch['y'].to(device))

#             x = x.squeeze(0)  # put slided time windows as batch dimension
#             y_true = y.item()
#             y = y.expand(x.size(0))

#             logit_map = model(x, device).squeeze(1)

#             # calculate validation loss
#             loss = loss_function(logit_map, y)
#             total_loss += loss

#             epoch_iterator_val.set_description(
#                 "Validation loss %2.5f " % (loss))

#             # calculate sequence accuracy
#             seq_pred = torch.round(torch.sigmoid(logit_map))
#             seq_acc = torch.eq(seq_pred, y).float().mean().item()
#             seq_acc_sum += seq_acc

#             # calculate subject accuracy
#             sub_pred = 1 if torch.sum(
#                 seq_pred == 1) > torch.sum(seq_pred == 0) else 0
#             correct_sub += (sub_pred == y_true)

#             total += 1

#     seq_acc = seq_acc_sum / total       # sequence acc
#     avg_loss = total_loss / total  # sequence loss
#     sub_acc = correct_sub / total

#     return seq_acc, avg_loss, sub_acc


# # def train(model, train_loader, val_loader, loss_function, scheduler, optimizer, device, writer, global_step, args):
# def train(model, val_loader, loss_function, scheduler, optimizer, device, writer, global_step, args):

#     val_loss_best = 10000

#     while global_step < args.num_steps:

#         # get a new train loader each epoch to crop different samples
#         train_loader = data_util.get_train_loader(args)
#         print(
#             f'{len(train_loader)} subjects for training, {len(val_loader)} subjects for testing')

#         model.train()
#         epoch_iterator = tqdm(
#             train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
#         loss_sum = 0.0

#         warnings.filterwarnings('ignore')

#         for step, batch in enumerate(epoch_iterator):
#             id, x, y = (batch['id'], batch['x'].to(
#                 device), batch['y'].to(device))

#             # change out size [4,1] to [4,]

#             logit_map = model(x, device).squeeze(1)

#             # BCE loss
#             loss = loss_function(logit_map, y)
#             loss_sum += loss

#             # if args.amp:
#             #     with amp.scale_loss(loss, optimizer) as scaled_loss:
#             #         scaled_loss.backward()
#             #     torch.nn.utils.clip_grad_norm_(
#             #         amp.master_params(optimizer), args.max_grad_norm)
#             # else:
#             #     loss.backward()

#             loss.backward()
#             # clip norm
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()
#             optimizer.zero_grad()

#             if args.lrdecay:
#                 scheduler.step()

#             epoch_iterator.set_description(
#                 "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss))

#             # calculate loss each every 50 steps
#             loss_step = 50
#             if step % loss_step == (loss_step-1):
#                 writer.add_scalar(
#                     "train/loss", scalar_value=loss, global_step=global_step)
#                 print('average loss for {} steps:{}'.format(
#                     loss_step, loss_sum/loss_step))
#                 loss_sum = 0.0

#             global_step += 1

#             if global_step % args.eval_num == 0 and global_step != 0:
#                 epoch_iterator_val = tqdm(
#                     val_loader, desc="Validation accuracy X.X, Validation loss X.X ", dynamic_ncols=True)
#                 seq_acc, val_loss, sub_acc = validation(
#                     model, loss_function, epoch_iterator_val, device)

#                 writer.add_scalar("Validation Sequence Accuracy",
#                                   scalar_value=seq_acc, global_step=global_step)
#                 writer.add_scalar("Validation Subject Accuracy",
#                                   scalar_value=seq_acc, global_step=global_step)
#                 writer.add_scalar("Validation Loss",
#                                   scalar_value=val_loss, global_step=global_step)

#                 if val_loss < val_loss_best:
#                     checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
#                                   'optimizer': optimizer.state_dict()}
#                     save_ckp(checkpoint, args.savepath + '/model.pt')
#                     val_loss_best = val_loss
#                     print('Model Was Saved ! Current Best Validation Loss: {}  Current Loss {}  Current Sub ACC: {} Current Sequence ACC {}'.format(
#                         val_loss_best, val_loss, sub_acc, seq_acc))
#                 else:
#                     print('Model Was NOT Saved ! Current Best Validation Loss: {}  Current Loss {}  Current Sub ACC: {} Current Sequence ACC {}'.format(
#                         val_loss_best, val_loss, sub_acc, seq_acc))
#                 model.train()

#     return global_step, val_loss_best

def train(model, val_loader, loss_function, scheduler, optimizer, device, writer, global_step, args):

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    val_loss_best = 10000
    epoch = 0

    while global_step < args.num_steps:

        # get a new train loader each epoch to crop different samples
        train_loader = data_util.get_train_loader(args)
        print(
            f'{len(train_loader)} subjects for training, {len(val_loader)} subjects for testing')

        model.train()
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        loss_sum = 0.0

        warnings.filterwarnings('ignore')
        epoch_step = 0

        for step, batch in enumerate(epoch_iterator):

            id, x, y = (batch['id'], batch['x'].to(
                device), batch['y'].to(device))

            if args.amp:
                # Runs the forward pass with autocasting
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass (e.g., compute the loss)
                    logit_map = model(x, device).squeeze(1)
                    loss = loss_function(logit_map, y)

                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)

                # Optimizer step and updates the scale for next iteration
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                # Forward pass
                logit_map = model(x, device)
                loss = loss_function(logit_map, y)

                # Backward pass without AMP
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_sum += loss

            if args.lrdecay:
                scheduler.step()
                # after_lr = optimizer.param_groups[0]["lr"]
                # print(after_lr)

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss))

            # calculate loss each every epoch
            # loss_step = 100
            # if step % loss_step == (loss_step-1):
            #     writer.add_scalar(
            #         "train/loss", scalar_value=loss, global_step=global_step)
            #     print('average loss for {} steps:{}'.format(
            #         loss_step, loss_sum/loss_step))
            #     loss_sum = 0.0
            epoch_step = step
            global_step += 1

            if global_step % args.eval_num == 0 and global_step != 0:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validation loss X.X, Last time point loss X.X ", dynamic_ncols=True)
                val_loss, val_loss_last = validation(
                    model, loss_function, epoch_iterator_val, device)

                writer.add_scalar("Validation Loss",
                                  scalar_value=val_loss, global_step=global_step)
                writer.add_scalar("Last time point Loss",
                                  scalar_value=val_loss_last, global_step=global_step)

                if val_loss < val_loss_best:
                    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, args.savepath + '/model.pt')
                    val_loss_best = val_loss
                    print('Model Was Saved ! Current Best Validation Loss: {}  Current Loss {} '.format(
                        val_loss_best, val_loss))
                else:
                    print('Model Was NOT Saved ! Current Best Validation Loss: {}  Current Loss {}'.format(
                        val_loss_best, val_loss))
                print('Current Best Validation Loss: {}  Current Loss {} '.format(
                    val_loss_best, val_loss))

            if global_step % 10000 == 0:
                checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                print(f'Model Was Saved on {global_step}!')
                save_ckp(checkpoint, args.savepath +
                         f'/model_{global_step}.pt')

        # calculate loss each every epoch
        epoch += 1
        writer.add_scalar(
            "train/loss", scalar_value=loss_sum/(epoch_step+1), global_step=global_step)
        print('average loss for epoch {} :{}'.format(
            epoch, loss_sum/(epoch_step+1)))
        loss_sum = 0.0

    return global_step, val_loss_best
