
import torch
import numpy as np
from tqdm import tqdm
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

            logit_map = model(x, device)

            # calculate validation loss

            loss = loss_function(logit_map, y)
            loss_last = loss_function(logit_map[:, :, -1], y[:, :, -1])
            total_loss += loss
            total_loss_last += loss_last

            epoch_iterator_val.set_description(
                "Validation loss %2.5f Last time point loss %2.5f" % (loss, loss_last))

            total += y.size(0)

    avg_loss = total_loss / total
    avg_loss_last = total_loss_last / total

    return avg_loss, avg_loss_last


# def train(model, train_loader, val_loader, loss_function, scheduler, optimizer, device, writer, global_step, args):
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
                    logit_map = model(x, device)
                    loss = loss_function(logit_map, y)

                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

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
                    model.parameters(), args.max_grad_norm)
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

            if global_step % 5000 == 0:
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
