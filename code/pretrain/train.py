
import torch
import numpy as np
from tqdm import tqdm
from apex import amp
import warnings


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def validation(model, loss_function, epoch_iterator_val, device):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            # get model output
            ids, x, y = (batch['id'], batch['x'].to(
                device), batch['y'].to(device))

            logit_map = model(x, device).squeeze(1)

            # calculate validation loss
            loss = loss_function(logit_map, y)
            total_loss += loss

            epoch_iterator_val.set_description(
                "Validation loss %2.5f " % (loss))

            # calculate if the classification is correct
            y_pred = torch.round(torch.sigmoid(logit_map))
            total += y.size(0)
            correct += (y_pred == y).sum().item()

    acc = correct/total
    avg_loss = total_loss / total

    return acc, avg_loss


def train(model, train_loader, val_loader, loss_function, scheduler, optimizer, device, writer, global_step, args):

    val_loss_best = 10000

    while global_step < args.num_steps:

        model.train()
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        loss_sum = 0.0

        warnings.filterwarnings('ignore')

        for step, batch in enumerate(epoch_iterator):
            id, x, y = (batch['id'], batch['x'].to(
                device), batch['y'].to(device))

            # change out size [4,1] to [4,]
            logit_map = model(x, device).squeeze(1)

            # BCE loss
            loss = loss_function(logit_map, y)

            # if args.amp:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(
            #         amp.master_params(optimizer), args.max_grad_norm)
            # else:
            #     loss.backward()

            loss.backward()
            optimizer.step()

            if args.lrdecay:
                scheduler.step()

            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss))

            # calculate loss each every 50 steps
            loss_step = 50
            if step % loss_step == (loss_step-1):
                writer.add_scalar(
                    "train/loss", scalar_value=loss, global_step=global_step)
                print('average loss for {} steps:{}'.format(
                    loss_step, loss_sum/loss_step))
                loss_sum = 0.0

            global_step += 1

            if global_step % args.eval_num == 0 and global_step != 0:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validation accuracy X.X, Validation loss X.X ", dynamic_ncols=True)
                acc, val_loss = validation(
                    model, loss_function, epoch_iterator_val, device)

                writer.add_scalar("Validation Accuracy",
                                  scalar_value=acc, global_step=global_step)
                writer.add_scalar("Validation Loss",
                                  scalar_value=val_loss, global_step=global_step)

                if val_loss < val_loss_best:
                    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, args.savepath + '/model.pt')
                    val_loss_best = val_loss
                    print('Model Was Saved ! Current Best Validation Loss: {}  Current Loss {}  Current ACC: {}'.format(
                        val_loss_best, val_loss, acc))
                else:
                    print('Model Was NOT Saved ! Current Best Validation Loss: {}  Current Loss {}  Current ACC: {}'.format(
                        val_loss_best, val_loss, acc))

    return global_step, val_loss_best
