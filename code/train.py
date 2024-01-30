
import torch
import numpy as np
import tqdm
from apex import amp

def train(model, train_loader, val_loader, mae_val_best, loss_function, scheduler, optimizer, device, writer, args ):
    
    while global_step < args.num_steps:
        loss_sum = 0.0 

        model.train()
        epoch_iterator = tqdm(train_loader,desc="Training (X / X Steps) (loss=X.X)",dynamic_ncols=True)

        warnings.filterwarnings('ignore')

        for step, batch in enumerate(epoch_iterator):
            x_image, x_label, y = (batch["image"].to(device), batch["label"].to(device), batch["score"])
            # x_image, y = (batch["image"].to(device), batch["score"])
            #transpose list scores
            y = [i.tolist() for i in y]
            y = np.array(y).T.tolist()
            y = torch.Tensor(y).to(device)
            y = y.to(torch.float32)

            x_input = torch.cat((x_image, x_label), dim=1)
            # x_input = x_image
            logit_map = model(x_input).to(torch.float32)


            #L1 loss
            loss = loss_function(logit_map, y)
    
            
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss))
            
            #calculate loss each epoch
            loss_step = 55
            if step % loss_step == (loss_step-1):
                writer.add_scalar("train/loss", scalar_value=loss_sum/loss_step, global_step=global_step)
                print('average loss for {} steps:{}'.format(loss_step,loss_sum/loss_step))
                loss_sum = 0.0


            global_step += 1
            if global_step % args.eval_num == 0 and global_step!=0:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (MAE=X.X)", dynamic_ncols=True)
                metrics = validation(epoch_iterator_val)

                # # #numclass = 5
                # mean_score = metrics[5]  # not mean, sum of mae
                # writer.add_scalar("Validation/Sum SubScore MAE", scalar_value=metrics[5], global_step=global_step)

                mean_score = np.sum(metrics)  # not mean, sum of mae
                # mean_score = metrics[5] + metrics[3] #pred_total and C3
                writer.add_scalar("Validation/Total Score MAE", scalar_value=metrics[5], global_step=global_step)
                writer.add_scalar("Validation/Sum SubScore MAE", scalar_value=metrics[6], global_step=global_step)

                for i in range(5):  
                    writer.add_scalar("Validation/C{} MAE".format(i+1), scalar_value= metrics[i], global_step=global_step)
                

                if mean_score < mae_val_best:
                    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, logdir + '/model.pt')
                    mae_val_best = mean_score
                    print('Model Was Saved ! Current Best Sum MAE: {}  Current MAE: {}'.format(mae_val_best, mean_score))
                else:
                    print('Model Was NOT Saved ! Current Best Sum MAE: {} Current MAE: {}'.format(mae_val_best,mean_score))
        return global_step, mae_val_best
    
    
    
    def validation(epoch_iterator_val):
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_image, val_label, val_scores = (batch["image"].to(device), batch["label"].to(device),batch["score"])
                # val_image, val_scores = (batch["image"].to(device),batch["score"])
                #transpose subscores
                val_scores = [i.tolist() for i in val_scores]
                val_scores = np.array(val_scores).T.tolist()[0]

                # val_scores.append(sum(val_scores[:5]))  #numclass == 5
                val_scores.append((val_scores[5])) #numclass ==6

                val_input = torch.cat((val_image, val_label), dim=1)
                # val_input = val_image
                name = batch["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
                
                val_outputs = model(val_input)
                val_outputs = val_outputs.cpu().numpy().tolist()[0]
                val_outputs.append(sum(val_outputs[:5]))

                y_true.append(val_scores)
                y_pred.append(val_outputs)

                epoch_iterator_val.set_description("Validate (%d / %d Steps) " % (global_step, 10.0))
        
        mae = calcMAE(y_true,y_pred)
        return mae