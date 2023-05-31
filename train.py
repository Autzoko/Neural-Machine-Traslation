import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import trange, tqdm


def train(args, model, data, loss_fn, eval_data):
    LOG_FILE = "./checkpoints/translation_model.log"
    tb_writer = SummaryWriter('./runs')

    t_total = args.num_epoch * len(data)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    global_step = 0
    total_num_words = total_loss = 0.
    logg_loss = 0.
    logg_num_words = 0.
    val_losses = []
    train_iterator = trange(args.num_epoch, desc='epoch')
    for epoch in train_iterator:
        model.train()
        epoch_iteration = tqdm(data, desc='iteration')
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(epoch_iteration):
            mb_x = torch.from_numpy(mb_x).to(args.device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(args.device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(args.device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(args.device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(args.device).long()

            mb_y_len[mb_y_len <= 0] = 1
            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)
            mb_out_mask = torch.arange(mb_y_len.max().item(), device=args.device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            if (it + 1) % 100 == 0:
                loss_scalar = (total_loss - logg_loss) / (total_num_words - logg_num_words)
                logg_num_words = total_num_words
                logg_loss = total_loss

                with open(LOG_FILE, "a") as fout:
                    fout.write(
                        f'epoch: {epoch}, iter: {it}, loss: {loss_scalar}, learning_rate: {scheduler.get_lr()[0]}\n')

                print(f'epoch: {epoch}, iter: {it}, loss: {loss_scalar}, learning_rate: {scheduler.get_lr()[0]}')

                tb_writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss_scalar, global_step)

        print("Epoch ", epoch, " Train Loss ", total_loss / total_num_words)
        eval_loss = evaluate(args, model, eval_data, loss_fn)
        with open(LOG_FILE, "a") as fout:
            fout.write("===========" * 20)
            fout.write(f"EVALUATE : epoch {epoch}, loss: {eval_loss}\n")
        if len(val_losses) == 0 or eval_loss < min(val_losses):
            print("Best model, val loss: ", eval_loss)
            torch.save(model.state_dict(), "./checkpoints/translate-best.th")
        val_losses.append(eval_loss)


def evaluate(args, model, data, loss_fn):
    model.eval()
    total_num_words = total_loss = 0
    eval_iteration = tqdm(data, desc='eval iteration')
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(eval_iteration):
            mb_x = torch.from_numpy(mb_x).to(args.device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(args.device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(args.device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(args.device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(args.device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=args.device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation Loss ", total_loss / total_num_words)
    return total_loss / total_num_words
