import os
import time

import torch.utils.data

from dicewars.ai.ladatron.ml.losses import SingleGPULossCompute

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_save_path(save_dir, epoch_num, batch_num):
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, F"model_{epoch_num}_{batch_num}.weights")


def run_epoch(data_iter, model: torch.nn.Module, loss_compute, verbose: bool, log_interval: int, save_model=False,
              epoch=-1, save_dir='models/'):
    model = model.to(device)
    total_loss = 0

    for batch_idx, batch in enumerate(data_iter):
        features, heuristics = batch
        features = features.to(device)
        heuristics = heuristics.to(device)

        batch_size = features.size(0)

        predicted_logits = model.forward(features)
        loss = loss_compute(predicted_logits, heuristics)
        total_loss += loss

        if verbose and batch_idx % log_interval == 0:
            print("Epoch step: {:5d}/{:5d} Loss: {:5.3f}".format(
                batch_idx + 1, int(len(data_iter) / batch_size), loss / batch_size))
            if save_model:
                torch.save(model.state_dict(), prepare_save_path(save_dir, epoch, batch_idx))


def train(model: torch.nn.Module, num_epochs, train_iter, valid_iter, save_dir, verbose, log_interval=10, lr=1e-3,
          lr_decay=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    epoch_start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        run_epoch(train_iter, model, SingleGPULossCompute(model, criterion, optimizer), verbose,
                  log_interval=log_interval)
        model.eval()
        run_epoch(valid_iter, model, SingleGPULossCompute(model, criterion), verbose,
                  log_interval=log_interval)

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              .format(epoch, time.time() - epoch_start_time))
        print('-' * 59)
        epoch_start_time = time.time()
        lr_scheduler.step()

        torch.save(model.state_dict(), prepare_save_path(save_dir, epoch, 'end'))
