import os
import time

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from dicewars.ai.xpolok03.ml.data import get_test_data_loader, get_train_data_loader
from dicewars.ai.xpolok03.ml.losses import SingleGPULossCompute
from dicewars.ai.xpolok03.ml.model import Network
from dicewars.ai.xpolok03.utils import make_timestamped_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_save_path(save_dir, epoch_num, batch_num):
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, F"model_{epoch_num}_{batch_num}.weights")


class Trainer():

    def __init__(self):
        self.tensorboard_logdir = make_timestamped_dir('tblogs/')
        self.writer = SummaryWriter(log_dir=self.tensorboard_logdir)
        self.batch_indices = {}

    def __del__(self):
        self.writer.close()

    def run_epoch(self, data_iter, model: torch.nn.Module, loss_compute, verbose: bool,
                  log_interval: int, epoch: int, run_type: str, save_model=False, save_dir='models/'):
        model = model.to(device)
        total_loss = 0

        for batch_idx, batch in enumerate(data_iter):
            features, heuristics = batch
            features = features.to(device).float()
            heuristics = heuristics.to(device).float()

            batch_size = features.size(0)

            predicted_logits = model.forward(features)
            loss = loss_compute(predicted_logits, heuristics[:, None])
            total_loss += loss
            self.writer.add_scalar(F"Loss/{run_type}_train", loss, self.batch_indices[run_type])

            if verbose and batch_idx % log_interval == 0:
                print("Epoch step: {:5d}/{:5d} Loss: {:5.3f}".format(
                    batch_idx + 1, int(len(data_iter) / batch_size), loss / batch_size))
                if save_model:
                    torch.save(model.state_dict(), prepare_save_path(save_dir, epoch, batch_idx))
            self.batch_indices[run_type] += 1
        return total_loss

    def train(self, model: torch.nn.Module, num_epochs, train_iter, valid_iter, save_dir,
              verbose, log_interval=10, lr=1e-3, lr_decay=0.05):
        self.batch_indices['train'] = 0
        self.batch_indices['valid'] = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

        epoch_start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            train_loss = self.run_epoch(train_iter, model, SingleGPULossCompute(model, criterion, optimizer), verbose,
                           log_interval, epoch, 'train')
            model.eval()
            valid_loss = self.run_epoch(valid_iter, model, SingleGPULossCompute(model, criterion), verbose,
                           log_interval, epoch, 'valid')

            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss: {:.5f} | valid loss: {:.5f}'
                  .format(epoch, time.time() - epoch_start_time, train_loss, valid_loss))
            print('-' * 59)
            epoch_start_time = time.time()
            # lr_scheduler.step()

            torch.save(model.state_dict(), prepare_save_path(save_dir, epoch, 'end'))


if __name__ == "__main__":
    batch_size = 16
    train_data_loader = get_train_data_loader(batch_size=batch_size)
    valid_data_loader = get_test_data_loader(batch_size=batch_size)

    model = Network(input_features=5, output_features=1)
    trainer = Trainer()
    trainer.train(model, 500, train_data_loader, valid_data_loader,
                  'dicewars/ai/xpolok03/ml/models', verbose=False, log_interval=100, lr=0.001)
