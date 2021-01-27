import os
import re

import torch
import torch.utils.data
import multiprocessing

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from rnn.CharDataset import CharDataset
from rnn.Encoder import Encoder


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--embedding-size', type=int, default=256)
    parser.add_argument('--rnn-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sequence-size', type=int, default=32)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--log-dir', default="./log/", type=str)
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--chars', default="./log/chars.pk", type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.02)
    return parser.parse_args()


def checkpoint(model, directory, file):
    path = os.path.join(directory, file)
    torch.save(model, path)


def get_checkpoint_model(model_path):
    model = torch.load(model_path)
    model.eval()

    m = re.search(r'(\d+).pth', model_path)

    return model, int(m.group(1))


def main():
    args = get_args()

    # If the number of workers wasn't set, use all available ones
    if args.workers is None:
        args.workers = multiprocessing.cpu_count()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cudnn.fastest = True

    dataset = CharDataset.from_file(args.input, args.sequence_size)
    dataset.serialize(args.chars)

    epoch = 0

    # If the use has provided an existing model then use that as a checkpoint.
    if args.model is not None:
        model, epoch = get_checkpoint_model(args.model)
    else:
        model = Encoder(
            dataset.get_vocab_size(), args.embedding_size, args.rnn_size, args.num_layers, args.dropout
        )

    model = model.to(device)
    model.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    writer = SummaryWriter(args.log_dir)

    for epoch in range(epoch, args.epochs):

        running_loss = 0.0

        state_hidden = model.zero_state(args.batch_size).to(device)

        print('Starting epoch {}...'.format(epoch + 1))

        for index, (train, labels) in enumerate(data_loader):

            if len(train) != args.batch_size or len(labels) != args.batch_size:
                continue

            train = train.to(device)
            labels = labels.to(device)

            predict, state = model(train, state_hidden)

            loss = criterion(predict.transpose(1, 2), labels)

            state_hidden = state.detach()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimiser.step()

            # Sum the loss from this batch into the checkpoint total
            running_loss += loss.item()

            if index % args.checkpoint == args.checkpoint - 1:
                writer.add_scalar(
                    'training loss', running_loss / args.batch_size, epoch * len(data_loader) + index
                )

                checkpoint(model, args.log_dir, "{}_{}.pth".format(epoch, index))
                running_loss = 0.0

        checkpoint(model, args.log_dir, "{}.pth".format(epoch + 1))

    checkpoint(model, args.log_dir, "final.pth")

    writer.close()


if __name__ == '__main__':
    main()
