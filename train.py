import os
import re

import pickle
import torch
import torch.utils.data
import multiprocessing

import torch.nn.functional as F

from argparse import ArgumentParser

from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from rnn.CharDataset import CharDataset
from rnn.Encoder import Encoder


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--embedding-size', type=int, default=256)
    parser.add_argument('--rnn1-size', type=int, default=512)
    parser.add_argument('--rnn2-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--log-dir', default="./log/", type=str)
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--chars', default="./log/chars.pk", type=str)
    parser.add_argument('--model', type=str)
    return parser.parse_args()


def get_chars(file):
    with open(file) as input_file:
        content = input_file.read()
        characters = sorted(list(set(content)))
        character_index = {c: i for i, c in enumerate(characters)}
        return content, len(characters), characters, character_index


def checkpoint(model, directory, file):
    path = os.path.join(directory, file)
    torch.save(model, path)


def write_chars(file, vocab_size, characters, character_index):
    with open(file, 'wb') as output_file:
        data = {
            'count': vocab_size,
            'characters': characters,
            'character_index': character_index
        }

        pickle.dump(data, output_file)

def get_checkpoint_model(model_path):
    model = torch.load(model_path)
    model.eval()
    model.train()

    m = re.search('(\d+).pth', model_path)
    
    return model, int(m.group(1)) + 1

def main():
    args = get_args()

    # If the number of workers wasn't set, use all available ones
    if args.workers is None:
        args.workers = multiprocessing.cpu_count()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
         torch.backends.cudnn.fastest = True

    content, vocab_size, characters, character_index = get_chars(args.input)

    # Write the required training data to file.
    write_chars(
        args.chars, vocab_size, characters, character_index
    )

    epoch = 0

    # If the use has provided an existing model then use that as a checkpoint.
    if args.model is not None:
        model, epoch = get_checkpoint_model(args.model)
    else:
        model = Encoder(
            vocab_size, args.embedding_size,  args.rnn1_size, args.rnn2_size
        )

    model = model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    data_loader = torch.utils.data.DataLoader(
        CharDataset(args.batch_size, vocab_size, content, character_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    writer = SummaryWriter(os.path.join(args.log_dir))

    for epoch in range(epoch, args.epochs):

        running_loss = 0.0

        state_hidden1, state_context1 = model.zero_state(args.batch_size, args.rnn1_size) 
        state_hidden2, state_context2 = model.zero_state(args.batch_size, args.rnn2_size)

        state_hidden1 = state_hidden1.to(device)
        state_context1 = state_context1.to(device)

        state_hidden2 = state_hidden2.to(device)
        state_context2 = state_context2.to(device)

        print('Starting epoch {}...'.format(epoch + 1))

        for index, (train, labels) in enumerate(data_loader):

            train = train.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            predict, state1, state2 = model(train, (state_hidden1, state_context1), (state_hidden2, state_context2))
            
            loss = F.cross_entropy(predict, labels.squeeze())

            state_hidden1 = state1[0].detach()
            state_context1 = state1[1].detach()

            state_hidden2 = state2[0].detach()
            state_context2 = state2[1].detach()

            loss.backward()
            optimiser.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # Sum the loss from this batch into the checkpoint total
            running_loss += loss.item()

            if index % args.checkpoint == args.checkpoint - 1:

                writer.add_scalar(
                    'training loss', running_loss, epoch * len(data_loader) + index
                )

                
                checkpoint(model, args.log_dir, "{}_{}.pth".format(epoch, index))
                running_loss = 0.0

        checkpoint(model, args.log_dir, "{}.pth".format(epoch))

    checkpoint(model, args.log_dir, "final.pth")

    writer.close()


if __name__ == '__main__':
    main()
