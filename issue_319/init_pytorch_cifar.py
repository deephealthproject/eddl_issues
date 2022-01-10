import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Flatten(nn.Module):
    def forward(self, x):
        # batch_size = x.shape[0]
        # n_features = -1
        # Use constant values to obtain a simpler ONNX
        batch_size = 100
        n_features = 64
        return torch.reshape(x, (batch_size, n_features))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    current_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        current_samples += data.size(0)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / current_samples))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * test_acc))

    return test_loss, test_acc


def predict(model, device, test_loader):
    model.eval()
    trues = []
    true_confs = []
    preds = []
    pred_confs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            trues.append(target)
            aux_confs, aux_preds = output.max(dim=1)
            preds.append(aux_preds)
            pred_confs.append(aux_confs)
            for i, true_class in enumerate(target):
                true_confs.append(output[i, true_class].item())

    trues = torch.cat(trues).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    pred_confs = torch.cat(pred_confs).cpu().numpy()
    true_confs = np.array(true_confs)

    return trues, preds, true_confs, pred_confs


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output-path', type=str,
                        default="pytorch_conv2D_cifar.onnx",
                        help='Output path to store the onnx file')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'pin_memory': True})

    # Prepare data generators
    transform = transforms.Compose([transforms.ToTensor()])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                                transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, drop_last=False,
                                               shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, drop_last=False,
                                              shuffle=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Save to ONNX file
    dummy_input = torch.randn(args.batch_size, 3, 32, 32, device=device)
    torch.onnx._export(model, dummy_input, f'before_fit_{args.output_path}',
                       keep_initializers_as_inputs=True)

    # Train
    test_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        _, test_acc = test(model, device, test_loader)

    trues, preds, true_confs, pred_confs = predict(model, device, test_loader)
    df = pd.DataFrame({'true': trues, 'pred': preds, 'confidence': pred_confs})
    df = pd.DataFrame({'true': trues,
                       'pred': preds,
                       'true_confidence': true_confs,
                       'pred_confidence': pred_confs})
    df.to_csv("pytorch_results.csv", index=False)

    # Save to ONNX file
    dummy_input = torch.randn(args.batch_size, 3, 32, 32, device=device)
    torch.onnx._export(model, dummy_input, args.output_path,
                       keep_initializers_as_inputs=True)


if __name__ == '__main__':
    main()
