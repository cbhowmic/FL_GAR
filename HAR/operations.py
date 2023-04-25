import torch
import torch.optim as optim
import torch.nn.functional as F

class FedSGDOptim(optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(FedSGDOptim, self).__init__(params, defaults)

    def step(self, grad_model=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in zip(group['params'], list(grad_model.parameters())):  # (p[0], p[1])
                if p[0].grad is None:
                    continue
                #                 d_p = p[0].grad.data # local model grads
                p[0].data.add_(-group['lr'], p[1].grad.data.clone())
        return loss


def train(args, client, device):
    client['model'].train()
    # print(client['trainset'])

    for epoch in range(1, args.local_rounds):
        for batch_idx, (data, target) in enumerate(client['trainset']):
            # print('data:', data)
            data = data.send(client['hook'])
            target = target.send(client['hook'])
            client['model'].send(data.location)
            data, target = data.to(device), target.to(device)
    #
            client['optim'].zero_grad()
            output = client['model'](data)
            loss = F.nll_loss(output, target)
            # loss = torch.nn.CrossEntropyLoss(output, target)
            loss.backward()
            client['model'].get()
    #
            if batch_idx % args.log_interval == 0:
                loss = loss.get()
                # print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(client['hook'].id, epoch, batch_idx * args.local_batches,
                #                                                 len(client['trainset']) * args.local_batches, 100. * batch_idx / len(client['trainset']), loss.item()))


def loss_and_accuracy(model, device, data_loader, name):
    # model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        # batch_x, batch_y = data_loader.next()
        for (data, target) in iter(data_loader):
            # data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(data_loader.dataset)
    print('\n{} set: Average loss for the global model: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(name, total_loss, correct, len(data_loader.dataset),
                                                                                              100. * correct / len(data_loader.dataset)))

    return total_loss, (correct / len(data_loader.dataset))