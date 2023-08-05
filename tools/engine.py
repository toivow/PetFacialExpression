import torch
from torch.nn.functional import one_hot

def metrics(y_pred, y):

    predicted = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    acc = torch.sum(predicted == y)/len(predicted)

    return acc


def epoch(model, data_loader, device, optimizer, loss_fn, train):

    acc = 0
    running_loss = 0
    for (x, y) in data_loader:

        y = y.to(device)
        x = x.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        acc += metrics(y_pred, y)

    return running_loss/len(data_loader), acc*100/len(data_loader)


def train_epoch(model, data_loader, device, optimizer, loss_fn):

    loss, acc = epoch(model, data_loader, device, optimizer, loss_fn, True)
    return loss, acc


@torch.inference_mode()
def val_epoch(model, data_loader, device, optimizer, loss_fn):

    with torch.no_grad():
        loss, acc = epoch(model, data_loader, device, optimizer, loss_fn, False)
        return loss, acc