import torch

def evaluation(loader, model, args):
    pred, true = [], []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            true.append(y)
            c_pred = model.predict(x)
            pred.append(c_pred.argmax(dim=1, keepdim=True))
            
    pred, true = torch.cat(pred).flatten(), torch.cat(true).flatten()
    acc = (pred == true).float().mean()
    return acc.item()