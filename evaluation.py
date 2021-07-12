from tqdm import tqdm
import torch
import torch.nn as nn

def prediction(loader, model, args, verbose=False):
    pred, true = [], []
    model.eval()
    with torch.no_grad():
        if verbose:
            pbar = tqdm(desc='Evaluation', total=len(loader))
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            output, _ = model.forward(x)
            pred.append(output.float().detach())
            true.append(y.float())
            if verbose:
                pbar.update(1)
    pred, true = torch.cat(pred), torch.cat(true)
    pred = nn.Softmax(dim=1)(pred)
    _, pred = torch.max(pred, 1)

    return pred.squeeze().detach().cpu(), true.cpu()

def cal_acc(loader, model, args, verbose=False):
    pred, true = prediction(loader, model, args, verbose)
    acc = (pred.float() == true).float().mean()
    return acc.item()

