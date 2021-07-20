from model import Model, lr_scheduler
from evaluation import cal_acc
from tqdm import tqdm

def train_step(num_iter, num_epoches, dloader, model, criterion, args):
    model.train()
    total_loss = 0
    total_length = 0
    for i, (x, y) in tqdm(enumerate(dloader),
                        desc='Iteration %02d/%02d' % (num_iter+1, num_epoches), 
                        total=len(dloader)):
        lr_scheduler(model.optimizer, num_iter, num_epoches)
        x = x.to(args.device)
    
        outputs, _ = model.forward(x)
        loss = criterion(outputs, y)

        total_loss += len(x)*loss.item()
        total_length += len(x)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    return total_loss / total_length

def target_train(dloaders, criterion, args, logging=True):
    model = Model(args, num_classes=args.dataset['num_classes'], logging=logging)
    model.to()

    num_iter = 0
    num_epoches = args.config['train']['target']['num_epoches']
    while num_iter < num_epoches:
        train_loss = train_step(num_iter, num_epoches, dloaders['mix'], model, criterion, args)

        if logging:
            model.logger.add_scalar('Training Loss', train_loss, num_iter)

        if (num_iter + 1) % args.config['train']['store_interval'] == 0:
            model.save(epoch=num_iter)
            # Evaluate the performance
            test_acc = 100*cal_acc(dloaders['target_test'], model, args, verbose=False)

            if logging:
                model.logger.add_scalar('Testing Accuracy', test_acc, num_iter)

        num_iter += 1
    model.save()

def source_train_val(dloaders, criterion, args, logging=True):
    # TODO: load model in any checkpoint
    model = Model(args, num_classes=args.dataset['num_classes'], logging=logging)
    model.to()

    num_iter = 0
    best_val_acc = 0
    best_iter = 0

    num_epoches = args.config['train']['source']['num_epoches']
    while num_iter < num_epoches:
        train_loss = train_step(num_iter, num_epoches, dloaders['source_train'], model, criterion, args)
        acc_val = 100*cal_acc(dloaders['source_val'], model, args)

        if logging:
            model.logger.add_scalar('Training Loss', train_loss, num_iter)
            model.logger.add_scalar('Validation Accuracy', acc_val, num_iter)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_iter = num_iter
        num_iter += 1
    return best_iter+1

def target_unify_train(dloaders, criterion, args, logging=False):
    model = Model(args, num_classes=args.dataset['num_classes']+1, logging=logging)
    model.to()

    num_iter = 0
    best_val_acc = 0
    best_iter = 0

    num_epoches = args.config['train']['source']['num_epoches']

    while num_iter < num_epoches:
        train_loss = train_step(num_iter, num_epoches, dloaders['target_unify_train'], model, criterion, args)
        acc_val = 100*cal_acc(dloaders['target_unify_val'], model, args)

def source_train_full(num_epoches, dloader, criterion, args, logging=True):
    model = Model(args, num_classes=args.dataset['num_classes'], logging=logging)
    model.to()

    print('---------- Full Training ----------')

    for num_iter in range(num_epoches):
        train_loss = train_step(num_iter, num_epoches, dloader, model, criterion, args)

        if logging:
            model.logger.add_scalar('Full Training Loss', train_loss, num_iter)

        if (num_iter+1) % args.config['train']['store_interval'] == 0:
            model.save(epoch=num_iter)
    model.save()
