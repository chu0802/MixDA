from evaluation import evaluation

def train_dann(args, dloaders, model, optimizer, lr_scheduler, logging=False):
    model.train()
    src_dloader, tgt_train_dloader, tgt_test_dloader = dloaders
    iter_src, iter_tgt = iter(src_dloader), iter(tgt_train_dloader)
    
    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        if i % args.eval_interval == 0:
            c_acc = evaluation(tgt_test_dloader, model)
            print('\nmodel acc: %.2f%%' % (100*c_acc))
            
            if logging:
                model.save(epoch=i)
                model.logger.add_scalar('Testing Accuracy', c_acc, i)
            model.train()
        
        (sx, sy), (tx, _) = next(iter_src), next(iter_tgt)
        sx, sy = sx.cuda(), sy.cuda()
        tx = tx.cuda()
        
        clf_loss, domain_loss = model(sx, tx, sy)
        loss = clf_loss + args.transfer_loss_weight * domain_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()

def train_source_only(args, dloaders, model, optimizer, lr_scheduler, logging=False):
    model.train()
    src_dloader, _, tgt_test_dloader = dloaders
    iter_src = iter(src_dloader)

    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        if i % args.eval_interval == 0:
            c_acc = evaluation(tgt_test_dloader, model)
            print('\nmodel acc: %.2f%%' % (100*c_acc))
            
            if logging:
                model.save(epoch=i)
                model.logger.add_scalar('Testing Accuracy', c_acc, i)
            model.train()
        
        sx, sy = next(iter_src)
        sx, sy = sx.cuda(), sy.cuda()
        clf_loss = model(sx, sy)

        optimizer.zero_grad()
        clf_loss.backward()
        optimizer.step()
        lr_scheduler.step()
