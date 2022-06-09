import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
from modules import GCNNet
from utils import Logger, evaluate, save_log_dir, load_data


def main(args):
    """
    Main function that starts the training as well as execution of the whole program

    Parameters
    ----------
    args** (see the complete list for more detals)

    Returns
    -------
    None
    """
    # here we are just using "amazon" dataset
    multilabel_data = set(['ppi', 'yelp', 'amazon'])
    multilabel = args.dataset in multilabel_data

    # load and preprocess dataset
    data = load_data(args, multilabel)
    # get the Graph g
    g = data.g
    
    # get train, validation and test node data-points
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    # get node lables
    labels = g.ndata['label']

    # get train node id-s
    train_nid = data.train_nid
    
    # number of input features
    in_feats = g.ndata['feat'].shape[1]
    
    # number of classes (this is a multiclass problem)
    n_classes = data.num_classes
    
    # get number of nodes and edges
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    # get number of training, validation and test samples
    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))

    # load sampler
    if args.sampler == "node":
        subg_iter = SAINTNodeSampler(args.node_budget, args.dataset, g,
                                     train_nid, args.num_repeat)
    elif args.sampler == "edge":
        subg_iter = SAINTEdgeSampler(args.edge_budget, args.dataset, g,
                                     train_nid, args.num_repeat)
    elif args.sampler == "rw":
        subg_iter = SAINTRandomWalkSampler(args.num_roots, args.length, args.dataset, g,
                                            train_nid, args.num_repeat)

    # set device for dataset tensors (because it takes ~30GB of RAM we recommed to do in on a CPU)
    if args.gpu < 0:
        cuda = False
    else:
        # if args.gpu >= 0 it will use cuda as a device (GPU)
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print("features shape:", g.ndata['feat'].shape)
    
    # initialize the model - GCNNet
    model = GCNNet(
        in_dim=in_feats,
        hid_dim=args.n_hidden,
        out_dim=n_classes,
        arch=args.arch,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
        aggr=args.aggr
    )
    
    # if device is cuda transfer the model to the GPU
    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("GPU memory allocated before training(MB)",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)

    start_time = time.time()
    best_f1 = -1
    
    # -------------- TRAINING EPOCHS -------------- 
    for epoch in range(args.n_epochs):
        start_epoch_time = time.time()
        for j, subg in enumerate(subg_iter):
            # sync with upper level training graph
            if cuda:
                subg = subg.to(torch.cuda.current_device())
            
            # set the model into the training state
            model.train()
            
            # forward propagation
            pred = model(subg)
            batch_labels = subg.ndata['label']
            
            # we have a multilabel problem here (in the amazon dataset)
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(pred, batch_labels, reduction='sum',
                                                          weight=subg.ndata['l_n'].unsqueeze(1))
            else:
                loss = F.cross_entropy(pred, batch_labels, reduction='none')
                loss = (subg.ndata['l_n'] * loss).sum()
            
            # reset optimizer grads to zero
            optimizer.zero_grad()
            
            # backward propagation
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            
            # optimizer update step of the model parameters
            optimizer.step()
            
            # end of the current epoch, hence print result
            if j == len(subg_iter) - 1:
                print(f"epoch:{epoch+1}/{args.n_epochs}, Iteration {j+1}/"
                      f"{len(subg_iter)}:training loss", loss.item())
        end_epoch_time = time.time()
        print(f"Finished training epoch = {epoch+1}. The training execution time = {round(end_epoch_time - start_epoch_time,2)} sec.",\
         end = "\n ---------------------------------------------------------------------------------------------------------------- \n")

        start_evaluation_time = time.time()
        
        # evaluate the model after every iteration args.val_every
        if epoch % args.val_every == 0:
            # calculate validation error (micro and macro)
            val_f1_mic, val_f1_mac = evaluate(model, g, labels, val_mask, multilabel)
            print("Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
                
            # if the model now is better save it as the best validation model
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pkl'))

        end_evaluation_time = time.time()
        print(f"Finished evaluating epoch = {epoch+1}. The evaluating execution time = {round(end_evaluation_time - start_evaluation_time,2)} sec.", end = "\n ---------------------------------------------------------------------------------------------------------------- \n")


    end_time = time.time()
    print(f'training using time {end_time - start_time}', end = "\n ---------------------------------------------------------------------------------------------------------------- \n")

    # test the best model and report the test error
    if args.use_val:
        model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pkl')))
    test_f1_mic, test_f1_mac = evaluate(model, g, labels, test_mask, multilabel)
    print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(test_f1_mic, test_f1_mac), end = "\n ---------------------------------------------------------------------------------------------------------------- \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAINT')
    # data source params
    parser.add_argument("--dataset", type=str, choices=['ppi', 'flickr', 'reddit', 'yelp', 'amazon'], default='ppi',
                        help="Name of dataset.")

    # cuda params
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU index. Default: -1, using CPU.")

    # sampler params
    parser.add_argument("--sampler", type=str, default="node", choices=['node', 'edge', 'rw'],
                        help="Type of sampler")
    parser.add_argument("--node-budget", type=int, default=6000,
                        help="Expected number of sampled nodes when using node sampler")
    parser.add_argument("--edge-budget", type=int, default=4000,
                        help="Expected number of sampled edges when using edge sampler")
    parser.add_argument("--num-roots", type=int, default=3000,
                        help="Expected number of sampled root nodes when using random walk sampler")
    parser.add_argument("--length", type=int, default=2,
                        help="The length of random walk when using random walk sampler")
    parser.add_argument("--num-repeat", type=int, default=50,
                        help="Number of times of repeating sampling one node to estimate edge / node probability")

    # model params
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="Number of hidden gcn units")
    parser.add_argument("--arch", type=str, default="1-0-1-0",
                        help="Network architecture. 1 means an order-1 layer (self feature plus 1-hop neighbor "
                             "feature), and 0 means an order-0 layer (self feature only)")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate")
    parser.add_argument("--no-batch-norm", action='store_true',
                        help="Whether to use batch norm")
    parser.add_argument("--aggr", type=str, default="concat", choices=['mean', 'concat'],
                        help="How to aggregate the self feature and neighbor features")

    # training params
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Frequency of evaluation on the validation set in number of epochs")
    parser.add_argument("--use-val", action='store_true',
                        help="whether to use validated best model to test")
    parser.add_argument("--note", type=str, default='none',
                        help="Note for log dir")

    args = parser.parse_args()

    print(args)

    main(args)
