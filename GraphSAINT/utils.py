import json
import os
from functools import namedtuple
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import dgl
import numpy as np
import torch
from sklearn.metrics import f1_score


class Logger(object):
    def __init__(self, path):
        """
        Initialize the logger.

        Parameters
        ----------
        path : str
        The file path where the output of the logger will be stored.

        Returns
        -------
        None
        """
        self.path = path

    def write(self, s):
        """
        Write operation into the logger

        Parameters
        ----------
        s : obj
        Object that is written after casted by str() function to the logger.

        Returns
        -------
        None
        """
        with open(self.path, 'a') as f:
            f.write(str(s))
        print(s)
        return


def save_log_dir(args):
    """
    The directory (where the logs will be held) specified with args.

    Parameters
    ----------
    args : obj
    Object that is controlled by the mains arguments.

    Returns
    -------
    str
    """
    log_dir = './log/{}/{}'.format(args.dataset, args.note)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def calc_f1(y_true, y_pred, multilabel):
    """
    Calculate the F1 accuracy metric (where avarage is micro and macro)

    Parameters
    ----------
    y_true : torch tensor or np.array
        true labels

    y_pred : torch tensor or np.array
        predicted labels
    
    multilabel : bool
        indicator that indicates the existance of multilabeled dataset
    
    Returns
    -------
    float, float
    """
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
        f1_score(y_true, y_pred, average="macro")


def evaluate(model, g, labels, mask, multilabel=False):
    """
    Evaluate the model based on the Graph g

    Parameters
    ----------
    model : nn.Module
        created model that is going to be evaluated

    g : DGL Graph
        feautre graph (i.e. data) that is used to evaluate the model
    
    labels : torch tensor or np.array
        label list
        
    mask : torch tensor or np.array
        select specific feature representation tensors or labels
     
    multilabel : bool
        indicator that indicates the existance of multilabeled dataset
        
    Returns
    -------
    float, float
    """
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        f1_mic, f1_mac = calc_f1(labels.cpu().numpy(),
                                 logits.cpu().numpy(), multilabel)
        return f1_mic, f1_mac


# load data of GraphSAINT and convert them to the format of dgl
def load_data(args, multilabel):
    
    #prefix = f"C:\\Users\\novakovi\\Downloads\\GraphSaint-master\\GraphSaint-master\\data\\{args.dataset}\\"
    prefix = "/content/drive/MyDrive/data/NML_Final_Project_GraphSAINT/data/" + args.dataset + "/"
    #prefix += "data\\{}\\".format(args.dataset)
    
    DataType = namedtuple('Dataset', ['num_classes', 'train_nid', 'g'])

    # Load full adjacency matrix and create the graph from it
    adj_full = scipy.sparse.load_npz('{}adj_full.npz'.format(prefix)).astype(np.bool)
    g = dgl.from_scipy(adj_full)
    num_nodes = g.num_nodes()

    # Load training adjacency matrix and create the a list of unique node ids
    adj_train = scipy.sparse.load_npz('{}adj_train.npz'.format(prefix)).astype(np.bool)
    train_nid = np.array(list(set(adj_train.nonzero()[0])))

    # Load full role JSON file and load training, validation, and test masks
    role = json.load(open('{}role.json'.format(prefix)))
    mask = np.zeros((num_nodes,), dtype=bool)
    train_mask = mask.copy()
    train_mask[role['tr']] = True
    val_mask = mask.copy()
    val_mask[role['va']] = True
    test_mask = mask.copy()
    test_mask[role['te']] = True

    # Load full feature representation matrix
    feats = np.load('{}feats.npy'.format(prefix))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    # Load class_map JSON file and create the labels
    class_map = json.load(open('{}class_map.json'.format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    if multilabel:
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_nodes, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_nodes,))
        for k, v in class_map.items():
            class_arr[k] = v
    
    # initialize graph node data with features, labels, training mask, validation mask and test mask
    g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    g.ndata['label'] = torch.tensor(class_arr, dtype=torch.float if multilabel else torch.long)
    g.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    g.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    g.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)
    
    # convert the graph into a DataType with specified number of classes and training nodes ids
    data = DataType(g=g, num_classes=num_classes, train_nid=train_nid)
    
    np.save('./in_feats.npy', data.g.ndata['feat'].shape[1])    # .npy extension is added if not given
    np.save('./n_classes.npy', data.num_classes)    # .npy extension is added if not given
    
    return data
