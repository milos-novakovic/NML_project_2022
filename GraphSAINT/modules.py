import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, order=1, act=None,
                 dropout=0, batch_norm=False, aggr="concat"):
        """
        Construciton of the GCNLayer


        Parameters
        ----------            
        in_dim : int
            Number of input features in the GCNLayer
            
        out_dim : int
            Number of output features in the GCNLayer
        
        order : int, optional
            Number of linear layers in GCNLayer (usually 1)
        
        act : function pointer or None, optinal
            Activation function at the end of GCNLayer
        
        dropout : float, optional
            Probability of the dropout rate that could be used in GCNLayer
        
        batch_norm : bool, optinal
            Indicator of using the betch normalization layer
        
        aggr : str, optional
            Aggregation function (usually "mean" or "concat")

        Returns
        -------
        None
        """
        
        super(GCNLayer, self).__init__()
        # add the nn Module list so that with can attach linear layers
        self.lins = nn.ModuleList()
        
        # all the parameters for biases
        self.bias = nn.ParameterList()
        
        # order = depth of the network
        for _ in range(order + 1):
            #
            self.lins.append(nn.Linear(in_dim, out_dim, bias=False))
            self.bias.append(nn.Parameter(th.zeros(out_dim)))
        
        # save arguments as part of the Class fields
        self.order = order
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        
        # if batch norm is true create batch norm layer
        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(order + 1):
                self.offset.append(nn.Parameter(th.zeros(out_dim)))
                self.scale.append(nn.Parameter(th.ones(out_dim)))
        
        # save arguments as part of the Class fields
        self.aggr = aggr
        
        # initialize parameters of the GCNLayer
        self.reset_parameters()

    def reset_parameters(self):
        """
        For every module in the nn.ModuleList() initialize it with Xavier_normal initialization

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for lin in self.lins:
            nn.init.xavier_normal_(lin.weight)

    def feat_trans(self, features, idx):
        """
        Feature transformation

        Parameters
        ----------
        features : th.tensor
            Feature representation tensor at layer idx
            
        idx : int
            Number of the idx-th layer in the linear and bias list of parameters

        Returns
        -------
        th.tensor
        """
        # linear feature transformation (here the dimensions could be changed)
        h = self.lins[idx](features) + self.bias[idx]
        
        # non-linear feature transformation (with activation function)
        if self.act is not None:
            h = self.act(h)
        
        # batch normalization is apply to the transformed feature tensor
        if self.batch_norm:
            mean = h.mean(dim=1).view(h.shape[0], 1)
            var = h.var(dim=1, unbiased=False).view(h.shape[0], 1) + 1e-9
            h = (h - mean) * self.scale[idx] * th.rsqrt(var) + self.offset[idx]

        return h

    def forward(self, graph, features):
        """
        Forward Propagation

        Parameters
        ----------
        graph : DGLGraph
            Graph that is propagated in the GCNLayer
            
        features : th.tensor
            Feature representation tensor

        Returns
        -------
        th.tensor
        """
        # return a graph object for usage in a local function scope
        g = graph.local_var()
        
        # apply dropout to features
        h_in = self.dropout(features)
        
        # initialize the feature list with input feautre vector (where the dropout was applied)
        h_hop = [h_in]
        
        # return a node data view for setting/getting node features (either train_D_norm or full_D_norm)
        D_norm = g.ndata['train_D_norm'] if 'train_D_norm' in g.ndata else g.ndata['full_D_norm']
        
        # for every order generate and store one hop feature vector h_hop
        for _ in range(self.order):
            # set and get feature ???h??? for a graph of a single node type.
            g.ndata['h'] = h_hop[-1]
            
            # if the graph is unweighted (make it weighted with ones)
            if 'w' not in g.edata:
                g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
            
            # send messages along all the edges of the specified type
            # and update all the nodes of the corresponding destination type
            
            # u_mul_e = builtin message function that computes a message on an edge 
            # by performing element-wise mul between features of u and e 
            # if the features have the same shape;
            #otherwise, it first broadcasts the features to a new shape and performs the element-wise operation.

            g.update_all(message_func = fn.u_mul_e(lhs_field = 'h', rhs_field = 'w', out ='m'),
                         reduce_func = fn.sum(msg ='m', out = 'h'))
            
            # get that particular feature representation
            h = g.ndata.pop('h')
            
            # multiply it with normalized adjacency matrix
            h = h * D_norm
            
            # save the newly computer hop feature representation
            h_hop.append(h)
        
        # push every feature tensor ft and it's corresponding idx into feature transform
        h_part = [self.feat_trans(ft, idx) for idx, ft in enumerate(h_hop)]
        
        # type of aggregation function
        if self.aggr == "mean":
            h_out = h_part[0]
            for i in range(len(h_part) - 1):
                h_out = h_out + h_part[i + 1]
        elif self.aggr == "concat":
            h_out = th.cat(h_part, 1)
        else:
            raise NotImplementedError
        
        return h_out


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, arch="1-1-0",
                 act=F.relu, dropout=0, batch_norm=False, aggr="concat
        """
        Construciton of the GCNNet


        Parameters
        ----------            
        in_dim : int
            Number of input features in the GCNLayer
            
        hid_dim : int
            Number of hidden features in the GCNLayer
        
        out_dim : int
            Number of output features in the GCNLayer
        
        arch : str, optional
            Architecture of the GCN Netowrk (for example "1-1-0"; and orders then becomes orders = [1,1,0])
        
        act : function pointer or None, optinal
            Activation function at the end of GCNLayer
        
        dropout : float, optional
            Probability of the dropout rate that could be used in GCNLayer
        
        batch_norm : bool, optinal
            Indicator of using the betch normalization layer
        
        aggr : str, optional
            Aggregation function (usually "mean" or "concat")

        Returns
        -------
        None
        """
        
        super(GCNNet, self).__init__()
        
        # add the nn Module list so that with can attach linear layers
        self.gcn = nn.ModuleList()
        
        # make orders from architectur (arch variable)
        orders = list(map(int, arch.split('-')))
        
        # append first GCNLayer to the list of nn.ModuleList-s
        self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, order=orders[0],
                                 act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        
        # number of neurons in a previous layer
        # [if aggregation is concatinate then have it multiply the number of hid_dim by the orders]
        pre_out = ((aggr == "concat") * orders[0] + 1) * hid_dim
        
        # append other layers (for more orders than 1) into the GCNNet (same procedure)
        # it is from orders[1], ... orders[len(orders)-2]
        for i in range(1, len(orders)-1):
            self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[i],
                                     act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
            pre_out = ((aggr == "concat") * orders[i] + 1) * hid_dim
        
        # make the last orders[len(orders)-1] output GCNLayer of the whole GCNNet network
        self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[-1],
                                 act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        pre_out = ((aggr == "concat") * orders[-1] + 1) * hid_dim
        
        # make an output GCNLayer of the whole GCNNet network
        self.out_layer = GCNLayer(in_dim=pre_out, out_dim=out_dim, order=0,
                                  act=None, dropout=dropout, batch_norm=False, aggr=aggr)

    def forward(self, graph):
        """
        Forward Propagation

        Parameters
        ----------
        graph : DGLGraph
            Graph that is propagated in the GCNNet

        Returns
        -------
        th.tensor
        """
        # get the feature representations out of graph data
        h = graph.ndata['feat']
        
        # do a forward propagation for every layer of GCNNet
        for layer in self.gcn:
            h = layer(graph, h)
        
        # normalize output with max(l2-norm of feature representation tensors)
        h = F.normalize(h, p=2, dim=1)
        
        # propagati in the last output layer
        h = self.out_layer(graph, h)

        return h

