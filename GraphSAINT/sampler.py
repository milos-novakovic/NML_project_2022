import math
import os
import time
import torch as th
import random
import numpy as np
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces

class SAINTSampler(object):
    def __init__(self, dn, g, train_nid, node_budget, num_repeat=50):
        """
        Construciton of the GCNLayer


        Parameters
        ----------            
        dn : str
            Dataset name
            
        g : DGLGraph
            DGLGraph Graph data structure
        
        train_nid : int, optional
            id-s of the training nodes
        
        node_budget : int
            expected number of sampled nodes
        
        num_repeat : int, optional
            number of times of repeating sampling one node
        
        Returns
        -------
        None
        """

        # save arguments as part of the Class fields
        self.g = g
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_repeat = dn, num_repeat
        
        # initialize node and edge counters
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None

        graph_fn, norm_fn = self.__generate_fn__()
        
        # if the sub-sampled graph is already available load it;
        # if it is not, generate it
        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            # make directory subgraphs
            os.makedirs('./subgraphs/', exist_ok=True)
            
            # init list of subgraphs
            self.subgraphs = []
            
            # init the counters
            self.N, sampled_nodes = 0, 0

            t = time.perf_counter()
            while sampled_nodes <= self.train_g.num_nodes() * num_repeat:
                # sample the subgraph
                subgraph = self.__sample__()
                
                # save the sampled subgraph
                self.subgraphs.append(subgraph)
                
                # increment the counters
                sampled_nodes += subgraph.shape[0]
                self.N += 1
            
            # print out sampling time
            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            
            # save the sub-sampled graph
            np.save(graph_fn, self.subgraphs)
            
            # create the normalized version
            t = time.perf_counter()
            
            # count number of sampled nodes and edges per subgraphs
            self.__counter__()
            
            # 
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f'Normalization time: [{time.perf_counter() - t:.2f}s]')
            np.save(norm_fn, (aggr_norm, loss_norm))
        
        # cast loss and weights to torch tensor
        self.train_g.ndata['l_n'] = th.Tensor(loss_norm)
        self.train_g.edata['w'] = th.Tensor(aggr_norm)
        
        # compute degree norm
        self.__compute_degree_norm()

        self.num_batch = math.ceil(self.train_g.num_nodes() / node_budget)
        random.shuffle(self.subgraphs)
        
        # clear the values of the sampler (probability mass function, node_counter, edge_counter, graph g)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))
        print("The size of subgraphs is about: ", len(self.subgraphs[-1]))

    def __clear__(self):
        """
        Clear all the atributes of SAINTsampler class
        
        Parameters
        ----------            
        None
        
        Returns
        -------
        None
        """
        self.prob = None
        self.node_counter = None
        self.edge_counter = None
        self.g = None

    def __counter__(self):
        """
        Count number of sampled nodes and edges per subgraphs
        
        Parameters
        ----------            
        None
        
        Returns
        -------
        None
        """
        for sampled_nodes in self.subgraphs:
            sampled_nodes = th.from_numpy(sampled_nodes)
            self.node_counter[sampled_nodes] += 1

            subg = self.train_g.subgraph(sampled_nodes)
            sampled_edges = subg.edata[dgl.EID]
            self.edge_counter[sampled_edges] += 1

    def __generate_fn__(self):
        # this is abstract class and can not generate any function (sub-classes implement this functionallity)
        raise NotImplementedError

    def __compute_norm__(self):
        """
        Computes 
        
        Parameters
        ----------            
        None
        
        Returns
        -------
        np.array, np.array
        """
        self.node_counter[self.node_counter == 0] = 1
        self.edge_counter[self.edge_counter == 0] = 1

        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()

        self.train_g.ndata['n_c'] = self.node_counter
        self.train_g.edata['e_c'] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'a_n'))
        aggr_norm = self.train_g.edata.pop('a_n')

        self.train_g.ndata.pop('n_c')
        self.train_g.edata.pop('e_c')

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):
        """
        Compute the degree matrix for training matrix and full matrix (from in degrees)
        
        Parameters
        ----------            
        None
        
        Returns
        -------
        None
        """
        # training degree matrix
        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        
        # full degree matrix
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __sample__(self):
        # this is abstract class and can not generate any function (sub-classes implement this functionallity)
        raise NotImplementedError

    def __len__(self):
        """
        Return the length of the batch size (maximum value of the iterator)
        
        Parameters
        ----------            
        None
        
        Returns
        -------
        int
        """
        return self.num_batch

    def __iter__(self):
        """
        Initialization of n in some loop
        
        Parameters
        ----------            
        None
        
        Returns
        -------
        int
        """
        
        self.n = 0
        return self

    def __next__(self):
        """
        Increment of n in some loop
        
        Parameters
        ----------            
        None
        
        Returns
        -------
        DGLGraph
        """
        if self.n < self.num_batch:
            result = self.train_g.subgraph(self.subgraphs[self.n])
            self.n += 1
            return result
        else:
            random.shuffle(self.subgraphs)
            raise StopIteration()


class SAINTNodeSampler(SAINTSampler):
    def __init__(self, node_budget, dn, g, train_nid, num_repeat=50):
        """
        Construciton of the SAINT Node Sampler

        Parameters
        ----------    
        node_budget : int
            expected number of sampled nodes

        dn : str
            Dataset name

        g : DGLGraph
            DGLGraph Graph data structure
        
        train_nid : int, optional
            id-s of the training nodes
        
        num_repeat : int, optional
            number of times of repeating sampling one node
        
        Returns
        -------
        None
        """
        # save the number of roots and length as class fields
        self.node_budget = node_budget
        
        # call the constructor of the super-class
        super(SAINTNodeSampler, self).__init__(dn, g, train_nid, node_budget, num_repeat)

    def __generate_fn__(self):
        """
        Generator function implemented in the sub-class of the super-class SAINTSampler
        (that returns the relative file paths of that particualar sampling)

        Parameters
        ----------
        None

        Returns
        -------
        str, str
        """
        graph_fn = os.path.join('/content/drive/MyDrive/data/NML_Final_Project_GraphSAINT/subgraphs/{}_Node_{}_{}.npy'.format(self.dn, self.node_budget,
                                                                       self.num_repeat))
        norm_fn = os.path.join('/content/drive/MyDrive/data/NML_Final_Project_GraphSAINT/subgraphs/{}_Node_{}_{}_norm.npy'.format(self.dn, self.node_budget,
                                                                           self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        """
        Sample function implemented in the sub-class of the super-class SAINTSampler
        (that returns the relative sampled nodes as np.array)

        Parameters
        ----------
        None

        Returns
        -------
        np.array
        """
        if self.prob is None:
            self.prob = self.train_g.in_degrees().float().clamp(min=1)
        
        # sample nodes acording to the probability mass function self.prob
        # (the number of samples in self.node_budget)
        # sample with replacement, but keep only the unique nodes (i.e. no duplicates at the end)
        sampled_nodes = th.multinomial(self.prob, num_samples=self.node_budget, replacement=True).unique()
        return sampled_nodes.numpy()


class SAINTEdgeSampler(SAINTSampler):
    def __init__(self, edge_budget, dn, g, train_nid, num_repeat=50):
        """
        Construciton of the SAINT Edge Sampler

        Parameters
        ----------    
        edge_budget : int
            expected number of sampled edges

        dn : str
            Dataset name

        g : DGLGraph
            DGLGraph Graph data structure
        
        train_nid : int, optional
            id-s of the training nodes
        
        num_repeat : int, optional
            number of times of repeating sampling one node
        
        Returns
        -------
        None
        """
        # save the number of roots and length as class fields
        self.edge_budget = edge_budget
        
        # call the constructor of the super-class
        super(SAINTEdgeSampler, self).__init__(dn, g, train_nid, edge_budget * 2, num_repeat)

    def __generate_fn__(self):
        """
        Generator function implemented in the sub-class of the super-class SAINTSampler
        (that returns the relative file paths of that particualar sampling)

        Parameters
        ----------
        None

        Returns
        -------
        str, str
        """
        graph_fn = os.path.join('./subgraphs/{}_Edge_{}_{}.npy'.format(self.dn, self.edge_budget,
                                                                       self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_Edge_{}_{}_norm.npy'.format(self.dn, self.edge_budget,
                                                                           self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        """
        Sample function implemented in the sub-class of the super-class SAINTSampler
        (that returns the relative sampled edges as np.array)

        Parameters
        ----------
        None

        Returns
        -------
        np.array
        """
        if self.prob is None:
            src, dst = self.train_g.edges()
            src_degrees, dst_degrees = self.train_g.in_degrees(src).float().clamp(min=1),\
                                       self.train_g.in_degrees(dst).float().clamp(min=1)
            self.prob = 1. / src_degrees + 1. / dst_degrees

        #print(f"self.prob = {self.prob}")
        #print(f"self.edge_budget = {self.edge_budget}")
        th.save(self.prob, 'prob.pt')


        #here we have a run time error
        #RuntimeError: number of categories cannot exceed 2^24
        #sampled_edges = th.multinomial(self.prob, num_samples=self.edge_budget, replacement=True).unique()        
        
        # sample edges acording to the probability mass function self.prob / (1.0* self.prob.sum())
        # (the number of samples in self.edge_budget)
        # sample with replacement, but keep only the unique nodes (i.e. no duplicates at the end)
        sampled_edges = th.from_numpy(
                          np.random.choice(
                              a = np.arange(len(self.prob)),
                              size = self.edge_budget,
                              replace = True,
                              p = self.prob / (1.0* self.prob.sum())
                          )
                        )
        
        # find sampled source and sampled destination edges
        sampled_src, sampled_dst = self.train_g.find_edges(sampled_edges)
        
        # concatinate the sampled source and sampled destination edges into sampled nodes (and return those nodes)
        sampled_nodes = th.cat([sampled_src, sampled_dst]).unique()
        return sampled_nodes.numpy()


class SAINTRandomWalkSampler(SAINTSampler):
    def __init__(self, num_roots, length, dn, g, train_nid, num_repeat=50):
        """
        Construciton of the SAINT Random Walk Sampler

        Parameters
        ----------    
        num_roots : int
            expected number of sampled edges

        length : int
            expected number of sampled edges

        dn : str
            Dataset name

        g : DGLGraph
            DGLGraph Graph data structure
        
        train_nid : int, optional
            id-s of the training nodes
        
        num_repeat : int, optional
            number of times of repeating sampling one node
        
        Returns
        -------
        None
        """
        # save the number of roots and length as class fields
        self.num_roots, self.length = num_roots, length
        
        # call the constructor of the super-class
        super(SAINTRandomWalkSampler, self).__init__(dn, g, train_nid, num_roots * length, num_repeat)

    def __generate_fn__(self):
        """
        Generator function implemented in the sub-class of the super-class SAINTSampler
        (that returns the relative file paths of that particualar sampling)

        Parameters
        ----------
        None

        Returns
        -------
        str, str
        """
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                            self.length, self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        """
        Sample function implemented in the sub-class of the super-class SAINTSampler
        (that returns the relative sampled random walks as np.array)

        Parameters
        ----------
        None

        Returns
        -------
        np.array
        """
        # sample nodes uniformly
        sampled_roots = th.randint(0, self.train_g.num_nodes(), (self.num_roots, ))
        
        # from DGL take sampling method called random_walk 
        # generate random walk traces from an array of starting nodes based on the given metapath
        traces, types = random_walk(g = self.train_g, nodes=sampled_roots, length=self.length)
        
        # pack the padded traces returned by random_walk() into a concatenated array
        # the padding values (-1) are removed, 
        # and the length and offset of each trace is returned along with the concatenated node ID and node type arrays.
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        
        # take only unique nodes (if there are duplicates we dont want to sample same nodes more than once)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()