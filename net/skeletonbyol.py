import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SkeletonBYOL(nn.Module):


    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, 
                 momentum=0.999,mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        m: momentum of updating key encoder (default: 0.999)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.online_encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

        else:
            self.m = momentum
            self.online_encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.target_encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.online_encoder.fc.weight.shape[1]
                self.online_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.BatchNorm1d(dim_mlp),
                                                  nn.ReLU(inplace=True),
                                                  self.online_encoder.fc)
                self.target_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.BatchNorm1d(dim_mlp),
                                                  nn.ReLU(inplace=True),
                                                  self.target_encoder.fc)
                self.online_predictor = nn.Sequential(nn.Linear(feature_dim, dim_mlp),
                                                        nn.BatchNorm1d(dim_mlp),
                                                        nn.ReLU(inplace=True),
                                                        nn.Linear(dim_mlp, feature_dim))    

                # self.online_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, 2*dim_mlp),
                #                                   nn.BatchNorm1d(2*dim_mlp),
                #                                   nn.ReLU(),
                #                                   nn.Linear(2*dim_mlp, dim_mlp))
                # self.target_encoder.fc =  nn.Sequential(nn.Linear(dim_mlp, 2*dim_mlp),
                #                                   nn.BatchNorm1d(2*dim_mlp),
                #                                   nn.ReLU(),
                #                                   nn.Linear(2*dim_mlp, dim_mlp))
                # self.online_predictor =  nn.Sequential(nn.Linear(dim_mlp, 2*dim_mlp),
                #                                   nn.BatchNorm1d(2*dim_mlp),
                #                                   nn.ReLU(),
                #                                   nn.Linear(2*dim_mlp, dim_mlp))      

            for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_1, im_2 = None):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        """

        if not self.pretrain:
            return self.online_encoder(im_1)

        q1 = self.online_predictor(self.online_encoder(im_1))  
        q2 = self.online_predictor(self.online_encoder(im_2))  

        # compute key features
        with torch.no_grad():  # no gradient to targets
            self._momentum_update_target_encoder()  # update the target encoder
            target1 = self.target_encoder(im_1)
            target2 = self.target_encoder(im_2)

        return q1, q2, target1.detach(), target2.detach()
        