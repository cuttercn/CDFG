import torch
import torch.nn.functional as F
import torch.nn as nn

import opt


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    # 图注意力层
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.alpha = alpha   # leakyrelu激活的参数

        if opt.args.name == "dblp" or opt.args.name == "acm" or opt.args.name == "usps" or opt.args.name == "hhar":
            self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        elif opt.args.name == "pubmed" or opt.args.name == "cite" or opt.args.name == "acm" \
                or opt.args.name == "cora" or opt.args.name == "wiki":
            self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 定义可训练参数，即论文中的W和a
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #初始化

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)
        if opt.args.name == "dblp" or opt.args.name == "acm" or opt.args.name == "usps" or opt.args.name == "hhar":
            self.Tanh = nn.Tanh()
        elif opt.args.name == "pubmed" or opt.args.name == "cite" or opt.args.name == "acm" \
                or opt.args.name == "cora" or opt.args.name == "wiki":
            self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        """
        input：[N,in_features],in_features表示特征向量的维数
        adj:邻接矩阵
        M:转移矩阵，如果存在边，那么等于节点度的倒数，不存在则为零
        """
        h = torch.mm(input, self.W)  # 输入矩阵和权重矩阵相乘
        #前馈神经网络
        attn_for_self = torch.mm(h,self.a_self)       #(N,1)
        attn_for_neighs = torch.mm(h,self.a_neighs)   #(N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs,0,1)  # 对邻居矩阵转置加上自身
        attn_dense = torch.mul(attn_dense,M)  # 点乘
        if opt.args.name == "dblp" or opt.args.name == "acm" or opt.args.name == "usps" or opt.args.name == "hhar":
            attn_dense = self.Tanh(attn_dense)
        elif opt.args.name == "pubmed" or opt.args.name == "cite" or opt.args.name == "acm" \
                or opt.args.name == "cora" or opt.args.name == "wiki":
            attn_dense = self.leakyrelu(attn_dense)            #(N,N)

        #掩码（邻接矩阵掩码）
        zero_vec = -1e12 * torch.ones_like(adj)   # 将没有连接的边置为负无穷
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(adj, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        h_prime = torch.matmul(attention,h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示

        if concat:
            return F.elu(h_prime)  # 激活函数
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'