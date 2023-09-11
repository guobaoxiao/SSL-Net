import torch
import torch.nn as nn
from loss import batch_episym
from torch.nn.parameter import Parameter



#------------------
class excavate_feature(nn.Module):    #

    def __init__(self, channel=128,G=8): #pro
        super().__init__()
        self.G=G
        self.channel=channel
        #
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.cweight1 = Parameter(torch.zeros(1, channel , 1, 1))
        self.cbias1 = Parameter(torch.ones(1, channel , 1, 1))
        self.sweight2 = Parameter(torch.zeros(1, channel , 1, 1))
        self.sbias2 = Parameter(torch.ones(1, channel , 1, 1))

        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))  #
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 1)),
            nn.InstanceNorm2d(channel),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, (1, 1)),
            nn.InstanceNorm2d(channel),
            nn.BatchNorm2d(channel),
        )
        self.conv2 = nn.Conv2d(channel , channel , kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
        self.shot_cut = nn.Conv2d(channel , channel , kernel_size=1)

        self.conv3 = nn.Conv2d(channel // (2 * G), channel // (2 * G), kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(channel // (2 * G))


    @staticmethod
    def c_s(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        # -----------------------   struct_H,W
        b, c, h, w = x.size()
        x1 = x
        x_h = self.pool_h(x1)  #
        x_w = self.pool_w(x1)  # 【

        x_w = x_w.permute(0, 1, 3, 2)  #
        # -------------------

        # ---------------

        y1 = torch.cat((x_h, x_w), dim=2)  #

        y1 = self.relu(self.bn1(self.conv2(y1)))  #

        x_h, x_w = torch.split(y1, [h, w],
                               dim=2)  # x_h:1 torc

        x_w = x_w.permute(0, 1, 3, 2)  # x_w t
        # -------------------
        x_h = self.cweight1 * x_h + self.cbias1
        x_w = self.sweight2 * x_w + self.sbias2
        # ---------------

        attention_h = self.sigmoid(self.conv2(x_h))  #
        attention_w = self.sigmoid(self.conv2(x_w))  #
        #------------------------
        out1 = x1 * attention_h * attention_w
        # -----------struct_2
        x_init = x
        x2 = out1.view(b * self.G, -1, h, w)  #
        # channel_split
        x_0, x_1 = x2.chunk(2,dim=1)  #


        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*
        #x_channel = self.cweight * x_channel
        x_channel = self.conv3(x_channel)
        x_channel = x_0 * self.sigmoid(x_channel)
        # spatial attention
        x_spatial = self.gn(x_1)  # bs
        #x_spatial = self.sweight * x_spatia
        x_spatial = self.conv3(x_spatial)

        x_spatial = x_1 * self.sigmoid(x_spatial)  #
        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs
        out = out.contiguous().view(b, -1, h, w)  # contigu

        # -----------struct_3

        out2 = self.c_s(out, 2)

        # ------------end_result
        out3 = self.conv1(out2) + x_init

        return torch.relu(out3)
#-------------------------

def feature_space_search(x, k): # feature space
    inner = -2*torch.matmul(x.transpose(2, 1), x) #in
    xx = torch.sum(x**2, dim=1, keepdim=True) #xx[32
    pairwise_distance = -xx - inner - xx.transpose(2, 1) #distance[32,20
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size
    return idx[:, :, :]

def get_SADgraph_feature(x, k=10, idx=None):
    #--------------
    batch_size = x.size(0)
    num_points = x.size(2)

    #-------------------------------
    x = x.view(batch_size, -1, num_points) #
    if idx is None:
        idx_out = feature_space_search(x, k=k) #
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base #

    idx = idx.view(-1) #
    #----------------------------------------
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() #x
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) #
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) #
    out = torch.cat((x, (x - feature)-torch.abs(x-feature)), dim=3).permute(0, 3, 1, 2).contiguous() #
    return out

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
    # x_in: bat
    mask = logits[:, 0, :, 0] #[32,
    weights = logits[:, 1, :, 0] #[3

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

class DGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv_annular = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), #
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)), #[3
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv_annular = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), #[
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)), #[3
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        #feature[32,128,2000,1]
        B, _, N, _ = features.shape
        out = get_SADgraph_feature(features, k=self.knn_num)
        out = self.conv_annular(out) #out[32,128,2000,1]
        return out

class GCN_Block(nn.Module):  # FROM GCN
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1) #w[32,
        A = torch.bmm(w.transpose(1, 2), w) #A[3
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size() #B=3
        with torch.no_grad():
            A = self.attention(w) #A[3
            I = torch.eye(N).unsqueeze(0).to(x.device).detach() #I[1,20
            A = A + I #A[
            D_out = torch.sum(A, dim=-1) #D_o
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D) #D[3
            L = torch.bmm(D, A)
            L = torch.bmm(L, D) #L[3
        out = x.squeeze(-1).transpose(1, 2).contiguous() #out[32
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous() #out

        return out

    def forward(self, x, w):

        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out

class SC_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=9, s_r=0.5):
        super(SC_Block, self).__init__()
        #-------------
        self.predict = predict
        self.sr = s_r
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        #---------------------
        self.conv_inti_cor = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  #
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        #---------------------
        self.gcn = GCN_Block(self.out_channel)

        self.feature_embed_0 = nn.Sequential(
            excavate_feature(self.out_channel, self.out_channel, G=8),
            excavate_feature(self.out_channel, self.out_channel, G=8),
            excavate_feature(self.out_channel, self.out_channel, G=8),
            excavate_feature(self.out_channel, self.out_channel, G=8),
            DGCNN_Block(self.k_num, self.out_channel),
            excavate_feature(self.out_channel, self.out_channel, G=8),
            excavate_feature(self.out_channel, self.out_channel, G=8),
            excavate_feature(self.out_channel, self.out_channel, G=8),
            excavate_feature(self.out_channel, self.out_channel, G=8)
        )





        self.feature_embed_1 = nn.Sequential(

            excavate_feature(self.out_channel, self.out_channel, G=8)
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:

            self.embed_2=excavate_feature(self.out_channel, self.out_channel,G=8)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)] #indi
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices) #y_out
            w_out = torch.gather(weights, dim=-1, index=indices) #w_out
        indices = indices.view(B, 1, -1, 1) #i

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) #x_
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) #x_out
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1)) #feat
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        #
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()  # cont
        out = self.conv_inti_cor(out)  # out

        out = self.feature_embed_0(out)  # ou
        w0 = self.linear_0(out).view(B, -1)  # w

        out_graph = self.gcn(out, w0.detach())  #
        out = out_graph + out

        out = self.feature_embed_1(out)
        w1 = self.linear_1(out).view(B, -1)  # w1[32,2000]

        if self.predict == False:  #
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1
            indices_1 = indices
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None,
                                                   self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], indices_1
        else:  #
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1
            indices_2 = indices
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)

            out = self.embed_2(out)
            w2 = self.linear_2(out)  # [3
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat, indices_2

class SSLNet(nn.Module):
    def __init__(self, config):
        super(SSLNet, self).__init__()

        self.SC_0 = SC_Block(initial=True, predict=False, out_channel=128, k_num=9,
                             sampling_rate=config.sr)  # sampling_rate=0.5
        self.SC_1 = SC_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        # x[32,1,2000,4],y[32,2000]
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0, indices_1 = self.SC_0(x, y)  #

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)  # 变成0到1的权重[32,1,1000,1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)  # 变成0到1的权重[32,1,1000,1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)  #

        x2, y2, ws1, w_ds1, e_hat, indices_2 = self.SC_1(x_, y1)  # x_[32,1,1000,6],y1[32,1000]

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)  # y_hat对称极线距离  e_hat:[1,9]


        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat, indices_1, indices_2