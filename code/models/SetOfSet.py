import torch
from torch import nn
import math
from models.baseNet import BaseNet
from models.layers import *
from pytorch3d import transforms as py3d_trans



class SetOfSetBlock(nn.Module):
    def __init__(self, d_in, d_out, conf):
        super(SetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size")
        self.use_skip = conf.get_bool("model.use_skip")

        modules = []
        modules.extend([SetOfSetLayer(d_in, d_out), NormalizationLayer()])
        for i in range(1, self.block_size):
            modules.extend([ActivationLayer(), SetOfSetLayer(d_out, d_out), NormalizationLayer()])
        self.layers = nn.Sequential(*modules)

        self.final_act = ActivationLayer()

        if self.use_skip:
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        xl = self.layers(x)
        if self.use_skip:
            xl = self.skip(x) + xl

        out = self.final_act(xl)
        return out


class SetOfSetNet(BaseNet):
    def __init__(self, conf):
        super(SetOfSetNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)

        # Integrate IMU encoding
        self.fourier_encoding = FourierProjection(input_dim=6, nmb=num_feats // 2, scale=10, device='cuda')
        self.pos_encoding = PositionalEncoding(num_feats, max_len=200, device='cuda')
        self.perciever_encoding = IMUEncoder(num_feats)
        self.fusion = FusionGRU(num_feats)
        self.lambda_fc = nn.Sequential(nn.Linear(num_feats, num_feats // 4), nn.ReLU(), nn.Linear(num_feats//4, 1))
        #self.refine = RefineGRU(num_feats)
        self.gru = nn.GRU(input_size=num_feats, hidden_size=num_feats, batch_first=True, num_layers=2)

        # Metric Scale factor
        #self.scale_factor = nn.Parameter(torch.tensor([1.0]))

    def forward(self, data):
        x = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]
        # Cameras predictions
        m_input = x.mean(dim=1) # [m,d_out]

        #### Add the IMU refined poses to the predicted cameras ####
        # Encode M-1 imus
        imu_embeddings = []
        for imu in data.imu_list:
            imu = imu.to(x.device).float().unsqueeze(0)
            imu = self.fourier_encoding(imu)
            imu = self.pos_encoding(imu)
            imu = self.perciever_encoding(imu)[0]
            imu_embeddings.append(imu)
        imu_embeddings = self.gru(torch.cat(imu_embeddings, dim=1))[0] # [1, m-1, d_out]

        # latent_relative_pose = torch.diff(m_input, dim=0) # [M-1, d_out]
        # cam_imu_diff = imu_embeddings[0] - latent_relative_pose
        # imu_out = self.m_net(imu_embeddings) # [m-1, d_m]

        #imu_embeddings = imu_embeddings.mean(dim=1)[0]
        #scale_lambda = self.lambda_fc(imu_embeddings).mean(dim=1)
        #refined_poses = self.refine(m_out, imu_embeddings)
        
        # Fuse IMU and Camera poses
        m_input = self.fusion(m_input, imu_embeddings) + m_input

        # Camera predictions
        m_out = self.m_net(m_input)  # [m, d_m]
        #m_out = torch.cat([m_out[:, :4], m_out[:,-3:]], dim=-1)

        # Extract rotation
        # RTs = py3d_trans.quaternion_to_matrix(refined_poses[:, :4])
        # # Get translation
        # minRTts = refined_poses[:, -3:]
        # # Get camera matrix
        # refined_Ps = torch.cat((RTs, minRTts.unsqueeze(dim=-1)), dim=-1)
        # # Add to predictions
        # pred_cam['IMU_refined_poses'] = refined_Ps

        # Points predictions
        n_input = x.mean(dim=0) # [n,d_out]
        n_out = self.n_net(n_input).T # [n, d_n] -> [d_n, n]

        pred_cam = self.extract_model_outputs(m_out, n_out, data)

        return pred_cam

### IMU ENCODING ###

class FourierProjection(nn.Module):
    def __init__(self, input_dim=2, nmb=256, scale=10, device='cpu'):
        """
        Converts [N, input_dim] inputs to [N, nmb*2] fourier features.
        Works well for encoding low dimensional coordinates eg. (x,y,z) 
        into a easily input representation (avoids spectral bias).
        """
        super(FourierProjection, self).__init__()
        self.b = (torch.randn(input_dim, nmb, device=device) * scale)
        self.pi = 3.14159265359
    def forward(self, v):
        x_proj = torch.matmul((2*self.pi*v).float(), self.b.float())
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20, device='cpu'):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Register as a buffer to avoid being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
	

class IMUEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=4):
        super(IMUEncoder, self).__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.randn((1, d_model)))
        self.MHA = nn.MultiheadAttention(d_model, 2, batch_first=True)

    def forward(self, x):
        B, L, D = x.shape 
        query = self.query.unsqueeze(0).repeat(B,1,1) # [B, 1, D]
        attn_output, attn_output_weights = self.MHA(query, x, x)

        return attn_output, attn_output_weights


#### FUSION NETWORK ####

class FusionGRU(nn.Module):
    def __init__(self, d_model, num_layers=2):
        super(FusionGRU, self).__init__()
        hidden_size = d_model * 3
        self.gru = nn.GRU(input_size=d_model*2, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, d_model)
        self.relu = nn.ReLU()

    def forward(self, pose_encodings, imu_encodings):
        # pose_encodings shape: [B, M, D]
        # imu_encodings shape: [B, M-1, D]
        if pose_encodings.dim() == 2:
            pose_encodings = pose_encodings.unsqueeze(0)
            
        # Pad imu_encodings with zeros at the beginning to match the shape of pose_encodings
        B, M, D = pose_encodings.shape
        zero_pad = torch.zeros((B, 1, D), device=pose_encodings.device, dtype=pose_encodings.dtype)

        padded_imu_encodings = torch.cat([zero_pad, imu_encodings], dim=-2)
        combined_input = torch.cat([pose_encodings, padded_imu_encodings], dim=-1)

        out, _ = self.gru(combined_input)
        out = self.relu((self.fc(out))).squeeze()

        return out
    
class RefineGRU(nn.Module):
    def __init__(self, d_model, num_layers=2):
        super(RefineGRU, self).__init__()
        hidden_size = d_model * 2
        self.gru = nn.GRU(input_size=d_model+7, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size, d_model)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model, 7)

    def forward(self, pose_encodings, imu_encodings):
        # pose_encodings shape: [B, M, D]
        # imu_encodings shape: [B, M-1, D]
        if pose_encodings.dim() == 2:
            pose_encodings = pose_encodings.unsqueeze(0)
            
        # Pad imu_encodings with zeros at the beginning to match the shape of pose_encodings
        B, M, D = imu_encodings.shape
        zero_pad = torch.zeros((B, 1, D), device=pose_encodings.device, dtype=pose_encodings.dtype)

        padded_imu_encodings = torch.cat([zero_pad, imu_encodings], dim=-2)
        combined_input = torch.cat([pose_encodings, padded_imu_encodings], dim=-1)

        out, _ = self.gru(combined_input)
        out = self.relu((self.fc1(out)))
        out = self.fc2(out).squeeze()

        return out