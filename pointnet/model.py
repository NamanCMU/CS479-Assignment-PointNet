import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        print("[Model] Device Before: ", device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        ### Input -> self.stn3 -> MLP(64,64) -> self.stn64 -> MLP(64,128,1024) -> Max pool
        self.conv1a = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv1b = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))

        self.conv2a = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv2b = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        print("[PN Feature] PC shape: ", pointcloud.shape)
        x = pointcloud
        if self.input_transform:
            x = self.stn3(pointcloud)
            x = torch.bmm(x, pointcloud)
            print("[PN Feature] STN3 output shape: ", x.shape)
        x = F.relu(self.conv1a(x)) 
        x = F.relu(self.conv1b(x))
        print("[PN Feature] MLP First output shape: ", x.shape)
        if self.feature_transform:
            y = self.stn64(x)
            x = torch.bmm(y, x)
            print("[PN Feature] STN64 output shape: ", x.shape)
        x = F.relu(self.conv2a(x)) 
        x = F.relu(self.conv2b(x))
        print("[PN Feature] MLP Second output shape: ", x.shape)
        x = torch.max(x, dim=2, keepdim=False)[0]
        print("[PN Feature] Max pooling x shape: ", x.shape)

        return x
        ###


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256))
        self.fc3 = nn.Linear(256, num_classes)
        ###

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        x = self.pointnet_feat(pointcloud)
        print("[CLS] First: ", x.shape)

        x = F.relu(self.fc1(x))
        print("[CLS] Second: ", x.shape)
        x = F.relu(self.fc2(x))
        print("[CLS] Third: ", x.shape)
        x = self.fc3(x)
        print("[CLS] Final: ", x.shape)

        return x
        ###


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.stn3 = STNKd(k=3)
        self.stn64 = STNKd(k=64)

        self.input_transform = True 
        self.feature_transform = True 

        self.conv1a = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv1b = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))

        self.conv2a = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv2b = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.conv3a = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512))
        self.conv3b = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256))
        self.conv3c = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128))

        self.conv4a = nn.Sequential(nn.Conv1d(128, m, 1), nn.BatchNorm1d(m))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        x = pointcloud
        print("[PN Feature] PC shape: ", pointcloud.shape)
        if self.input_transform:
            x = self.stn3(pointcloud)
            x = torch.bmm(x, pointcloud)
            print("[PN Feature] STN3 output shape: ", x.shape)
        x = F.relu(self.conv1a(x)) 
        x = F.relu(self.conv1b(x))
        print("[PN Feature] MLP First output shape: ", x.shape)
        if self.feature_transform:
            y = self.stn64(x)
            x = torch.bmm(y, x)
            print("[PN Feature] STN64 output shape: ", x.shape)
        xdash = x
        x = F.relu(self.conv2a(x)) 
        x = F.relu(self.conv2b(x))
        print("[PN Feature] MLP Second output shape: ", x.shape)
        x = torch.max(x, dim=2, keepdim=False)[0]
        print("[PN Feature] Max pooling x shape: ", x.shape)

        # Seg network now
        x_expanded = x.unsqueeze(2).repeat(1, 1, 2048)
        print("[PN Part seg] X expanded shape: ", x_expanded.shape)
        x = torch.cat([xdash, x_expanded], dim=1)
        print("[PN Part seg] concatenated output: ", x.shape)
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv3c(x))
        print("[PN Part seg] MLP Third output shape: ", x.shape)
        x = F.relu(self.conv4a(x))
        print("[PN Part seg] Final output score shape: ", x.shape)

        return x
        ###


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.num_points = num_points
        self.fc1 = nn.Sequential(
            nn.Linear(1024, num_points // 4),
            nn.BatchNorm1d(num_points // 4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_points // 4, num_points // 2),
            nn.BatchNorm1d(num_points // 2),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(num_points // 2, num_points),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_points),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(num_points, num_points * 3)
        ###

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        print("[AE] Zero: ", pointcloud.shape)
        x = self.pointnet_feat(pointcloud)
        print("[AE] First: ", x.shape)
        x = self.fc1(x)
        print("[AE] Second: ", x.shape)
        x = self.fc2(x)
        print("[AE] Third: ", x.shape)
        x = self.fc3(x)
        print("[AE] Fourth: ", x.shape)
        x = self.fc4(x)
        print("[AE] Fifth: ", x.shape)
        x = x.view(-1, self.num_points, 3)
        print("[AE] Sixth: ", x.shape)

        return x
        ###


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
