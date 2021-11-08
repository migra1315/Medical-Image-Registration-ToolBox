import numpy as np
import os, torch
from scipy import interpolate
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
'''
***********************************************
本代码的大部分内容来自 https://github.com/vincentme/GroupRegNet
朱振宇基于上述文档，完成了适用于pairwise的TRE计算方法
纪宇整合相关内容并添加了部分注释
***********************************************
'''
class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image
    
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.grid_dict = {}
        self.norm_coeff_dict = {}

    def forward(self, input_image, flow):   
        '''
        input_image: (n, 1, h, w) or (n, 1, d, h, w)
        flow: (n, 2, h, w) or (n, 3, d, h, w)
        
        return: 
            warped moving image, (n, 1, h, w) or (n, 1, d, h, w)
        '''
        img_shape = input_image.shape[2:]
        if img_shape in self.grid_dict:
            grid = self.grid_dict[img_shape]
            norm_coeff = self.norm_coeff_dict[img_shape]
        else:
            grids = torch.meshgrid([torch.arange(0, s) for s in img_shape]) 
            grid  = torch.stack(grids[::-1], dim = 0) # 2 x h x w or 3 x d x h x w, the data in second dimension is in the order of [w, h, d]
            grid  = torch.unsqueeze(grid, 0)
            grid = grid.to(dtype = flow.dtype, device = flow.device)
            norm_coeff = 2./(torch.tensor(img_shape[::-1], dtype = flow.dtype, device = flow.device) - 1.) # the coefficients to map image coordinates to [-1, 1]
            self.grid_dict[img_shape] = grid
            self.norm_coeff_dict[img_shape] = norm_coeff
        new_grid = grid + flow 

        if self.dim == 2:
            new_grid = new_grid.permute(0, 2, 3, 1) # n x h x w x 2
        elif self.dim == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1) # n x d x h x w x 3
            
        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)

        warped_input_img =  F.grid_sample(input_image, new_grid*norm_coeff - 1. , mode = 'bilinear', align_corners = True, padding_mode = 'border')
        return warped_input_img
        
class CalTRE():
    '''
    根据形变场计算配准后的TRE
    构建一个规范网格，及网格上各点的位移场，根据此采样LandMark的形变场
    '''
    def __init__(self, grid_tuple, disp_f2m):
        self.dim = 3
        self.spatial_transformer = SpatialTransformer(dim = self.dim)
        '''
        STN网络中用到的都是反向映射，即warpped中(x,y,z)处的点来自moving的哪一处
        WARPPED_INT(x,y,x) = MVOING_INT(x-disp_x, y-disp_y, z-disp_z)
        正向映射才是我们计算TRE需要的形变场，反映了mving中的点经过形变场后到了哪一个位置
        '''
        disp_m2f = self.inverse_disp(disp_f2m)
        '''
        一个采样器，给出一个3维网格，和网格上的数据点 -> 也就是各处的形变场
        '''
        self.inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(disp_m2f.detach().cpu().numpy(), 0, -1))

    def inverse_disp(self, disp, threshold = 0.01, max_iteration = 20):
        '''
        compute the inverse field. implementation of "A simple fixed‐point approach to invert a deformation field"

        disp : (2, h, w) or (3, d, h, w)
            displacement field
        '''
        forward_disp = disp.detach().to(device = 'cuda')
        if disp.ndim < self.dim + 2:
            forward_disp = torch.unsqueeze(forward_disp, 0)
        backward_disp = torch.zeros_like(forward_disp)
        backward_disp_old = backward_disp.clone()
        for i in range(max_iteration):
            backward_disp = -self.spatial_transformer(forward_disp, backward_disp)
            diff = torch.max(torch.abs(backward_disp - backward_disp_old)).item()
            if diff < threshold:
                break
            backward_disp_old = backward_disp.clone()
        if disp.ndim < self.dim + 2:
            backward_disp = torch.squeeze(backward_disp, 0)

        return backward_disp

    def cal_disp(self, landmark_moving, landmark_fixed, spacing):
        diff_list = []
        for i in range(300):
            # landmark_moving[i]处的推理形变场pred
            # landmark_moving[i]处的真实形变场gt
            pred = self.inter(landmark_moving[i])
            gt= np.flip((landmark_fixed[i] - landmark_moving[i]),0) # 对应的方向分别为[240,157,83]
            diff_list.append(pred-gt)
        diff_voxel = np.array(diff_list).squeeze(1)
        # 计算300个点对的欧氏距离
        diff = (np.sum(((diff_voxel)*spacing)**2, 1))**0.5
        return np.mean(diff),np.std(diff),diff

def load_data(data_folder,crop_range):
    # 导入数据集，尺寸: [10, 1, 94, 256, 256]/归一化/裁切
    image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('mhd')])
    image_list = []
    image_list = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, file_name))) for file_name in image_file_list]
    input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)

    input_image = (input_image - input_image.min())/(input_image.max() - input_image.min())

    input_image = input_image[:, :, crop_range[0], crop_range[1], crop_range[2]]
    image_shape = input_image.size()[2:] # (d, h, w)
    num_image = input_image.shape[0] # number of image in the group
    return input_image,image_shape,num_image

'''
导入数据
'''
case = 2
crop_range = [slice(5, 98), slice(30, 195), slice(8, 243)]
pixel_spacing = np.array([1.16, 1.16, 2.5], dtype = np.float32)
data_folder = f'/data/JY/Dirlab/case{case}/'
landmark_file = f'/data/JY/Dirlab/case{case}/Case{case}_300_00_50.pt'
states_folder = 'result'
input_image,image_shape,num_image = load_data(data_folder,crop_range)

# 导入标记点，后续计算TRE
landmark_info = torch.load(landmark_file)
landmark_00 = landmark_info['landmark_00']
landmark_50 = landmark_info['landmark_50']

'''
根据对图像的裁切，调整原标志点的坐标位置
'''
grid_tuple = [np.arange(grid_length, dtype = np.float32) for grid_length in image_shape]
landmark_00_converted = np.flip(landmark_00, axis = 1) - np.array([crop_range[0].start, crop_range[1].start, crop_range[2].start], dtype = np.float32)
landmark_50_converted = np.flip(landmark_50, axis = 1) - np.array([crop_range[0].start, crop_range[1].start, crop_range[2].start], dtype = np.float32)

'''
计算形变场，flow是网络推理得到的*反向映射形变场*
'''
flow = torch.randn([3, 93, 165, 235])
calTRE = CalTRE(grid_tuple, flow)
mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
print(mean,std)