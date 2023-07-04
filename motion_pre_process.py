import sys
import torch
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
import torch.nn.functional as F
import importlib
from torchvision.transforms import ToTensor, ToPILImage, Resize
from torchvision.utils import save_image
from PIL import Image

import cv2
import numpy
import torch
# import softsplat # the custom softmax splatting layer
# import run

sys.path.append('.')

def warp(tenOne, flow):

    output = softsplat.softsplat(tenIn=tenOne, tenFlow=flow, tenMetric=None, strMode='avg')

    return output

def frame_interpolation(latents, last_attn_1_1_resize, last_attn_2_1_resize, i, device):
    # latent.shape=[2, 4, 2, 64, 64]
    latents_1 = latents[1]
    full_attn = torch.cat([last_attn_1_1_resize,last_attn_2_1_resize],1) > 0.008

    if i%5 == 0:
        one_tensor = torch.ones(4, 2, 64, 64)
        one_tensor_np = np.array(one_tensor.cpu())
        latent = np.array(latents_1.cpu())
        latent = np.reshape(latent,(-1, 2, 64, 64)) # [4, 2, 64, 64]

        # backward_flow = torch.load("/data/prof1/Video-P2P-jsh/jsh/0514/backward_flow_0000.pt")
        # backward_flow = np.reshape(backward_flow,(2, 200, 200))
        # backward_flow = np.resize(backward_flow, (2, 64, 64))

        flow = torch.load('/data/prof1/Video-P2P-jsh/jsh/0514/backward_flow_0000.pt')
        flow_x = flow[0]
        flow_y = flow[1]
        flow_s_x=flow_x**2
        flow_s_y=flow_y**2
        flow_m=(flow_s_y+flow_s_x)**0.5
        flow_m=np.resize(flow_m, (64,64)) # size issue base size reshape maybe

        print("backward_flow.shape = ", backward_flow.shape)
        aaa

        one_tensor_np[:,:1,:,:] = 1
        one_tensor_np[:,1:,:,:] = 0

        latent = torch.tensor(latent).to(device)
        one_tensor_np = torch.tensor(one_tensor_np).to(device)
        attn_motion = one_tensor_np*full_attn.to(device)
        img_warp = warp(latent, attn_motion)
        one_tensor_2 = torch.ones(4, 1, 64, 64).to(device)
        mask_warp = warp(one_tensor_2, attn_motion)

        mask_warp_randn = torch.randn(4, 1, 64, 64).to(device)
        img_warp = mask_warp*img_warp + (1 - mask_warp)*mask_warp_randn
        latents_1 = torch.reshape(img_warp, (1, 4, 2, 64, 64))
        latents[1] = latents_1
        output = latents
    else:
        output = latents

    return output


def motion(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    flow_x = flow[0,:,:]
    flow_y = flow[1,:,:]
    # flow_x = flow[:,:,0]
    # flow_y = flow[:,:,1]
    flow_s_x=flow_x**2
    flow_s_y=flow_y**2
    flow_m=(flow_s_y+flow_s_x)**0.5

    # flow_m = l_1_norm(flow)
    # flow_m = l_infinity_norm(flow)
    
    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)
    return flow_m


def motion_y(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if flow.shape[0] == 2:
        flow = torch.permute(flow,(1,2,0))

    flow_y_mask = flow[:,:,1:] < 0

    flow = flow*flow_y_mask

    # flow_x = flow[:,:,0]
    # flow_y = flow[:,:,1]
    # flow_s_x=flow_x**2
    # flow_s_y=flow_y**2
    # flow_m=(flow_s_y+flow_s_x)**0.5

    # flow_m = l_1_norm(flow)
    flow_m = l_2_norm(flow)
    # flow_m = l_infinity_norm(flow)

    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)

    return flow_m

def motion_x(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if flow.shape[0] == 2:
        flow = torch.permute(flow,(1,2,0))
        
    flow_x_mask = flow[:,:,:1] < 0
    
    flow = flow*flow_x_mask

    # flow_x = flow[:,:,0]
    # flow_y = flow[:,:,1]
    # flow_s_x=flow_x**2
    # flow_s_y=flow_y**2
    # flow_m=(flow_s_y+flow_s_x)**0.5

    # flow_m = l_1_norm(flow)
    flow_m = l_2_norm(flow)
    # flow_m = l_infinity_norm(flow)

    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)

    return flow_m



def l_1_norm(flow):
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    abs_x = np.abs(flow_x)
    abs_y = np.abs(flow_y)
    l_1_norm = abs_x + abs_y

    return l_1_norm

def l_2_norm(flow):
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    flow_s_x=flow_x**2
    flow_s_y=flow_y**2
    l_2_norm=(flow_s_y+flow_s_x)**0.5

    return l_2_norm


#이거는 마스크가 있을때만 사용해야 할듯?
def l_infinity_norm(flow):
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    diff_matrix = torch.abs(flow_x - flow_y)
    max_norm = torch.max(diff_matrix.sum(dim=1))
    l_infinity_norm = np.full((flow.shape[0], flow.shape[1]), max_norm)

    return l_infinity_norm


def comp_global_motion(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    flow_x = flow[0,:,:]
    flow_y = flow[1,:,:]

    flow_x_mean = torch.mean(flow_x)
    flow_y_mean = torch.mean(flow_y)

    comp_flow_x = flow_x - flow_x_mean
    comp_flow_y = flow_y - flow_y_mean

    flow_s_x=comp_flow_x**2
    flow_s_y=comp_flow_y**2
    flow_m=(flow_s_y+flow_s_x)**0.5

    # flow_m = l_1_norm(flow)
    # flow_m = l_infinity_norm(flow)
    
    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)
    return flow_m


def magnitude(w, flow):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    flow_x = flow[0,:,:]
    flow_y = flow[1,:,:]
    flow_s_x=flow_x**2
    flow_s_y=flow_y**2
    flow_m=(flow_s_y+flow_s_x)**0.5
    resize = Resize((w,w))
    flow_m = torch.tensor(flow_m).to(device).unsqueeze(0)
    flow_m = resize(flow_m).unsqueeze(3)
    print(flow_m.shape)
    return flow_m



def template_matching_ncc(src, temp):
    h, w = src.shape[1:3]
    ht, wt = temp.shape[1:3]

    score = np.empty((h-ht+1, w-wt+1))

    src.cpu()

    src = np.array(src.cpu(), dtype="float")
    temp = np.array(temp.cpu(), dtype="float")

    for dy in range(0, h - ht+1):
        for dx in range(0, w - wt+1):
            roi = src[dy:dy + ht, dx:dx + wt]
            num = np.sum(roi * temp)
            den = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(temp ** 2)) 
            if den == 0: score[dy, dx] = 0
            score[dy, dx] = num / den

    return score


def calculate_correlation_score(prompt, attn_map, mag, x, start, end, cur_step, output_folder):
    split_prompt = prompt.split(" ")

    frame_per_one_attention = torch.mean(attn_map[:8], dim=0)
    frame_per_one_attention_np = np.array(frame_per_one_attention.cpu())

    for i in range(1, len(split_prompt)+1):
        image = frame_per_one_attention[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save('/home/jsh/neurips/Video-P2P-combined/jsh/0626/' + "step" + str(cur_step).zfill(2) + "_" + str(split_prompt[i-1]) +'.png')

    mag_ori_np = mag
    mag = mag.squeeze(-1).squeeze(0)
    mag_np = np.array(mag.cpu())

    score_list = []
    for p_idx in range(1, len(split_prompt)+1):
        correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCOEFF_NORMED)
        correlation_score_norm = (correlation_score + 1)/2
        score_list.append(correlation_score_norm)

    for i in range(len(score_list)):
        if cur_step > 0:
            attn_map[:8,:,:,i+1:i+2] = (score_list[i].item() * mag_ori_np) * x /cur_step + attn_map[:8,:,:,i+1:i+2]
            # attn_map[:8,:,:,start:end] = mag_ori_np * x /cur_step + attn_map[:8,:,:,start:end]
            # attn_map[:8,:,:,start:end] = score_list[start].item() * mag_ori_np + attn_map[:8,:,:,start:end]

    return attn_map


def calculate_correlation_score_many_method(prompt, attn_map, mag, x, start, end, cur_step, output_folder):
    split_prompt = prompt.split(" ")

    frame_per_one_attention = torch.mean(attn_map[:8], dim=0)
    frame_per_one_attention_np = np.array(frame_per_one_attention.cpu())

    for i in range(1, len(split_prompt)+1):
        image = frame_per_one_attention[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save('/home/jsh/neurips/Video-P2P-combined/jsh/0626/' + "step" + str(cur_step).zfill(2) + "_" + str(split_prompt[i-1]) +'.png')

    mag_ori_np = mag
    mag = mag.squeeze(-1).squeeze(0)
    mag_np = np.array(mag.cpu())

    score_list = []
    for p_idx in range(1, len(split_prompt)+1):
        # cv2.TM_SQDIFF 일치: 0 / 불일치 : 255
        correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_SQDIFF)
        correlation_score_norm = 1 - (correlation_score/255)

        # # cv2.TM_SQDIFF_NORMED / 위에 것을 0~1로 정규화
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_SQDIFF_NORMED)
        # correlation_score_norm = 1 - (correlation_score)

        # # cv2.TM_CCORR / 일치: 255 / 불일치 : 0
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCORR)
        # correlation_score_norm = correlation_score/255

        # # cv2.TM_CCORR_NORMED / 위에거 norm
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCORR_NORMED)
        # correlation_score_norm = correlation_score
        
        # # cv2.TM_CCOEFF / 일치 : 255 / 불일치 : 0
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCOEFF)
        # correlation_score_norm = correlation_score

        #cv2.TM_CCOEFF_NORMED 1: 일치, 0: 불일치, 역일치: -1
        # correlation_score = cv2.matchTemplate(frame_per_one_attention_np[:,:,p_idx], mag_np, cv2.TM_CCOEFF_NORMED)
        # correlation_score_norm = (correlation_score + 1)/2

        score_list.append(correlation_score_norm)


    for i in range(len(score_list)):
        if cur_step > 0:
            attn_map[:8,:,:,i+1:i+2] = (score_list[i].item() * mag_ori_np) * x /cur_step + attn_map[:8,:,:,i+1:i+2]
            # attn_map[:8,:,:,start:end] = mag_ori_np * x /cur_step + attn_map[:8,:,:,start:end]
            # attn_map[:8,:,:,start:end] = score_list[start].item() * mag_ori_np + attn_map[:8,:,:,start:end]

    return attn_map









    