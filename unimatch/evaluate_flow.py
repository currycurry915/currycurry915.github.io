from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from unimatch.dataloader.flow.datasets import FlyingChairs, FlyingThings3D, MpiSintel, KITTI
from unimatch.utils import frame_utils
from unimatch.utils.flow_viz import save_vis_flow_tofile, flow_to_image
import imageio

from unimatch.utils.utils import InputPadder, compute_out_of_boundary_mask
from glob import glob
from unimatch.unimatch.geometry import forward_backward_consistency_check
from unimatch.utils.file_io import extract_video

@torch.no_grad()
def inference_flow(model,
                   inference_dir,
                   inference_video=None,
                   output_path=None,
                   padding_factor=8,
                   inference_size=None,
                   save_flo_flow=False,  # save raw flow prediction as .flo
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   fwd_bwd_consistency_check=False,
                   save_video=False,
                   concat_flow_img=False,
                   ):
    """ Inference on a directory or a video """
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    if save_video:
        assert inference_video is not None

    fixed_inference_size = inference_size
    transpose_img = False

    if inference_video is not None:
        filenames, fps = extract_video(inference_video)  # list of [H, W, 3]
    else:
        filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    print('%d images found' % len(filenames))

    vis_flow_preds = []
    ori_imgs = []
    forward_list = []
    backward_list = []

    for test_id in range(0, len(filenames) - 1):
        if (test_id + 1) % 50 == 0:
            print('predicting %d/%d' % (test_id + 1, len(filenames)))

        if inference_video is not None:
            image1 = filenames[test_id]
            image2 = filenames[test_id + 1]
        else:
            image1 = frame_utils.read_gen(filenames[test_id])
            image2 = frame_utils.read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        if concat_flow_img:
            ori_imgs.append(image1)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)

        if pred_bwd_flow:
            image1, image2 = image2, image1

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             pred_bidir_flow=pred_bidir_flow,
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        forward_list.append(flow)

        fname_list = []
        for filename in filenames:
            fname = os.path.basename(filename)
            fname_list.append(fname[:-4])

        # only else
        if inference_video is not None:
            output_file = os.path.join(output_path,'flow','%04d_flow.jpg' % int(fname_list[test_id]))
        else:
            output_file = os.path.join(output_path, 'flow','%04d.jpg' % int(fname_list[test_id]))

        flow_save_path = os.path.join(output_path, 'flow')

        if not os.path.exists(flow_save_path):
            os.makedirs(flow_save_path)

        if inference_video is not None and save_video:
            vis_flow_preds.append(flow_to_image(flow))
        else:
            # save vis flow
            save_vis_flow_tofile(flow, output_file)

        # also predict backward flow
        if pred_bidir_flow:
            assert flow_pr.size(0) == 2  # [2, H, W, 2]
            flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
            backward_list.append(flow_bwd)
            output_file_ = os.path.join(output_path, 'flow','%04d.jpg' % int(fname_list[-1]))

            # save vis flow
            save_vis_flow_tofile(flow_bwd, output_file_)

            # forward-backward consistency check
            # occlusion is 1
            if fwd_bwd_consistency_check:
                fwd_occ, bwd_occ = forward_backward_consistency_check(flow_pr[:1], flow_pr[1:])  # [1, H, W] float

                if inference_video is not None:
                    fwd_occ_file = os.path.join(output_path, '%04d_occ_fwd.png' % test_id)
                    bwd_occ_file = os.path.join(output_path, '%04d_occ_bwd.png' % test_id)
                else:
                    fwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ_fwd.png')
                    bwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ_bwd.png')

                Image.fromarray((fwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(fwd_occ_file)
                Image.fromarray((bwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(bwd_occ_file)

        if save_flo_flow:
            if inference_video is not None:
                output_file = os.path.join(output_path, '%04d_pred.flo' % test_id)
            else:
                output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_pred.flo')
            frame_utils.writeFlow(output_file, flow)
            if pred_bidir_flow:
                if inference_video is not None:
                    output_file_bwd = os.path.join(output_path, '%04d_pred_bwd.flo' % test_id)
                else:
                    output_file_bwd = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_pred_bwd.flo')
                frame_utils.writeFlow(output_file_bwd, flow_bwd)

    if save_video:
        suffix = '_flow_img.mp4' if concat_flow_img else '_flow.mp4'
        output_file = os.path.join(output_path, os.path.basename(inference_video)[:-4] + suffix)

        if concat_flow_img:
            results = []
            assert len(ori_imgs) == len(vis_flow_preds)

            concat_axis = 0 if ori_imgs[0].shape[0] < ori_imgs[0].shape[1] else 1
            for img, flow in zip(ori_imgs, vis_flow_preds):
                concat = np.concatenate((img, flow), axis=concat_axis)
                results.append(concat)
        else:
            results = vis_flow_preds

        imageio.mimwrite(output_file, results, fps=fps, quality=8)

    print('Done!')
    return forward_list, backward_list
