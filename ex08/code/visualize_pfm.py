from typing import List
from PIL import Image

from util import io
import numpy as np
import open3d as o3d

import os
import imageio
import glob
import argparse
import matplotlib.pyplot as plt


class Sample:
    def __init__(self):
        # gt_base_path = "/project/cv-ws2122/shared-data1/monodepth/test_images/scannet_gt/"
        self.gt_base_path = "/home/bechtolj/Desktop/newex08/scannet_gt/"
        self.rgb = None
        self.K_depth = None
        self.K_rgb = None
        self.K_depth_o3d = None
        self.K_rgb_o3d = None
        self.gt_depth = None
        self.pred_disp = None
        self.mask = None
        self.sampleid = ''
        self.pred_depth = None

    def load(self, sampleid):
        scene = os.path.basename(sampleid).split('-')[0]
        img_name = os.path.basename(sampleid).split('.')[0]
        prediction, _ = io.read_pfm(sampleid)
        # intrinsics txt
        intr_path = os.path.join(self.gt_base_path, scene + '-intrinsic_depth.txt')
        intrinsics_txt = np.loadtxt(intr_path)
        intrinsics_small_txt = intrinsics_txt.copy()
        intrinsics_txt *= 2 # double everything for highres
        intrinsics_txt[2,2] = 1.
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.intrinsic_matrix = intrinsics_txt[:3,:3]
        intrinsics_small = o3d.camera.PinholeCameraIntrinsic()
        intrinsics_small.intrinsic_matrix = intrinsics_small_txt[:3,:3]
        # rgb
        rgb_path = os.path.join("input/scannet/", img_name + '.jpg')
        rgb = imageio.imread(rgb_path)
        # gt .png
        gt_path = os.path.join(self.gt_base_path, img_name + '_gt.png')
        gt_depth = imageio.imread(gt_path).astype(np.float32)
        gt_depth /= 1000.0
        mask = gt_depth != 0
        # set things
        self.pred_disp = prediction
        self.gt_depth = gt_depth
        self.K_depth = intrinsics_small_txt[:3,:4]
        self.K_depth_o3d = intrinsics_small
        self.K_rgb = intrinsics_txt[:3,:4]
        self.K_rgb_o3d = intrinsics
        self.rgb = rgb
        self.mask = mask
        self.sampleid = sampleid

    def set_predicted_aligned_depth(self, depth):
        self.pred_depth = depth


def sample_reader() -> List:
    files = glob.glob("output/scannet/*.pfm")
    print(files)
    samples = []
    for f in files:
        s = Sample()
        s.load(f)
        samples.append(s)
    return samples


def pil_resize(array, shape, interpolation="nearest"):
    """ resize using PIL.Image
    Params:
        array: array to rescale
        shape: tuple (w,h) as expected by PIL
        interpolation: nearest for depth, any for color
    Returns:
        resized image as np array
    """
    pilimage = Image.fromarray(array)
    # pil seems to want w,h - numpy gives h,w
    pilimage = pilimage.resize(shape, resample=Image.NEAREST)
    return np.asarray(pilimage)

def point_cloud_from_rgbd(depth, intrinsics,rgb=None):
    depthimage = o3d.geometry.Image(depth)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depthimage, intrinsics,
                                                                depth_scale=1.0, depth_trunc=100.0,)#0.0)
    if rgb is not None:
        rgbimage = o3d.geometry.Image(rgb)
        rgbdimage = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbimage,
                                                                       depthimage,
                                                                       depth_scale=1.0,
                                                                       depth_trunc=100.0,
                                                                       convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbdimage, intrinsics)
    return pcd

def show_pointcloud(depth, intrinsics, usemeshframe=False, rgb=None, name=''):
    pcd = point_cloud_from_rgbd(depth, intrinsics, rgb=rgb)
    R = pcd.get_rotation_matrix_from_xyz([3.1416,0,0])
    pcd = pcd.rotate(R) # rotate around x
    # pcd = pcd.rotate(np.asarray([3.1416,0,0]), center=False) # rotate around x
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1,
        origin=[0,0,0])
    draw = [pcd]
    if usemeshframe:
        draw.append(mesh_frame)
    o3d.visualization.draw_geometries(draw, window_name=name)

def save_pointcloud(filename, depth, intrinsics, rgb=None):
    pcd = point_cloud_from_rgbd(depth, intrinsics, rgb=rgb)
    o3d.io.write_point_cloud(filename, pcd)


def compute_scale_and_shift(prediction, target, mask):
    """ Implements h_opt from the assignment
        You can test with python test: python -m doctest -v visualize_pfm.py
        And you can compare your result against this:
        s,t = np.polyfit(prediction[mask], target[mask], deg=1)
    Args:
        prediction (float np.ndarray hxw): predicted depth image
        target (float np.ndarray hxw): ground truth depth image
        mask (bool np.ndarray hxw): masks the valid pixels
    >>> pred = np.asarray([1,2,3])
    >>> mask = np.asarray([1,1,1])
    >>> "{0[0]:.1f} {0[1]:.1f}".format(compute_scale_and_shift(pred,pred,mask))
    '1.0 0.0'
    >>> pred = np.asarray([1,2])
    >>> gt = np.asarray([5,7])
    >>> "{0[0]:.1f} {0[1]:.1f}".format(compute_scale_and_shift(pred,gt,np.ones_like(pred)))
    '2.0 3.0'
    """
    # START TODO #################
    # 2 x 1 = 2 x 2            @    2 x 1
    # hopt = sum(di @ di.T)^-1 @ sum(di*d_star)
    raise NotImplementedError
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]

    # right hand side: b = [b_0, b_1]

    # solution: ...

    # A needs to be a positive definite matrix.

    # END TODO ###################
    s,t = np.polyfit(prediction[mask], target[mask], deg=1)
    assert f'{s:.3f}' == f'{x_0:.3f}' and f'{t:.3f}' == f'{x_1:.3f}'
    return x_0, x_1


def get_aligned_depth(smpl, visualize=False):
    # ground truth depth and rgb image (therefore also prediction) have a different resolution
    # resize the prediction
    prediction_lowres = pil_resize(smpl.pred_disp,
                                   (smpl.gt_depth.shape[1],
                                    smpl.gt_depth.shape[0]))

    if visualize:
        show_pointcloud(1./smpl.pred_disp, smpl.K_rgb_o3d, name='raw prediction')
        show_pointcloud(smpl.gt_depth, smpl.K_depth_o3d, name='ground truth')

    # only predict s,t for valid pixels
    # prediction is disparity
    # convert gt_depth into disparity
    gt_disp = np.zeros_like(smpl.gt_depth)
    gt_disp[smpl.mask] = 1.0/smpl.gt_depth[smpl.mask]
    s, t = compute_scale_and_shift(prediction_lowres, gt_disp, smpl.mask)

    # large resolution,
    aligned_depth = 1.0/(s*smpl.pred_disp+t)

    # show_pointcloud(smpl.gt_depth, intrinsics)
    if visualize:
        show_pointcloud(aligned_depth, smpl.K_rgb_o3d, name='aligned prediction')
        show_pointcloud(smpl.gt_depth, smpl.K_depth_o3d, name='ground truth')

    outfile = smpl.sampleid.replace('.pfm', '.ply')
    save_pointcloud(outfile, aligned_depth, smpl.K_rgb_o3d, rgb=smpl.rgb)
    smpl.set_predicted_aligned_depth(aligned_depth)


def create_3D_image(depth, rgb, K, K_o3d, gif_name):
    assert depth.shape[:2] == rgb.shape[:2], "rgb and depth must have same shape"
    gt_pcd = point_cloud_from_rgbd(depth, K_o3d, rgb=rgb)
    gif_rgb = []
    # to the right
    for i in range(15):
        rot_rad = (i-(10/2.3))/180. * np.pi
        # rot_rad = (i-angle_range)/180. * np.pi
        R = gt_pcd.get_rotation_matrix_from_xyz([0,rot_rad,0])
        rot_pcd = gt_pcd.rotate(R)
        _, rot_color = pc_to_depth(rot_pcd, K, depth)
        gif_rgb.append((rot_color * 255).astype(np.uint8))
    gif_rgb.extend(gif_rgb[::-1]) # back to start
    save_gif(gif_name, gif_rgb)

def save_gif(filename, frames):
    # for rgb assume this happened before: im = Image.fromarray((x * 255).astype(np.uint8))
    frames = [Image.fromarray(image) for image in frames]
    frame_one = frames[0]
    frame_one.save(f"{filename}.gif", format="GIF", append_images=frames,
               save_all=True, duration=500, loop=0)

def pc_to_depth(gt_pcd, intrinsic_depth, gt_depth):
    """ create the depth image given a pointcloud and the intrinsics
        use K to project points into the image plane
        assemble the rgb values of the pointcloud into a new image """
    points_3d = np.asarray(gt_pcd.points)
    points_3d_color = np.asarray(gt_pcd.colors)

    # START TODO #################
    # project the points
    raise NotImplementedError

    # create a new image

    # assemble the points_3d_color in a new image,
    # using the indices of the projected points

    # END TODO ###################
    return new_depth, new_color

def main():
    samples = sample_reader()
    for sample in samples:
        # TASK 1 + 2
        get_aligned_depth(sample, visualize=True)

        # TASK 3
        # create gif for ground truth depth
        rgb_small = pil_resize(sample.rgb,
            (sample.gt_depth.shape[1], sample.gt_depth.shape[0])
        )
        gif_name = f"{sample.sampleid}-gt"
        create_3D_image(sample.gt_depth, rgb_small, sample.K_depth,
                        sample.K_depth_o3d, gif_name)

        # create gif for predicted depth
        gif_name = f"{sample.sampleid}-pred"
        create_3D_image(sample.pred_depth, sample.rgb,
                        sample.K_rgb, sample.K_rgb_o3d, gif_name)

if __name__=="__main__":
    main()
