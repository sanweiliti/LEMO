# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import torch
import torch.nn as nn

from smplx.lbs import transform_mat


PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


class PerspectiveCamera(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)    # Adds a persistent buffer to the module
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)     # [bs, 2]

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)    # [bs, 3, 3]

        rotation = nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)  # Adds a parameter to the module, shape [1,3,3],  [[1,0,0],[0,1,0],[0,0,1]]

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)   # [bs, 3]

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)  # all 0

    def forward(self, points):
        device = points.device  # [bs, 118, 3]

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y  # [bs, 2, 2], each batch: [[f_x, 0], [0, f_y]]

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))   # [bs, 4, 4], each batch: I
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)     # [bs, 118, 1]
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)   # [1, 118, 4]

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])   # [1, 118, 4]

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))  # [1, 118, 2]
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) + self.center.unsqueeze(dim=1)
        return img_points   # [1, 118, 2]
