#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 23:00:00 2021

@author: frhag
"""

import numpy as np
import trimesh
import pyrender
import cv2
import os

def loadModel(model_name_def, camera_matrix, image_size):
  fuze_trimesh = trimesh.load( model_name_def )
  fuze_trimesh = fuze_trimesh.apply_scale(0.001)

  mesh = pyrender.Mesh.from_trimesh(fuze_trimesh, smooth=False)

  r = pyrender.OffscreenRenderer(image_size[0], image_size[1])

  scene = pyrender.Scene()
  scene.add(mesh)

  camera = pyrender.IntrinsicsCamera(
      fx = camera_matrix[0,0],
      fy = camera_matrix[1,1],
      cx = camera_matrix[0,2],
      cy = camera_matrix[1,2],
  )

  nc = pyrender.Node(camera=camera)
  scene.add_node(nc)

  return scene, nc, r

def computeSingleCheck(object_detection, orig_depth, scene_def, nc_def, r_def, depth_scale, back_dist, acc_dist):

    new_object_detection = object_detection[:]

    x_rot = np.array([
      [1.0, 0,   0,   0.0],
      [0.0,  -1.0, 0.0, 0.0],
      [0.0,  0,   -1.0,   0.0],
      [0.0,  0.0, 0.0, 1.0],
    ])

    # Read object pose and set camera
    obj_pose = object_detection[0].copy() #
    obj_pose[:3,3] *= 0.001
    camera_pose = np.dot(np.linalg.inv(obj_pose), x_rot)

    # Update camera position
    scene_def.set_pose(nc_def, pose=camera_pose)

    # Render the scene
    synth_depth = r_def.render(scene_def, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)

    # First calculate object mask from synth depth image
    depth_mask_orig = np.float32(synth_depth != 0)
    kernel = np.ones((3,3), np.uint8) # TODO

    depth_mask = np.where( ( synth_depth - (depth_scale*orig_depth/1000.0)) < back_dist, depth_mask_orig, np.zeros_like(depth_mask_orig) )
    depth_mask_edge = cv2.dilate(depth_mask, kernel)
    depth_mask = cv2.erode(depth_mask, kernel)

    # Calculate the depth comparison
    depth_count = np.where( depth_mask == 1, np.abs(np.float32(depth_scale*orig_depth)/1000.0 - np.float32(synth_depth)) < acc_dist, np.zeros_like(depth_mask) )
    new_object_detection[1]["depth_count"] = np.sum(depth_count)/np.sum(depth_mask)

    return new_object_detection
