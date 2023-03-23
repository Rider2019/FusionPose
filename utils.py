import numpy as np
import json
import torch
import cv2

def farthest_point_sample(xyz, npoint):
    ndataset = xyz.shape[0]
    if ndataset<npoint:
        if ndataset == 0:
            ndataset = ndataset+1
        repeat_n = int(npoint/ndataset)
        xyz = np.tile(xyz,(repeat_n,1))
        xyz = np.append(xyz,xyz[:npoint%ndataset],axis=0)
        return xyz
    centroids = np.zeros(npoint)
    distance = np.ones(ndataset) * 1e10
    farthest =  np.random.randint(0, ndataset)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[int(farthest)]
        dist = np.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return xyz[np.int32(centroids)]

def norm_pc(points):
    pc_mean = np.mean(points,0).reshape(1,-1)
    new_pc = points - pc_mean
    return new_pc,pc_mean

def get_camera_parameters(file_name):
    # word2cam: x = ptX, without using r
    with open(file_name) as fp:
        cp = json.load(fp)
        p,r,t = cp['p'],cp['r'], cp['t']
        p,r,t = np.array(p),np.array(r),np.array(t)
    return p,r,t

def world2cam(pc,camera_json = 'camera.json'):
    p,r,t = get_camera_parameters(camera_json)

    # cart2hom
    B,T,K,C = pc.shape
    # n = pc.shape[0]
    # ones = torch.ones((1,1,K,1)).cuda()
    # print(pc.shape,ones.shape)
    # pc_hom = torch.cat((pc,ones),3)

    temp = torch.matmul(pc,torch.tensor(t.T.reshape(1,1,4,4)).float().cuda())
    pts_2d = torch.matmul(temp,torch.tensor(p.T.reshape(1,1,4,3)).float().cuda())

    # pc_proj_to_img
    # pts_2d = np.dot(np.dot(pc_hom,t.T),p.T)
    kp_2d_pre = torch.zeros_like(pts_2d)
    # hom2cart
    kp_2d_pre[...,0] = pts_2d[...,0]/pts_2d[...,2]
    kp_2d_pre[...,1] = pts_2d[...,1]/pts_2d[...,2]

    return kp_2d_pre[...,:2]

def world2cam_one_seq(pc,camera_json = 'camera_new.json'):
    p,r,t = get_camera_parameters(camera_json)

    # cart2hom
    n = pc.shape[0]
    pc_hom = np.hstack((pc,np.ones((n,1))))
    pts_2d = np.dot(np.dot(pc_hom,t.T),p.T)

    # hom2cart
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,:2]

def show_pose_on_image(pose_pc,pose_gt,ori_pcd,img_path,img_width,img_height,camera_json = './camera.json',bbox= np.array((0,0))):
    img = cv2.imread(img_path)
    for single_pose in [pose_pc,pose_gt]:
        color = tuple([0,0,255])   # gt
        print(single_pose.shape)
        if single_pose.shape[-1] != 2:
            color = np.random.randint(0,255,(1,3)).squeeze(0)
            color = tuple([0,255,125])
        
            single_pose = world2cam_one_seq(single_pose,camera_json)
        pose2d = single_pose[:,:2]
        if bbox.shape[0] != 0:
            pose2d = pose2d-bbox
        # Index of points in fov
        fov_inds = (
            (pose2d[:, 0] < img_width)
            & (pose2d[:, 0] >= 0)
            & (pose2d[:, 1] < img_height)
            & (pose2d[:, 1] >= 0)
        )
        # fov_inds = fov_inds & (single_pose[:, 0] > 0)
#         print(single_pose,fov_inds,img_width)
        imgfov_pose2d = pose2d[fov_inds,:]
        for i in range(imgfov_pose2d.shape[0]):
            cv2.circle(
                img,
                (np.int64(imgfov_pose2d[i, 0]), np.int64(imgfov_pose2d[i, 1])),
                1,
                color=color,
                thickness=-1,
            )
            
        connect = np.array([(0,4),(1,5),(2,6),(2,19),(3,7),(3,20),(4,8),(5,9),
                              (6,10),(7,12),(8,14),(9,14),(10,11),(11,12),(11,14),(13,14),
                              (13,15),(13,16),(15,17),(16,18)])
#         connect = np.array([[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]])-1

        for i in range(len(connect[:,0])):
            if fov_inds[connect[i][0]] & fov_inds[connect[i][1]]:
                pt1 = np.round(pose2d[connect[i][0]])
                pt2 = np.round(pose2d[connect[i][1]])
                cv2.line(img,tuple(pt1.astype(np.int)),tuple(pt2.astype(np.int)),color)


        color_pcd = tuple([125,125])   # gt
        pcd_2d =  world2cam_one_seq(ori_pcd,camera_json)
        if bbox.shape[0] != 0:
            pcd_2d = pcd_2d-bbox
        for i in range(ori_pcd.shape[0]):
            pcd_2d_each = pcd_2d[i]

            cv2.circle(img,tuple([pcd_2d_each[0].astype(np.int),pcd_2d_each[1].astype(np.int)]),1,color_pcd)
            
            
            
    cv2.imwrite("/remote-home/xuyt/gallery_pic/lidarcap_attentionfusion/image.png",img)