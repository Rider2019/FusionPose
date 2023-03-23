import torch
import torch.utils.data
import pickle
import numpy as npco
from utils import *
from PIL import Image
from plyfile import PlyData
connect = np.array([(0,4),(1,5),(2,6),(2,19),(3,7),(3,20),(4,8),(5,9),
                              (6,10),(7,12),(8,14),(9,14),(10,11),(11,12),(11,14),(13,14),
                              (13,15),(13,16),(15,17),(16,18)])
ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
in_matrix = np.array([[683.8, 0.0, 673.5907],
                     [0.0, 684.147, 372.8048],
                     [0.0, 0.0, 1.0]])
def crop_pc(pointcloud):
    crop_range =np.array([-0.2,0.2,-0.15,0.15,-0.3,0.3])
    shift =  np.random.uniform(-0.3,0.3,2)
    shift_z = np.random.uniform(-0.6,0.6,1)
    crop_range += np.array([shift[0],shift[0],shift[1],shift[1],shift_z[0],shift_z[0]])
    # crop_range += shift2
    mask = np.ones(pointcloud.shape[0], np.bool)
    crop_ind = np.where(
        (pointcloud[:,0] > crop_range[0]) & (pointcloud[:,0] < crop_range[1]) &
        ( pointcloud[:,1]>crop_range[2]) & ( pointcloud[:,1]<crop_range[3]) &
        (pointcloud[:,2] > crop_range[4]) & (pointcloud[:,2] < crop_range[5]) 
    )
    mask[crop_ind] = False
    return pointcloud[mask]

   
# class HybridcapDataset(torch.utils.data.Dataset):
    
#     def __init__(self,args,m,transform=None,):
#         self.dataset = []
#         self.root_path = '/remote-home/share/Lidar-IMU/hybridcap/'
#         self.num_points = args.num_points
#         self.crop_agu = args.crop_agu
#         self.args = args
#         self.transform = transform
#         # data_info_path = '/remote-home/share/STCrowd/0319_annots.npz'
#         data_info_path = self.root_path + 'info.pkl'
#         T = args.frames
#         if m == 'e':
#             data_info_path = data_info_path.replace('train','test')
#         datas = list(np.load(data_info_path,allow_pickle=True))
#         old_motion_id = datas[0]['group']

#         seq = []
#         if T == 1:
#             self.dataset = [[data] for data in datas]
#         else:
#             for i in range(0,len(datas)):
#                 motion_id = datas[i]['group']
        
#                 if motion_id == old_motion_id:
#                     seq.append(datas[i])
#                 else:
#                     old_motion_id = motion_id
#                     seq=[datas[i]]
#                     #break
#                 if len(seq) == T:
#                     self.dataset.append(seq)
#                     seq=[]
    
#     def __getitem__(self, index):
#         example_seq = self.dataset[index]
#         seq_pc = []
#         seq_pc_mean = []
#         seq_j = []
#         seq_image = []
#         seq_kp_3d = []
#         seq_image_path = []


#         for example in example_seq:
#             image_path = self.root_path + example['image_path']
#             seq_image_path.append(image_path)
#             if self.transform is not None:
#                 img = self.transform(img)
#                 seq_image.append(np.array(img).transpose(1,2,0))

#             pc_path = self.root_path + example['lidar_path']
#             pc_data = np.fromfile(pc_path,dtype=np.float32).reshape([-1,3])[:,:3]- np.array([-5.0,0.0,0.0],dtype=np.float32) # -5,0,0 fix for project
#             pc_data,data_mean = norm_pc(pc_data)
#             if self.crop_agu:
#                 if np.random.random() > 0.4: 
#                     pc_data = crop_pc(pc_data)
            
#             if len(pc_data)==0:
#                 print(pc_path)
#             pc_data = farthest_point_sample(pc_data,self.num_points)
#             seq_pc.append(pc_data)
#             seq_pc_mean.append(data_mean)
            
#             seq_j.append(example['kp_2d'])
#             seq_kp_3d.append(np.array(example['kp_3d']).reshape(-1,3)- np.array([-5.0,0.0,0.0]) - data_mean)

#         Item = {
#             'img_path': seq_image_path,
#             'data': np.array(seq_pc),
#             'gt': np.array(seq_j),
#             'gt_3d': np.array(seq_kp_3d),
#             'data_ori_mean': np.array(seq_pc_mean),
#             'camera_calib':'camera_hybridcap.json',
#             'image_data': np.array(seq_image),
#         }

#         return Item
#     def __len__(self):
#         return len(self.dataset)

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,args,m,transform=None,):
        self.dataset = []
        # data_info_path = '/remote-home/share/STCrowd/0319_annots.npz'
        if args.kp == 21:
            # data_info_path = './kp21_openpose2.pkl'
            data_info_path = './kp21_openpose_xyt.pkl'
        elif args.kp == 17:
            data_info_path = './17joints.npz'
        self.num_points = args.num_points
        self.crop_agu = args.crop_agu
        self.args = args
        self.transform = transform
        T = args.frames
        # datas = list(np.load(data_info_path,allow_pickle=True)['annots'])
        datas = list(np.load(data_info_path,allow_pickle=True))
        if m == 't':
            datas = datas[:43869]
        else:
            datas = datas[43869:]
        # if m == 't':
        #     datas = datas[42869:43869]
        # else:
        #     datas = datas[43869:44869]
        # print("------",len(datas))
        datas.sort(key=lambda x:int(x['group']))
        old_motion_id = datas[0]['group']

        seq = []
        if T == 1:
            self.dataset = [[data] for data in datas]
        else:
            for i in range(0,len(datas)):
                motion_id = datas[i]['group']
                vali = datas[i]['valid']
                # raise RuntimeError(self.args.motion)
                self.args.motion=False
                if self.args.motion:
                    if vali != True:
                        seq = []
                        continue
                else:
                    if vali == False:
                        seq = []
                        continue
                if motion_id == old_motion_id:
                    seq.append(datas[i])
                else:
                    old_motion_id = motion_id
                    seq=[datas[i]]
                    #break
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]
    
    def __getitem__(self, index):
        example_seq = self.dataset[index]
        seq_pc = []
        seq_pc_mean = []
        seq_j = []
        seq_image_ori_path = []
        seq_image = []
        seq_imgkp = []
        seq_pseudo_kp = []

        example_list = []

        for example in example_seq:
            image_path = '/storage/data/xuyt1/dataset/STCrowd/crop_result/' + '/'.join(example['image_path'].split('/')[-2:])
            # reading the images and converting them to correct size and color
            img = Image.open(image_path)
            if self.transform is not None:
                
                img = self.transform(img)
                seq_image.append(np.array(img).transpose(1,2,0))

            pc_path = '/storage/data/xuyt1/dataset/STCrowd/crop_result/'+example['lidar_path'].split('/')[-2]+'/'+example['lidar_path'].split('/')[-1]
            pc_data = np.fromfile(pc_path,dtype=np.float32).reshape([-1,4])[:,:3]
            if len(pc_data)==0:
                print(pc_path)

            pc_data,data_mean = norm_pc(pc_data)
            if self.crop_agu:
                if np.random.random() > 0.4: 
                    pc_data = crop_pc(pc_data)
            
            if len(pc_data)==0:
                print(pc_path)
            pc_data = farthest_point_sample(pc_data,self.num_points)
            seq_pc.append(pc_data)
            seq_pc_mean.append(data_mean)
            seq_image_ori_path.append(example['image_ori_path'])
            if self.args.motion:
                seq_pseudo_kp.append(example['kp_3d_mean']-data_mean)

            gt_j = example['kp_2d']
            seq_j.append(gt_j)
            img_kp_gt = np.zeros_like(gt_j)
            w,h = example['image_bbox'][2], example['image_bbox'][3]
            img_kp_gt[:,0] = (gt_j[:,0] - (example['image_bbox'][0]-w/2))/ w*128 # x --- 
            img_kp_gt[:,1] = (gt_j[:,1] - (example['image_bbox'][1]-h/2))/ h*256 # y |

            # img_kp_gt,img_kp_gt_mean = norm_pc(gt_j)
            seq_imgkp.append(img_kp_gt)
            # example_list.append(example) # used for anno pkl generation ( corresponding with path and bbox )

        seq_v = []
        if self.args.motion:
            for i in range(len(example_seq)-1):
                seq_v.append(seq_pseudo_kp[i+1]-seq_pseudo_kp[i])
            seq_v.append(seq_pseudo_kp[0]-seq_pseudo_kp[i+1])


        Item = {
            'img_path': seq_image_ori_path,
            'data': np.array(seq_pc),
            'gt': np.array(seq_j),
            'imggt': np.array(seq_imgkp),
            'data_ori_mean': np.array(seq_pc_mean),
            'camera_calib':'camera.json',
            'image_data': np.array(seq_image),
            'pseudo_kp3d':np.array(seq_pseudo_kp),
            'motion':np.array(seq_v)
            # 'example_list':example_list
        }

        return Item
    def __len__(self):
        return len(self.dataset)

# class FusionDataset(torch.utils.data.Dataset):
    
#     def __init__(self,args,m,transform=None):
#         self.dataset = []
#         self.error_num=0
#         # data_info_path = '/remote-home/share/STCrowd/0319_annots.npz'
#         # if args.kp == 21:
#         #     data_info_path = '/remote-home/xuyt/code_share/lidar-kp/kp21_openpose.pkl'
#         # elif args.kp == 17:
#         #     data_info_path = './17joints.npz'
#         data_info_path = '/public/home/xuyt1/lidar-kp/kp21_openpose_xyt.pkl'
#         self.num_points = args.num_points
#         self.args = args
#         self.transform = transform
#         T = args.frames
#         # datas = list(np.load(data_info_path,allow_pickle=True)['annots'])
#         datas = list(np.load(data_info_path,allow_pickle=True))
#         if m == 't':
#             datas = datas[:43869]
#         else:
#             datas = datas[43869:]
#         # if m == 't':
#         #     datas = datas[:43869]
#         # else:
#         #     datas = datas[43869:44869]
        
#         # print("------",len(datas))
#         datas.sort(key=lambda x:int(x['group']))
#         old_motion_id = datas[0]['group']

#         seq = []
#         if T == 1:
#             self.dataset = [[data] for data in datas]
#         else:
#             for i in range(0,len(datas)):
#                 motion_id = datas[i]['group']
#                 vali = datas[i]['valid']
#                 if vali == False:
#                     seq = []
#                     continue
#                 if motion_id == old_motion_id:
#                     seq.append(datas[i])
#                 else:
#                     old_motion_id = motion_id
#                     seq=[datas[i]]
#                     #break
#                 if len(seq) == T:
#                     self.dataset.append(seq)
#                     seq=[]
    
#     def __getitem__(self, index):

#         # debug

        
#         example_seq = self.dataset[index]
#         seq_pc = []
#         seq_pc_mean = []
#         seq_j = []
#         seq_image_ori_path = []
#         seq_image = []
#         seq_imgkp = []
#         pixel_xy_image_list = np.around(np.zeros((len(example_seq),256,2))).astype(int)
#         # np.around(pixel_xy).astype(int)
#         image_bbox = np.zeros((len(example_seq),4))
#         for c, example in enumerate(example_seq):
#             seq_image_ori_path.append(example['image_ori_path'])
#             image_bbox[c] = np.array(example['image_bbox'])

#             gt_j = example['kp_2d'] # ？
#             seq_j.append(gt_j)
#             # ======= START ======= # Modified by xuyt on 2020/4/8 
#             # img_kp_gt = gt_j.copy()
#             img_kp_gt = np.array(gt_j)
#             img_kp_gt[:,0] -= example['image_bbox'][0]
#             img_kp_gt[:,1] -= example['image_bbox'][1]
#             img_kp_gt[:,0] *= (1 / example['image_bbox'][2] * 128)
#             img_kp_gt[:,1] *= (1 / example['image_bbox'][3] * 256)
#             img_kp_gt[:,0] += 64
#             img_kp_gt[:,1] += 128
#             seq_imgkp.append(img_kp_gt)
#             image_path = '/storage/data/xuyt1/dataset/STCrowd/crop_result/' + '/'.join(example['image_path'].split('/')[-2:])
            
#             img = Image.open(image_path)
#             if self.transform is not None:
#                 img = self.transform(img)
#             seq_image.append(np.array(img).transpose(1,2,0))

#             pc_path = '/storage/data/xuyt1/dataset/STCrowd/crop_result/'+example['lidar_path'].split('/')[-2]+'/'+example['lidar_path'].split('/')[-1]
#             pc_data = np.fromfile(pc_path,dtype=np.float32).reshape([-1,4])[:,:3]
#             if len(pc_data)==0:
#                 print(pc_path)

#             # pc_data = farthest_point_sample(pc_data,self.num_points)
#             # norm_pc_data,data_mean = norm_pc(pc_data)
#             # seq_pc.append(norm_pc_data)
#             # seq_pc_mean.append(data_mean)
            
#             points_T_3 = np.transpose(pc_data) # 3 256
#             points_T = np.ones((4,points_T_3.shape[1]))
#             points_T[:3,:]=points_T_3
#             points_T_camera = np.dot(ex_matrix, points_T)
#             pixel = np.dot(in_matrix, points_T_camera).T
            
#             pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
            
#             pixel_xy = np.around(pixel_xy).astype(int)

#             w,h = example['image_bbox'][2],example['image_bbox'][3]
#             c_w = example['image_bbox'][0]
#             c_h = example['image_bbox'][1]
#             pixel_xy[:,0] = np.around((pixel_xy[:,0] - (c_w-w/2))/w*128).astype(int)
#             pixel_xy[:,1] = np.around((pixel_xy[:,1] - (c_h-h/2))/h*256).astype(int)
#             n = 0
#             points_image = []
#             pixel_xy_image = []
#             invalid_index=[]
#             valid_index=[]
#             for i in range(pixel_xy.shape[0]):
#                 if pixel_xy[i][0] >= 0 and pixel_xy[i][0] < 128 \
#                     and pixel_xy[i][1] >= 0 and pixel_xy[i][1] < 256 and points_T_camera[2, i] > 0:
#                     n += 1
#                     valid_index.append(i)
#                     points_image.append(points_T_camera[0:3, i])
#                     pixel_xy_image.append(pixel_xy[i, :])
#                 else:
#                     invalid_index.append(i)
            
#             points_image = np.array(points_image)
#             pixel_xy_image = np.array(pixel_xy_image)
            
#             # ######### debug 可视化 4月24日 START CROP
            
#             # image = Image.open(image_path)
#             # pil_img=Image.fromarray(np.uint8(image))
#             # vis_transform = transforms.Compose([
#             #     transforms.Resize((128,64), interpolation=InterpolationMode.BICUBIC),
#             # ])
#             # image = np.array(vis_transform(pil_img)) # 256 128 3
#             # point_color = (0, 255, 255) # BGR
            

#             # # image = np.zeros((1280, 1280, 3), np.uint8)
#             # # image+=255
#             # for j in range(n):
#             #     cv2.circle(image, (pixel_xy_image[j, 0], pixel_xy_image[j, 1]), 1, point_color, -1)
#             # cv2.imwrite("vis_{}.png".format(c),image)
#             # np.set_printoptions(threshold=1000000)
#             # raise RuntimeError(pixel_xy_image.shape)

#             # ######### debug 可视化 4月24日 END CROP



#             pc_data = pc_data[(valid_index)] # 原来的pc_data (2691, 3)
            
#             pc_data = farthest_point_sample(pc_data,self.num_points)
#             norm_pc_data,data_mean = norm_pc(pc_data)
#             seq_pc.append(norm_pc_data)
            
#             seq_pc_mean.append(data_mean)
#             # 第二次
#             points_T_3 = np.transpose(pc_data) # 3 256
#             points_T = np.ones((4,points_T_3.shape[1]))
#             points_T[:3,:]=points_T_3
#             points_T_camera = np.dot(ex_matrix, points_T)
#             pixel = np.dot(in_matrix, points_T_camera).T
            
#             if pixel.shape[0] == 0:
#                 self.error_num+=1
#                 raise RuntimeError(self.dataset[index].keys())

#             else:
#                 pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
#                 pixel_xy = np.around(pixel_xy).astype(int)
#                 w,h = example['image_bbox'][2],example['image_bbox'][3]
#                 c_w = example['image_bbox'][0]
#                 c_h = example['image_bbox'][1]
#                 pixel_xy[:,0] = np.around((pixel_xy[:,0] - (c_w-w/2))/w*128).astype(int)
#                 pixel_xy[:,1] = np.around((pixel_xy[:,1] - (c_h-h/2))/h*256).astype(int)
#                 n = 0
#                 points_image = []
#                 pixel_xy_image = []
#                 invalid_index=[]
#                 valid_index=[]
#                 for i in range(pixel_xy.shape[0]):
#                     if pixel_xy[i][0] >= 0 and pixel_xy[i][0] <= 128 \
#                         and pixel_xy[i][1] >= 0 and pixel_xy[i][1] <= 256 and points_T_camera[2, i] > 0:
#                         n += 1
#                         valid_index.append(i)
#                         points_image.append(points_T_camera[0:3, i])
#                         pixel_xy_image.append(pixel_xy[i, :])
#                     else:
#                         invalid_index.append(i)
#                 points_image = np.array(points_image)
#                 pixel_xy_image = np.array(pixel_xy_image)
                
#                 pixel_xy_image_list[c] = pixel_xy_image
            
#         Item = {
#             'image_bbox': image_bbox,
#             'img_path': seq_image_ori_path,
#             'data': np.array(seq_pc),
#             'gt': np.array(seq_j),
#             'imggt': np.array(seq_imgkp),
#             'data_ori_mean': np.array(seq_pc_mean),
#             'gt_3d':np.zeros((self.args.frames,int(self.args.kp),3)),
#             'camera_calib':'camera.json',
#             'image_data': np.array(seq_image),
#             'pixel_xy_image_list':pixel_xy_image_list
#         }
#         # raise RuntimeError("STOP")
#         return Item
#     def __len__(self):
#         return len(self.dataset)



# class LidarCapDataset(Dataset):
#     def __init__(self,args,m,transform=None,):
#         self.dataset = []
#         # self.root_path = '/storage/data/xuyt1/dataset/lidarhuman26M/'
#         self.root_path = '/remote-home/share/Lidar-IMU/lidarhuman26M/'
#         self.num_points = args.num_points
#         self.crop_agu = args.crop_agu
#         self.args = args
#         self.transform = transform
#         # data_info_path = '/remote-home/share/STCrowd/0319_annots.npz'
#         data_info_path = self.root_path + 'train_info.pkl'
#         T = args.frames
#         if m == 'e':
#             data_info_path = data_info_path.replace('train','test')
#         datas = list(np.load(data_info_path,allow_pickle=True))
#         old_motion_id = datas[0]['group']

#         seq = []
#         if T == 1:
#             self.dataset = [[data] for data in datas]
#         else:
#             for i in range(0,len(datas)):
#                 motion_id = datas[i]['group']
#                 vali = datas[i]['valid']
#                 if vali == False or vali == 'empty':
#                     seq = []
#                     continue
#                 # if motion_id == 10 or motion_id == 11 or motion_id == 14 or motion_id == 15:
#                 #     continue
#                 if motion_id == old_motion_id:
#                     seq.append(datas[i])
#                 else:
#                     old_motion_id = motion_id
#                     seq=[datas[i]]
#                     #break
#                 if len(seq) == T:
#                     self.dataset.append(seq)
#                     seq=[]
    
#     def __getitem__(self, index):
        
#         example_seq = self.dataset[index]
#         seq_pc = []
#         seq_pc_mean = []
#         seq_j = []
#         seq_kp_3d = []
#         seq_image = []
#         seq_imgkp = []
#         seq_image_path = []
#         bbox_list = []

#         for example in example_seq:
#             # raise RuntimeError(example.keys())
#             image_path = self.root_path + example['image_path']
#             seq_image_path.append(image_path)
#             # reading the images and converting them to correct size and color

#             plydata = PlyData.read(self.root_path + example['pc_path'])
#             pc_data = np.transpose(np.array([plydata['vertex']['x'],
#                                         plydata['vertex']['y'],
#                                         plydata['vertex']['z']]).reshape(3,-1),(1,0))
#             pc_data,data_mean = norm_pc(pc_data)
            
#             if self.crop_agu:
#                 if np.random.random() > 0.4: 
#                     pc_data = crop_pc(pc_data)


            
            
#             pc_data = farthest_point_sample(pc_data,self.num_points)
#             seq_pc.append(pc_data)
#             seq_pc_mean.append(data_mean)
#             if 'kp_2d' not in example.keys():
#                 print(example)
#             gt_j = np.array(example['kp_2d'])
#             seq_j.append(gt_j)

#             seq_kp_3d.append(np.array(example['gt_joint']).reshape(-1,3))
            
            
#             img_kp_gt = np.zeros_like(gt_j)
#             if self.transform is not None:
#                 img = Image.open(image_path)
#                 # raise RuntimeError(np.array(img).shape)
#                 img = self.transform(img)
#                 seq_image.append(np.array(img).transpose(1,2,0))
#                 w,h = np.array(img).shape[1],np.array(img).shape[2]
#                 img_kp_gt[:,0] = (gt_j[:,0] - example['bbox'][0])/ w*128 # x --- 
#                 img_kp_gt[:,1] = (gt_j[:,1] - example['bbox'][1])/ h*128 # y |

#                 bbox_list.append(example['bbox'])
            
#                 # img_kp_gt,img_kp_gt_mean = norm_pc(gt_j)
#                 seq_imgkp.append(img_kp_gt)
#             # example_list.append(example) # used for anno pkl generation ( corresponding with path and bbox )

#         Item = {
#             'img_path': seq_image_path,
#             'data': np.array(seq_pc),
#             'gt': np.array(seq_j),
#             'imggt': np.array(seq_imgkp),
#             'data_ori_mean': np.array(seq_pc_mean),
#             'gt_3d':np.array(seq_kp_3d),
#             'camera_calib':'camera_lidarcap.json',
#             'image_data': np.array(seq_image),
#             'bbox':np.array(bbox_list)
#             # 'example_list':example_list
#         }

#         return Item
#     def __len__(self):
#         return len(self.dataset)


     
# # no image input, 3d kp supervise
# class GT_Dataset(torch.utils.data.Dataset):
#     def __init__(self,args,m='with image'):
#         self.dataset = []
#         T = args.frames
#         self.crop_agu = args.crop_agu
#         self.num_points = args.num_points
#         if m == 'with image':
#             info_path = '/remote-home/share/Lidar-IMU/data/data_img.pkl'
#         elif m == 'train':
#             info_path = '/remote-home/share/Lidar-IMU/data/AMASS_train.pkl'
#         elif m == 'test':
#             info_path = '/remote-home/share/Lidar-IMU/data/AMASS_test.pkl'
#         # elif m == 'H36M-train':
#         #     with open('/remote-home/share/POSE_dataset/H36M_simulate/anno.pkl','rb') as f:
#         #         datas = pickle.load(f)[:3000]
#         # elif m == 'H36M-test':
#         #     with open('/remote-home/share/POSE_dataset/H36M_simulate/anno.pkl','rb') as f:
#         #         datas = pickle.load(f)[3000:]
#         else:
#             info_path = '/remote-home/share/Lidar-IMU/data/Test_1.pkl'
#         with open(info_path,'rb') as f:
#             datas = pickle.load(f) 

#         if 'seq_path' in datas[0].keys():
#             old_motion_id = datas[0]['seq_path']
#         else:
#             old_motion_id = datas[0]['motion_id']
#         seq = []
#         if T == 1:
#             self.dataset = [[data] for data in datas]
#         else:
#             for i in range(0,len(datas)):
#                 if 'seq_path' in datas[0].keys():
#                     motion_id = datas[i]['seq_path']
#                 else:
#                     motion_id = datas[i]['motion_id']
#                 if motion_id == old_motion_id:
#                     seq.append(datas[i])
#                 else:
#                     old_motion_id = motion_id
#                     seq=[datas[i]]
#                     #break
#                 if len(seq) == T:
#                     self.dataset.append(seq)
#                     seq=[]
#     def __getitem__(self, index):

#         example_seq = self.dataset[index]
#         seq_pc = []
#         seq_pc_mean = []
#         seq_j = []
#         seq_kp_3d = []
#         seq_image_ori_path = []
#         for example in example_seq:
#             pc_path = example['pc'].replace('renym','share')
#             pc_data = np.fromfile(pc_path,dtype=np.float32).reshape([-1,3])[:,:3]

#             if len(pc_data)==0:
#                 print(pc_path)

#             pc_data,data_mean = norm_pc(pc_data)

#             if self.crop_agu:
#                 if np.random.random() > 0.4: 
#                     pc_data = crop_pc(pc_data)

#             pc_data = farthest_point_sample(pc_data,self.num_points)
#             seq_pc.append(pc_data)
#             seq_pc_mean.append(data_mean)
#             if 'image' in example.keys():
#                 image_ori_path = example['image']
#             else:
#                 image_ori_path = ''
#             seq_image_ori_path.append(image_ori_path)
#             # gt_j = example['kp_2d']
#             # seq_j.append(gt_j)
#             seq_kp_3d.append(np.array(example['gt_joint']).reshape(-1,3))
#             seq_j.append(world2cam_one_seq(np.array(example['gt_joint']).reshape(-1,3)+data_mean,camera_json = 'camera_new.json'))

#         Item = {
#             'img_path': seq_image_ori_path,
#             'data': np.array(seq_pc),
#             'gt': np.array(seq_j),
#             'data_ori_mean': np.array(seq_pc_mean),
#             'gt_3d':np.array(seq_kp_3d),
#             'camera_calib':'camera_new.json'
#         }

#         return Item

#     def __len__(self):
#         return len(self.dataset)
  
class HybridcapDataset(torch.utils.data.Dataset):
    
    def __init__(self,args,m,transform=None,):
        self.dataset = []
        self.root_path = '/remote-home/share/Lidar-IMU/hybridcap_crop/'
        # self.root_path = '/storage/data/xuyt1/dataset/hybridcap_crop/'
        self.num_points = args.num_points
        self.crop_agu = args.crop_agu
        self.args = args
        self.transform = transform
        self.trans = np.array([-5.0,0.0,0.0],dtype=np.float32)
        self.image_size = [4096,2160]
        # data_info_path = '/remote-home/share/STCrowd/0319_annots.npz'
        data_info_path = self.root_path + 'info_train_crop.pkl'
        T = args.frames
        if m == 't':
            datas = list(np.load(data_info_path,allow_pickle=True)) + list(np.load(data_info_path.replace('train','test'),allow_pickle=True))
        elif m == 'e':
            datas = list(np.load(data_info_path.replace('train','test'),allow_pickle=True))
        old_motion_id = datas[0]['group']
        seq = []
        if T == 1:
            self.dataset = [[data] for data in datas]
        else:
            for i in range(0,len(datas)):
                motion_id = datas[i]['group']
        
                if motion_id == old_motion_id:
                    seq.append(datas[i])
                else:
                    old_motion_id = motion_id
                    seq=[datas[i]]
                    #break
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]
    
    def __getitem__(self, index):
        example_seq = self.dataset[index]
        seq_pc = []
        seq_pc_mean = []
        seq_j = []
        seq_image = []
        seq_kp_3d = []
        seq_image_path = []
        seq_imgkp = []
        seq_v = []


        for example in example_seq:
            image_path = self.root_path + example['image_crop_path']
            seq_image_path.append(image_path)

            gt_j = example['kp_2d']
            seq_j.append(gt_j)
            
            if self.transform is not None:
                img = Image.open(image_path)

                img = self.transform(img)
                seq_image.append(np.array(img).transpose(1,2,0))
                img_kp_gt = np.zeros_like(gt_j)
                w,h = self.image_size[0],self.image_size[1]
                img_kp_gt[:,0] = gt_j[:,0]/ w*128 # x --- 
                img_kp_gt[:,1] = gt_j[:,1]/ h*128 # y |
                seq_imgkp.append(img_kp_gt)


            pc_path = self.root_path + example['lidar_path']
            pc_data = np.fromfile(pc_path,dtype=np.float32).reshape([-1,3])[:,:3]- self.trans # -5,0,0 fix for project
            pc_data,data_mean = norm_pc(pc_data)
            if self.crop_agu:
                if np.random.random() > 0.4: 
                    pc_data = crop_pc(pc_data)
            
            if len(pc_data)==0:
                print(pc_path)
            pc_data = farthest_point_sample(pc_data,self.num_points)
            seq_pc.append(pc_data)
            seq_pc_mean.append(data_mean)
            

            seq_kp_3d.append(np.array(example['kp_3d'])- self.trans - data_mean)
        # self.args.motion = False
        if self.args.motion:
            for i in range(len(example_seq)-1):
                seq_v.append(seq_kp_3d[i+1]-seq_kp_3d[i])
            # print(np.array(seq_v).shape)
            seq_v.append(np.zeros((21,3)))


        Item = {
            'img_path': seq_image_path,
            'data': np.array(seq_pc),
            'gt': np.array(seq_j),
            'imggt': np.array(seq_imgkp),
            'gt_3d': np.array(seq_kp_3d),
            'data_ori_mean': np.array(seq_pc_mean),
            'camera_calib':'camera_hybridcap.json',
            'image_data': np.array(seq_image),
            'motion':np.array(seq_v)
        }

        return Item
    def __len__(self):
        return len(self.dataset)