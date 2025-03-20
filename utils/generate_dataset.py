import os
from networks.CNN import ResNet
from utils.KTS.cpd_auto import cpd_auto
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py

class Generate_Dataset:
    def __init__(self, video_path, save_path):
        self.resnet = ResNet()
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        # self.h5_file = h5py.File(save_path, 'w')
        #self._set_video_list(video_path)

    def _set_video_list(self, video_path):
        # import pdb;pdb.set_trace()
        if os.path.isdir(video_path):
            self.video_path = video_path
            fileExt = r".mp4",".avi"
            self.video_list = [_ for _ in os.listdir(video_path) if _.endswith(fileExt)]
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            # self.h5_file.create_group('video_{}'.format(idx+1))

    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        res_pool5 = self.resnet(frame)
        frame_feat = res_pool5.cpu().data.numpy().flatten()
        return frame_feat

    def get_change_points(self, video_feat, n_frame, fps):
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0], change_points, [n_frame-1]))
        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]
            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))
        arr = change_points
        list1 = arr.tolist()
        list2 = list1[-1].pop(1) #pop [-1]value 
        print(list2)
        print(list1)
        print("****************") # [-1][-1] value find and divided by 15
        cps_m = math.floor(arr[-1][1]/15)
        list1[-1].append(cps_m)             #append to list 
        print(list1)
        print("****************") #list to nd array convertion
        arr = np.asarray(list1)
        print(arr)
        arrmul = arr * 15
        print(arrmul)
        print("****************")   
        # print(type(change_points))
        # print(n_frame_per_seg)
        # print(type(n_frame_per_seg))
        median_frame = []
        for x in arrmul:
            print(x)
            med = np.mean(x)
            print(med)
            int_array = med.astype(int)
            median_frame.append(int_array)
        print(median_frame)
        #   print(type(int_array))
        # self.h5_file.close()
        return arrmul
        
    # TODO : save dataset
    def _save_dataset(self):
        pass

    # def generate_dataset(self,save_path):
    def generate_dataset(self, video_path,save_path):
        print('[INFO] CNN processing')

        video_path = video_path
        print(f"Video Path in generate initial part is ........{video_path}.")
        ### video list part ....
        video_list = []
        dataset = {}
        h5_file = h5py.File(save_path, 'w')
        if os.path.isdir(video_path):
            video_path = video_path
            fileExt = r".mp4",".avi"
            video_list = [_ for _ in os.listdir(video_path) if _.endswith(fileExt)]
            video_list.sort()
            print("In If part of video list..")
        else:
            print("In ELse part of video list..")
            video_path = ''
            video_list.append(video_path)
        
        print(f"Printing video LLList{video_list}")

        for idx, file_name in enumerate(video_list):
            print(f"idxxxxxx {idx},{file_name}")
            dataset['video_{}'.format(idx+1)] = {}
            h5_file.create_group('video_{}'.format(idx+1))

            print(f"H55555 file creation...")



        # video_list = _set_video_list(video_path)
        for video_idx, video_filename in enumerate(video_list):

            # h5_file = h5py.File(save_path, 'w')
            # video_path = video_path

            print(video_filename)
            if os.path.isdir(video_path):
                print("Expecting .... directory...")
                video_path = os.path.join(video_path, video_filename)
                print(f"Combined file video path is {video_path}")
            video_basename = os.path.basename(video_path).split('.')[0]
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_list = []
            picks = []
            video_feat = None
            video_feat_for_train = None
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                # print(f"Successsssss {success}")
                if frame_idx % 15 == 0:
                    if success:
                        frame_feat = self._extract_feature(frame)                    
                        picks.append(frame_idx)
                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))
                        if video_feat is None:
                            video_feat = frame_feat
                        else:
                            video_feat = np.vstack((video_feat, frame_feat))
                    else:
                        break
            video_capture.release()

            # return video_feature
            ## return the above def generate_dataset t
            arrmul = self.get_change_points(video_feat, n_frames, fps)
            h5_full_path = video_path.split('.')[0] + '.h5'
            with h5py.File(h5_full_path, 'w') as h5_file:
                print(f"inside h5 file creation")
                h5_file['features'] = list(video_feat_for_train)
                h5_file['picks'] = np.array(list(picks))
                h5_file['n_frames'] = n_frames
                h5_file['fps'] = fps
                h5_file['video_name'] = video_filename.split('.')[0]
                h5_file['change_points'] = arrmul