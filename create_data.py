import argparse
from datetime import date
from utils.generate_dataset import Generate_Dataset
from joblib import Parallel, delayed
import configparser
# import multiprocessing as mp
from multiprocessing import Pool
from datetime import datetime

config = configparser.ConfigParser()
config.read('config.ini')
num_cam = config['DEFAULT']['num_cam']
# print(con)

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('--input', '--split', type=str, help="input video")
parser.add_argument('--output', type=str, default='', help="out data")

args = parser.parse_args()
def call_summ(i):
    print(datetime.now())
    gen = Generate_Dataset(args.input, args.output + str(i))
    file_path = "C" + str(i)
    print("***************************")
    
    input_path = args.input + "/" + file_path
    output_file = "./dataset" + "/" + file_path + ".h5"
    gen.generate_dataset(input_path,output_file)

if __name__ == "__main__":
    pool = Pool()
    pool.map(call_summ, range(0, 3))
    # pool.join()
    pool.close()
    # gen = Generate_Dataset(args.input, args.output)
    # gen._generator()
    # gen.h5_file.close()
    # jobs = []
    # for i in range(len(num_cam)):
    #     import pdb;pdb.set_trace()
    #     file_path = "C" + str(i)
    #     input_path = args.input + "/" + file_path
    #     print(input_path)
    #     output_file = "./dataset" + "/" + file_path + ".h5"
    #     gen = Generate_Dataset(input_path, output_file)
        
    #     process = mp.Process(target= gen._generator(i,file_path))
    #     jobs.append(process)

    # for j in jobs:
    #     j.start()

    # for j in jobs:
    #     j.join()


    # results = Parallel(n_jobs=2)(delayed(gen._generator())(i) for i in range(num_cam))
    # pool = multiprocessing.Pool()
    # inputs = list(num_cam)
    # outputs_async = pool.map_async(square, inputs)
    # outputs = outputs_async.get()
    
    
    # cam_list = [(i) for i in range(num_cam)]
    # print(cam_list)
    # pool = Pool()
    # cam_list = [int(i) for i in range(int(num_cam))]
    # for i in cam_list:
    #     pool.map(call_summ(i))

    # pool = Pool()
    # pool.map(call_summ, range(0, 3))
    # pool.close()


    # with Pool(num_cam) as p:
        # for i in range(num_cam):
        #     print(p.map(call_summ, [i]))

# from joblib import Parallel, delayed
# def process(i):
#     return i * i
    
# results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
# print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# def generate_dataset(self):
#     print('[INFO] CNN processing')
#     Parallel(n_jobs=-1)(delayed(self._generator)(video_idx, video_filename) for video_idx, video_filename in enumerate(self.video_list))


# if __name__ == '__main__':
#     pool = multiprocessing.Pool()
#     inputs = [0,1,2,3,4]
#     outputs_async = pool.map_async(square, inputs)
#     outputs = outputs_async.get()
#     print("Output: {}".format(outputs))


#     import multiprocessing as mp

# def f_print(str1):
#     print(str1)
# video_list = ["/home/vinsent/Music/Fraction_chuttyboy/videos/47.mp4","/home/vinsent/Music/Fraction_chuttyboy/videos/48.mp4"]
# jobs = []
# for i in range(len(video_list)):

#     process = mp.Process(target= f_print,args=([video_list[i]]))
#     jobs.append(process)

# for j in jobs:
#     j.start()

# for j in jobs:
#     j.join()