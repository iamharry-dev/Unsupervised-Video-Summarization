
import os 
import cv2 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

listing = os.listdir('./videos')
counter = 0
for vid in listing:
    createFolder('./data/')
    video = "./videos/"+vid
    ret = 1
    cap = cv2.VideoCapture(video)
    while True: 
        ret, frame = cap.read()
        if not ret : break
        location = './data/'
        cv2.imwrite(os.path.join(location , "%d.jpg"  % counter), frame)
        # cv2.imwrite("frame%d.jpg" % counter, frame) 
        counter = counter + 1 
