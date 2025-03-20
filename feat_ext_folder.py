
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
createFolder('./data/'+listing)
for vid in listing:
    
    vedeo = "./videos/"+vid
    ret = 1
    cap = cv2.VideoCapture(vedeo)
    while True: 
        ret, frame = cap.read()
        if not ret : break
        location = './data/'+vid
        cv2.imwrite(os.path.join(location , "%d.jpg"  % counter), frame)
        # cv2.imwrite("frame%d.jpg" % counter, frame) 
        counter = counter + 1 