import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-s", "--save", help="whether save",action="store_true")
parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
args = parser.parse_args()

conf = get_config(False)

mtcnn = MTCNN()
print('arcface loaded')

learner = face_learner(conf, True)
learner.threshold = args.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')

if args.update:
    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
    print('facebank updated')
else:
    targets, names = load_facebank(conf)
    print('facebank loaded')

# inital camera

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)


class faceRec:
    def __init__(self):
        self.width = 800
        self.height = 800
        self.image = None
    def main(self): 
        while cap.isOpened():
            isSuccess,frame = cap.read()
            if isSuccess:            
                try:
    #                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                    image = Image.fromarray(frame)
                    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                    bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1,-1,1,1] # personal choice    
                    results, score = learner.infer(conf, faces, targets, args.tta)
                    # print(score[0])
                    for idx,bbox in enumerate(bboxes):
                        if args.score:
                            frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                        else:
                            if float('{:.2f}'.format(score[idx])) > .98:
                                name = names[0]
                            else:    
                                name = names[results[idx]+1]
                            frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                except:
                    pass    
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tostring()
                # cv2.imshow('Arc Face Recognizer', frame)


            if cv2.waitKey(1)&0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()    
