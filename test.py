import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import os
import numpy as np
from multiprocessing import Process, Pipe, Manager, Queue
from multiprocessing.managers import BaseManager
import cv2
import time

class Img_Buffer(object):
    def __init__(self):
        self.image = np.array(np.NaN)
        self.ret = False

    def put(self, data):
        self.image = data[0]
        self.ret = data[1]

    def get(self):
        return [self.image, self.ret]

def run_video(buffer, source):
    cap_main = cv2.VideoCapture(source)
    # cap_main = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    cap_main.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    width = cap_main.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT)

    total_frame = cap_main.get(cv2.CAP_PROP_FRAME_COUNT)
    cap_main.set(cv2.CAP_PROP_POS_FRAMES, total_frame / 3)

    disconnect_time = time.time()

    while True:
        ret, frame = cap_main.read()

        if ret:
            buffer.put([frame, ret])
            disconnect_time = time.time()
            time.sleep(0.02)

        else:
            print("camera disconnect")
            if time.time() - disconnect_time > 60:
                break


def main():

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to("cuda")

    # question = "What is the person doing?"
    # question = "If there is a person in the current image, is the person falling down?"
    question = "If there is a person in the current image, are they fighting?"
    # question = "Do you see fire or smoke?"

    video_path = "./videos/fight_5s.mp4"

    BaseManager.register('img_buffer', Img_Buffer)
    manager = BaseManager()
    manager.start()
    img_buffer_1 = manager.img_buffer()

    thread_1 = Process(target=run_video, args=[img_buffer_1, video_path])

    thread_1.start()

    while True:
        start=time.time()

        img_buffer_1_data = img_buffer_1.get()

        if img_buffer_1_data[1] == True: frame_ori = img_buffer_1_data[0]
        else : continue
        frame = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)
        inputs = processor(frame, question, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)

        print("Question :", question)
        print("Answer :", processor.decode(out[0], skip_special_tokens=True))

        cv2.imshow("img", frame_ori)

        key = cv2.waitKey(1)

        if key == 27:
            thread_1.terminate() 
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()


