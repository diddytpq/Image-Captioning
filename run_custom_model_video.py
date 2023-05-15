
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import time
import numpy as np
from multiprocessing import Process, Pipe, Manager, Queue
from multiprocessing.managers import BaseManager
import cv2




def predict_step(image, model, feature_extractor, device, tokenizer, gen_kwargs):
    img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = feature_extractor(images=[img_input], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    print(pixel_values.shape)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

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

    # cap_main.set(cv2.CAP_PROP_FRAME_WIDTH, width/2)
    # cap_main.set(cv2.CAP_PROP_FRAME_HEIGHT, height/2)

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
    model = VisionEncoderDecoderModel.from_pretrained("./model/image-captioning-output")
    feature_extractor = ViTImageProcessor.from_pretrained("./model/image-captioning-output")
    tokenizer = AutoTokenizer.from_pretrained("./model/image-captioning-output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    video_path = "./videos/C002100_006.mp4"

    BaseManager.register('img_buffer', Img_Buffer)
    manager = BaseManager()
    manager.start()
    img_buffer_1 = manager.img_buffer()

    thread_1 = Process(target=run_video, args=[img_buffer_1, video_path])

    thread_1.start()

    while True:
        start=time.time()

        img_buffer_1_data = img_buffer_1.get()

        if img_buffer_1_data[1] == True: frame = img_buffer_1_data[0]
        else : continue

        result = predict_step(frame, model, feature_extractor, device, tokenizer, gen_kwargs)
        print(result)

        cv2.imshow("img",frame)

        key = cv2.waitKey(1)

        if key == 27:
            thread_1.terminate()
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    main()