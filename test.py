import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import os

processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-capfilt-large", torch_dtype=torch.float16).to("cuda")

img_path = os.getcwd() + '/images/fire/fire_00007690.jpg' 
raw_image = Image.open(img_path).convert('RGB')

# question = "Is the person currently lying down?"
question = "Do you currently see fire?"

inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))