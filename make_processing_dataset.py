import os
import datasets
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, ViTImageProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
os.environ["WANDB_DISABLED"] = "true"

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# from PIL import Image
import cv2
import numpy as np
import evaluate

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    metric = evaluate.load("rouge")
    ignore_pad_token_for_loss = True

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions, padding="max_length", max_length=max_target_length).input_ids

    return labels

def feature_extraction_fn(image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """

    model_inputs = {}

    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                # img = Image.open(image_file)
                img = cv2.imread(image_file)
                if len(img.shape) == 2:  # 이미지가 2차원인 경우
                    img = np.expand_dims(img, axis=2)  # 차원을 확장하여 3차원으로 만듦

                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [cv2.imread(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values

def preprocess_fn(examples, max_target_length, check_image = True):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']
    captions = examples['caption']    
    
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)

    return model_inputs

# data_path = "D:\yoseph\DataSet\Captioning_dataset/coco2017_data/"
# ds = datasets.load_dataset(path = "make_coco_train_data", name = "2017", data_dir=data_path)

data_path = "D:\yoseph\DataSet\Captioning_dataset/coco2017_data_test/"
ds = datasets.load_dataset(path = "make_coco_train_data", name = "2017", data_dir=data_path)

new_data = [{"image_id" : -1, 
            "caption_id" : -1, 
            "height" : 720, 
            "width" : 1280, 
            "file_name" : "test.png", 
            "coco_url" : "none", 
            "image_path" : os.getcwd() + "/images/test.png",
            "caption": "The man is building a fire."},
            {"image_id" : -1, 
            "caption_id" : -1, 
            "height" : 720, 
            "width" : 1280, 
            "file_name" : "fire_00007690.jpg", 
            "coco_url" : "none", 
            "image_path" : os.getcwd() + "/images/fire_00007690.jpg",
            "caption": "The pot is on fire."},
            {"image_id" : -1, 
            "caption_id" : -1, 
            "height" : 720, 
            "width" : 1280, 
            "file_name" : "fire_00007622.jpg", 
            "coco_url" : "none", 
            "image_path" : os.getcwd() + "/images/fire_00007622.jpg",
            "caption": "A person is sitting next to a pot."},
            {"image_id" : -1, 
            "caption_id" : -1, 
            "height" : 720, 
            "width" : 1280, 
            "file_name" : "fire_00007628.jpg", 
            "coco_url" : "none", 
            "image_path" : os.getcwd() + "/images/fire_00007628.jpg",
            "caption": "A person is standing next to a pot and the pot is on fire."}]
new_ds = ds

for data in new_data:
    new_ds["train"] = new_ds["train"].add_item(data)

# image_encoder_model = "google/vit-base-patch16-224-in21k"
# text_decode_model = "gpt2"

# model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decode_model)

# # image feature extractor
# feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
# # text tokenizer
# tokenizer = AutoTokenizer.from_pretrained(text_decode_model)


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# processed_dataset = ds.map(function=preprocess_fn, batched=True, fn_kwargs={"max_target_length": 2000}, remove_columns=ds['train'].column_names)
processed_dataset = new_ds.map(function=preprocess_fn, batched=True, fn_kwargs={"max_target_length": 128}, remove_columns=ds['train'].column_names)

# processed_dataset.save_to_disk('./processed_dataset')
training_args = Seq2SeqTrainingArguments(predict_with_generate=True, evaluation_strategy="epoch", per_device_train_batch_size=4, per_device_eval_batch_size=4, output_dir="./model/image-captioning-output",num_train_epochs = 5)
trainer = Seq2SeqTrainer(model=model, tokenizer=feature_extractor, args=training_args, compute_metrics=compute_metrics, train_dataset=processed_dataset['train'], eval_dataset=processed_dataset['validation'], data_collator=default_data_collator,)

trainer.train()

trainer.save_model("./model/image-captioning-output")
tokenizer.save_pretrained("./model/image-captioning-output")