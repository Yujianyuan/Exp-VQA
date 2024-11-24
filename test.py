import torch
from PIL import Image
from mylavis.models import my_load_model_and_preprocess

# load sample image
raw_image = Image.open("figs/happy.png").convert("RGB")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads Exp-VQA model
# this also loads the associated image processors
checkpoint_path = './exp_vqa_trimmed.pth'
model, vis_processors, _ = my_load_model_and_preprocess(name="blip2_vicuna_instruct",
                model_type="vicuna7b", dict_path = checkpoint_path, is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" 
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# input your question
question = "How can this person's emotion be inferred from their facial actions?"

# generate caption
print('[1 caption]:',model.generate({"image": image, "prompt":question}))

# use nucleus sampling for diverse outputs 
print('[3 captions]:',model.generate({"image": image, "prompt":question}, use_nucleus_sampling=True, num_captions=3))
