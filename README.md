# Official PyTorch Implementation of Exp-VQA (Pattern Recongition submission).

> [**Exp-VQA: Fine-grained Facial Expression Analysis via Visual Question Answering**]<br>
> [Yujian Yuan](https://vipl.ict.ac.cn/edu/student/master/202210/t20221019_123529.html), [Jiabei Zeng](https://vipl.ict.ac.cn/edu/teacher/mastersupvisor/202205/t20220517_35778.html), [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ&hl=zh-CN)<br>Institute of Computing Technology, Chinese Academy of Sciences;
 University of Chinese Academy of Sciences



## 📰 News

**[2024.11.24]** Training and test codes of Exp-VQA are available. <br>
**[2024.11.24]** Sythesized VQA pairs used for training are available. <br>
**[2024.10.24]** ~~Code and trained models will be released here.~~ Welcome to **watch** this repository for the latest updates. <br>
**[2024.10.24]** This work is an extension of our preliminary work [Exp-BLIP](https://github.com/Yujianyuan/Exp-BLIP).


## ⬇️ Captions and Models Download


### <div id="custom-id">(1) Sythesized VQA pairs</div>
| VQA type                         |                                                    Link                                                    |
|:------------------------------------|:-------------------------------------------------------------------------------------------------------:| 
| Global facial expression captioning (Q1)    					   |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/EVmcitRwqihDuDcPUopvNccBOUcxW6CYFcYGcy9K0sn6BQ?e=gzWrA0)|
| Local facail actions captioning (Q2)                   |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/EaL_izsHY6lHjzaKoEb-_y4BBOu73rJLVGBD2IGhJftIQA?e=ieXgEN)   |
| Single AU detection (Q3)          |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/EdRcpUjKadVEpMGH14Lm1NQBpue5JO3k8aXC1ggocF7dig?e=Jg9qhl)    | 


<a name="text"></a>
### (2) Trained models
| Model                         |                                                    Link                                                    |
|:------------------------------------|:-------------------------------------------------------------------------------------------------------:| 
| Exp-VQA    			|     [OneDrive]()|
| Exp-VQA(fz)             |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/EZLg4kUnqiNPpyuFFxko_nEB4MWSP2zku8go9nZRczkbDw?e=xbH3UD)    |


## 🔨 Installation

1. (Optional) Creating conda environment

```bash
conda create -n expvqa python=3.8.12
conda activate expvqa
```

2. Download the packages in requirements.txt 

```bash
pip install -r requirements.txt 
```

3. Download this repo. 
```bash
git clone https://github.com/Yujianyuan/Exp-VQA.git
cd Exp-VQA
```

## 🚀 Getting started

### (1) Training

You should finish the two steps sequentially for training.

1. fill the blank labeled by 'TODO' in Exp-VQA/mylavis/projects/blip2/train/vqa_ft_vicuna7b_vqa.yaml

2. training for Exp-VQA
```bash
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path mylavis/projects/blip2/train/vqa_ft_vicuna7b_vqa.yaml
```
### (2) Test

1. in test.py, finish the image path and model path
```python
import torch
from PIL import Image
from mylavis.models import my_load_model_and_preprocess

# load sample image
raw_image = Image.open("figs/happy.jpg").convert("RGB")
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
```
Then run it, you can get the captions.
```bash
python test.py
```



## 🤝 Acknowledgement
This work is supported by National Natural Science Foundation of China (No. 62176248). We also thank ICT computing platform for providing GPUs. We thank Salesforce Research sharing the code of InstructBLIP via [LAVIS](https://github.com/salesforce/LAVIS). Our codes are based on LAVIS.




