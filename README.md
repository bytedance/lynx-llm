# What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?

<div align="center">
  <img width="20%" src="images/logo_plus.png">
</div>

**Yan Zeng\*, Hanbo Zhang\*, Jiani Zheng\*, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, Tao Kong**  
*Equal Contribution


[![Project](http://img.shields.io/badge/Project-Lynx-E3E4C8.svg)](https://lynx-llm.github.io/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2307.02469-99D4C8.svg)](https://arxiv.org/abs/2307.02469)

**update**
- Jul 2023: Release preprint in [arXiv](https://arxiv.org/abs/2307.02469), and [page](https://lynx-llm.github.io/)


Lynx (8B parameters):
<div align="center">
  <img width="70%" src="images/lynx.png">
</div>

results on Open-VQA image testsets
<div align="center">
  <img width="70%" src="images/open_vqa_image_result.png">
</div>

results on Open-VQA video testsets && OwlEval human eval && MME benchmark
<div align="center">
  <img width="70%" src="images/result_other.png">
</div>

ablation result
<div align="center">
  <img width="70%" src="images/ablation.png">
</div>


## Quick Start

### environment
```angular2html
conda env create -f environment.yml
conda activate lynx
```

### prepare data
#### step 1: prepare annotation file
Open-VQA annotations file is under the path `data/Open_VQA_images.jsonl` and `data/Open_VQA_videos.jsonl`, there is an example:
```angular2html
{
  "dataset": "Open_VQA_images", # the dataset name of your data
  "question": "What is in the image?", 
  "answer": ["platform or tunnel"], # list
  "index": 1, 
  "image": "images/places365/val_256/Places365_val_00000698.jpg", # relative path of image
  "origin_dataset": "places365", 
  "class": "Place", # eight image VQA types and two video VQA types correspond to the open_VQA dataset
}
```
You can also convert your own data in jsonl format, the keys `origin_dataset` and `class` are optional.

#### step 2: prepare images
Download raw images from corresponding websites: [Places365(256x256)](http://places2.csail.mit.edu/download.html), [VQAv2](https://visualqa.org/download.html), [OCRVQA](https://ocr-vqa.github.io/), [Something-Something-v.2](https://developer.qualcomm.com/software/ai-datasets/something-something), [MSVD-QA](https://github.com/xudejing/video-question-answering), [NeXT-QA](https://github.com/doc-doc/NExT-QA) and [MSRVTT-QA](https://github.com/xudejing/video-question-answering).

#### step 3: modify the default setting in the code
You need the check some import settings in the configs `configs/LYNX.yaml`, for example:
```yaml
# change this prompt for different task, this is the default prompt
prompt: "User: {question}\nBot:"
# the key must match the vision key in test_files
# if you test Open_VQA_videos.jsonl, need to change to "video"
vision_prompt_dict: "image"
output_prompt_dict: "answer"
```


### prepare checkpoint
- step 1: download the `eva_vit_1b` on official [website](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA01_g_psz14.pt) and put it under the `data/`, rename it as `eva_vit_g.pth`
- step 2: prepare the `vicuna-7b` and put it under the `data/`
  - method 1: download from [huggingface](https://huggingface.co/lmsys/vicuna-7b-v1.1) directly.
  - method 2:
    - download Vicuna’s **delta** weight from [v1.1 version](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) (use git-lfs)
    - get `LLaMA-7b` from [here](https://huggingface.co/docs/transformers/main/model_doc/llama) or from the Internet.
    - install FastChat `pip install git+https://github.com/lm-sys/FastChat.git`
    - run `python -m fastchat.model.apply_delta --base /path/to/llama-7b-hf/  --target ./data/vicuna-7b/  --delta /path/to/vicuna-7b-delta-v1.1/`
- step 3: download the [pretrain_lynx.pt](https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/lynx_release/pretrain_lynx.pt) or [finetune_lynx.pt](https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/lynx_release/finetune_lynx.pt) and put it under the `data/`(please check the `checkpoint` in the config is match the file you download.)

organize the files like this:
```angular2html
lynx-llm/
    data/
        Open_VQA_images.jsonl
        Open_VQA_videos.jsonl
        eva_vit_g.pth
        vicuna-7b/
        finetune_lynx.pt
        pretrain_lynx.pt
    images/
        vqav2/val2014/*.jpg
        places365/val_256/*.jpg
        ocrvqa/images/*.jpg
        sthsthv2cap/val/*.mp4
        msvdqa/test/*.mp4
        nextqa/*.mp4
        msrvttqa/*.mp4
```

### infer
```angular2html
sh generate.sh
```

## Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@article{zeng2023matters,
  title={What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?},
  author={Zeng, Yan and Zhang, Hanbo and Zheng, Jiani and Xia, Jiangnan and Wei, Guoqiang and Wei, Yang and Zhang, Yuchen and Kong, Tao},
  journal={arXiv preprint arXiv:2307.02469},
  year={2023}
}
```


## Contact
For issues using this code, please submit a GitHub issue.


## License

This project is licensed under the [Apache-2.0 License](LICENSE).
