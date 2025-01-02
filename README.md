# MLLM4Rec : multimodal information enhancing LLM for sequential recommendation
The source code for our **Journal of Intelligent Information Systems(2024)** Paper [[MLLM4Rec](https://doi.org/10.1007/s10844-024-00915-3)]


<img src=media/MLLMRec.png width=1000>



In this work, we propose a framework to enhance the recommendation performance of large language models using multimodal information, which we call **M**ultimodal Information Enhancing **L**arge **L**anguage **M**odel **F**or Sequential **Rec**ommendation (MLLM4Rec). Specifically, we first use text to describe each image, a process that helps bridge the gap between text and image modality representations. Subsequently, to enable the large language model to better learn user interaction features, we propose a hybrid prompt learning method and introduce role-playing to assist in fine-tuning the model. Finally, the fine-tuned large language model retains its generative capabilities while learning to rank items based on user preferences. Experiments on four public benchmark datasets demonstrate that our proposed MLLM4Rec framework outperforms partially LLM-based recommendation models, traditional sequential recommendation models, and sequential recommendation models pre-trained with multi-modal information.


## Dependencies 
Python 3.9.18

- torch==2.0.0
- transformers==4.33.3
- peft==0.5.0
- bitsandbytes==0.41.2.post2


For our detailed running environment see requirement.txt.

The computer platform we are using is win10Pro 12700kf+32G DDR4+RTX4090

When the model is running, it may require 23G of video memory

## Dataset and LLM Download
### Step 1
Download the dataset and images.

You need to switch the dataset manually, the options are "beauty", "toys", "games", "ml-100k", modify the **dataset_code** parameter.

```bash
python process_item.py --dataset_code beauty
```

### Step 2
Download blip2 from [Hugging Face](https://huggingface.co/Salesforce/blip2-opt-2.7b). 

Using blip2 to process images into text.

You need to switch the dataset manually, the options are "beauty", "toys", "games", "ml-100k", modify the **dataset_code** parameter.

Also, you need to change the path where the blip2 model is stored, modify the **model_path** parameter.
```bash
python process_item_blip2.py --dataset_code beauty --model_path BLIP2_PATH
```

### Step 3
Download llama2 from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf).

The first time you download the model, you may need Hugging Face's review.

## Training
### Step 1
The command below starts the training of the retriever model LRURec

```bash
python train_retriever.py --dataset_code beauty
```
You can set additional arguments like **weight_decay** to change the hyperparameters. 

You need to switch the dataset manually, the options are "beauty", "toys", "games", "ml-100k", modify the **dataset_code** parameter.

Once training is finished, evaluation is automatically performed with the best retriever model.

### Step 2
run the following command to train the ranker model based on Llama 2

You need to specify the following parameters:
- dataset_code: must run the same as train_retriever.
- llm_base_model: storage path for the llama2 model.

```bash
python train_ranker.py --dataset_code beauty --llm_base_model LLAMA2_PATH
```

All weights and results are saved under ./experiments.

## Contact
If you have any questions related to the code or the paper, feel free to create an issue or email Yuxiang Wang(wangyuxiang123@outlook.com), the corresponding author of the 
**Journal of Intelligent Information Systems** paper. Thanks!


## Cite 
If you find this repo useful, please cite
```
@article{wang2024mllm4rec,
  title={MLLM4Rec: multimodal information enhancing LLM for sequential recommendation},
  author={Wang, Yuxiang and Shi, Xin and Zhao, Xueqing},
  journal={Journal of Intelligent Information Systems},
  pages={1--17},
  year={2024},
  publisher={Springer}
}

@article{yuxiang2024mllm4rec,
  title={MLLM4Rec: Multimodal Information Enhancing LLM for Sequential Recommendation},
  author={Yuxiang, Wang and Xin, Shi and Xueqing, Zhao},
  year={2024}
}
```

## Also thanks
The code is implemented based on [LlamaRec](https://github.com/yueeeeeeee/llamarec?tab=readme-ov-file)
