# MLAN: Language-Based Instruction Tuning Improves Zero-Shot Generalization of Multimodal Large Language Models
📃 [Paper](https://arxiv.org/abs/2411.10557) • 💻 [Github](https://github.com/wang-research-lab/mlan) • 🤗 [HuggingFace](https://huggingface.co/collections/WangResearchLab/mlan-673be70728a904fca2c2a661) • 🗂️ [Dataset](https://huggingface.co/datasets/WangResearchLab/MLAN)

<img src="images/logo.png" alt="MLAN Logo" width="250">


## ⚒️ Installation

Our training code is built upon the [LLaVA repo](https://github.com/haotian-liu/LLaVA).

1. Clone this repository
```
git clone https://github.com/wang-research-lab/mlan
cd mlan
```

2. Install the training packages. *Note: May need to call ```pip install wheel``` and/or set ```FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE``` for flash-attn to build.*
```
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

3. Install our modified evaluation packages
```
pip install git+https://github.com/wang-research-lab/lm_eval.git
pip install git+https://github.com/wang-research-lab/lmms-eval.git
```

## 📖 Data Preparation

The text and image data can be accessed directly through our Huggingface repository. You should download them into the `playground/data` folder. The following script automatically downloads the pretraining and finetuning data into `playground/data` for you.

```
bash scripts/prepare_data.sh 
```

[MLAN_80k](https://huggingface.co/datasets/WangResearchLab/MLAN/resolve/main/MLAN_80k.json): contains 80k **language-only** instruction tuning data collected from public datasets.

[MLAN_v_50l_80k](https://huggingface.co/datasets/WangResearchLab/MLAN/resolve/main/MLAN_v_50l_80k.json): contains 40k **language-only** and 40k **vision-language** instruction following data for Vicuna series models.

[MLAN_v_88l_80k](https://huggingface.co/datasets/WangResearchLab/MLAN/resolve/main/MLAN_v_88l_80k.json): contains 70k **language-only** and 10k **vision-language** instruction following data for pretrained LLaMA2 models.

[images_mlan_v](https://huggingface.co/datasets/WangResearchLab/MLAN/resolve/main/images_mlan_v.zip): contains the corresponding images for MLAN_v_80k.

## 🏋️‍♂️ Train

MLAN training consists of 2 phases:

1. Feature alignment: we use LLaVA-CC3M-Pretrain-595K to make the visual encoder outputs compatible with the base language model.
2. Supervised finetuning: we use our MLAN_80k or MLAN_v_80k to instruction tune the language model and the projector.

### Pretraining

Pretraining takes around 3.5 hours for a 7B model. Our experiments are conducted on single nodes with 8xA6000 (48G) or 4xA100 (80G). Please note that the global batch size (num_gpus * per_device_batchsize * gradient_accumulation_steps) needs to be kept the same.

```
bash scripts/pretrain.sh
```

### Instruction Tuning

Thanks to the reduced usage of image inputs, finetuning with MLAN takes under 1 hour and with MLAN_v takes under 2 hours on 8xA6000. 

```
bash scripts/finetune.sh
```

## 💾 Checkpoints

For evaluation purposes, we release our checkpoints for Llama 2 and Vicuna 1.5 fine-tuned with MLAN and MLAN_v on our Huggingface repo.

| Setting           | Model                  | Link                                                   |
| ----------------- | ---------------------- | -------------------------------------------------------|
MLAN (Llama 2) | llava-mlan-llama2-7b   | <https://huggingface.co/WangResearchLab/llava-mlan-llama2-7b>   |
MLAN (Vicuna) | llava-mlan-vicuna-7b   | <https://huggingface.co/WangResearchLab/llava-mlan-vicuna-7b>   |
MLAN_v (Llama 2) | llava-mlan-v-llama2-7b | <https://huggingface.co/WangResearchLab/llava-mlan-v-llama2-7b> |
MLAN_v (Vicuna) | llava-mlan-v-vicuna-7b | <https://huggingface.co/WangResearchLab/llava-mlan-v-vicuna-7b> |

When you directly specify the model in the evaluation script (e.g., `MODEL=WangResearchLab/llava-mlan-llama2-7b`), it will automatically download the weights. Note for this to work, you may need to use huggingface-cli to login prior to running the evaluation scripts.


## 📝 Evaluation

Our testing environments are built upon lm-eval and lmms-eval platforms, for language-only and vision-language tasks respectively. We use customized answer parsers to extract short answers. Take a look at the task definitions written in the `scripts/eval/custom` directory for more information. Note that evaluation scripts by default run on only one GPU and thus may take long (~1 hour) to complete with the default settings.

To evaluate on the datasets used in our paper, run the following commands with the desired model:
```
MODEL={MODEL_NAME} bash scripts/eval/lm-eval.sh  # for language-only datasets
MODEL={MODEL_NAME} bash scripts/eval/lmm-eval.sh  # for vision-language datasets
```

## Citations
```
@misc{tu2024mlan,
      title={MLAN: Language-Based Instruction Tuning Improves Zero-Shot Generalization of Multimodal Large Language Models}, 
      author={Jianhong Tu and Zhuohao Ni and Nicholas Crispino and Zihao Yu and Michael Bendersky and Beliz Gunel and Ruoxi Jia and Xin Liu and Lingjuan Lyu and Dawn Song and Chenguang Wang},
      year={2024},
      eprint={2411.10557},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.10557}, 
}
```

# Acknowledgement
1. [LLaVA](https://github.com/haotian-liu/LLaVA): our code is built upon their wonderful scripts.
2. [LM-EVAL](https://github.com/EleutherAI/lm-evaluation-harness): we customized their pipeline for language evaluation.
3. [LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval): we customized their pipeline for vision evaluation. 
