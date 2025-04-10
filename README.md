# Predictive Modeling: BIM Command Recommendation Based on Large-Scale Usage Logs
This repository contains implementation of the Predictive Modeling paper: https://arxiv.org/abs/2504.05319

>The adoption of Building Information Modeling (BIM) and model-based design within the Architecture, Engineering, and Construction (AEC) industry has been hindered by the perception that using BIM authoring tools demands more effort than conventional 2D drafting. To enhance design efficiency, this paper proposes a BIM command recommendation framework that predicts the optimal next actions in real-time based on users' historical interactions. We propose a comprehensive filtering and enhancement method for large-scale raw BIM log data and introduce a novel command recommendation model. Our model builds upon the state-of-the-art Transformer backbones originally developed for large language models (LLMs), incorporating a custom feature fusion module, dedicated loss function, and targeted learning strategy. In a case study, the proposed method is applied to over 32 billion rows of real-world log data collected globally from the BIM authoring software Vectorworks. Experimental results demonstrate that our method can learn universal and generalizable modeling patterns from anonymous user interaction sequences across different countries, disciplines, and projects. When generating recommendations for the next command, our approach achieves a Recall@10 of approximately 84%.

All experiments were done on a Linux machine with a Quadro RTX 8000 GPU (and WSL).

![demo](command_prediction.gif)

Clone the repo with submodule:
```
git clone --recurse-submodules git@github.com:dcy0577/BIM-Command-Recommendation.git
```

## Install conda env

```
# create a conda env
# may need run 'export CUDA_HOME=/usr/local/cuda'
# may need run 'export PATH=$CUDA_HOME/bin:$PATH' for pip install error
conda env create -f t4rec_new_transformer.yml

# export env if needed
conda env export --no-builds > t4rec_new_transformer.yml
```

## Repository structure

- `data_processing` contains implementation of data filtering and enhancement pipeline. Please refer to the README file inside for more details.
- `model` contains the training and evaluation scripts and is built on `transformers4rec` and `peft`, which are custom versions of the original code base from Nvidia and Huggingface. Please refer to the README file inside for more details.
- `deployment` contains scripts for ensembling and deploying trained models on the Triton inference server, as well as launching the services. Please refer to the README file inside for more details.
- `prototype` contains the implementation of the web application, which polls the Vectorworks log and generates the next best command suggestion in real time. Please refer to the README file inside for more details.
- `visualization` contains the visualization scripts that generates figures for anlaysis in the paper.


## TODO
1. Flash attention doesn't work on current GPU
2. Somehow the model.pt of Mixtral will use both gpu and cpu on triton, which will cause errors... but current Python backend works on triton.
3. Try loftq or qa-lora in peft, need to update the peft version
4. Model Parallelism using deepspeed for distributed training? https://github.com/deepspeedai/DeepSpeed
4. It seems that increasing the size of the input feature dimension is beneficial for the model. Large models benefit even more!


## Acknowledgement
The implementation is based on https://github.com/NVIDIA-Merlin/Transformers4Rec, https://github.com/huggingface/transformers, and https://github.com/huggingface/peft 

## Citation
```
@misc{du2025predictivemodelingbimcommand,
      title={Predictive Modeling: BIM Command Recommendation Based on Large-scale Usage Logs}, 
      author={Changyu Du and Zihan Deng and Stavros Nousias and André Borrmann},
      year={2025},
      eprint={2504.05319},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2504.05319}, 
}
```