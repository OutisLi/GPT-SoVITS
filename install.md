# Installation

## Linux

```shell
git clone git@github.com:OutisLi/GPT-SoVITS.git
cd GPT-SoVITS
# by default: ModelScope & uvr5
./model_download.sh --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
conda create -n GPTSoVits python=3.12 -y
conda activate GPTSoVits

# ignore if you already have these
# conda install -c conda-forge gcc=14 -y
# conda install -c conda-forge gxx -y
# conda install ffmpeg cmake -y
# conda install git-lfs -y
# conda install zip -y
# sudo apt install ffmpeg
# sudo apt install libsox-dev

git-lfs install
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install faster-whisper
pip install -r requirements.txt
```

## Windows

```shell
git clone git@github.com:OutisLi/GPT-SoVITS.git
cd GPT-SoVITS

conda create -n GPTSoVits python=3.12 -y
conda activate GPTSoVits

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install faster-whisper
pip install -r requirements.txt
```

## ASR model

```shell
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --local_dir tools/asr/models
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --local_dir tools/asr/models
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch --local_dir tools/asr/models
```
