#!/usr/bin/env zsh

# cd into GPT-SoVITS Base Path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

set -e

if ! command -v conda &>/dev/null; then
    echo "Conda Not Found"
    exit 1
fi

trap 'echo "Error Occured at \"$BASH_COMMAND\" with exit code $?"; exit 1' ERR

is_HF=false
is_HF_MIRROR=false
is_MODELSCOPE=true
DOWNLOAD_UVR5=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    --source)
        case "$2" in
        HF)
            is_HF=true
            ;;
        HF-Mirror)
            is_HF_MIRROR=true
            ;;
        ModelScope)
            is_MODELSCOPE=true
            ;;
        *)
            echo "Error: Invalid Download Source: $2"
            echo "Choose From: [HF, HF-Mirror, ModelScope]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --download-uvr5)
        DOWNLOAD_UVR5=true
        shift
        ;;
    -h | --help)
        print_help
        exit 0
        ;;
    *)
        echo "Unknown Argument: $1"
        echo "Use -h or --help to see available options."
        exit 1
        ;;
    esac
done

if ! $is_HF && ! $is_HF_MIRROR && ! $is_MODELSCOPE; then
    echo "Error: Download Source is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if [ "$is_HF" = "true" ]; then
    echo "Download Model From HuggingFace"
    PRETRINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
elif [ "$is_HF_MIRROR" = "true" ]; then
    echo "Download Model From HuggingFace-Mirror"
    PRETRINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
elif [ "$is_MODELSCOPE" = "true" ]; then
    echo "Download Model From ModelScope"
    PRETRINED_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
    G2PW_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    UVR5_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip"
fi

if find "GPT_SoVITS/pretrained_models" -mindepth 1 ! -name '.gitignore' | grep -q .; then
    echo "Pretrained Model Exists"
else
    echo "Download Pretrained Models"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "$PRETRINED_URL"

    unzip pretrained_models.zip
    rm -rf pretrained_models.zip
    mv pretrained_models/* GPT_SoVITS/pretrained_models
    rm -rf pretrained_models
fi

if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo "Download G2PWModel"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "$G2PW_URL"

    unzip G2PWModel.zip
    rm -rf G2PWModel.zip
    mv G2PWModel GPT_SoVITS/text/G2PWModel
else
    echo "G2PWModel Exists"
fi

if [ "$DOWNLOAD_UVR5" = "true" ]; then
    if find "tools/uvr5/uvr5_weights" -mindepth 1 ! -name '.gitignore' | grep -q .; then
        echo "UVR5 Model Exists"
    else
        echo "Download UVR5 Model"
        wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "$UVR5_URL"

        unzip uvr5_weights.zip
        rm -rf uvr5_weights.zip
        mv uvr5_weights/* tools/uvr5/uvr5_weights
        rm -rf uvr5_weights
    fi
fi

echo "Model Downloads completed successfully!"
