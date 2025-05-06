import argparse
import os
import soundfile as sf

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav


def synthesize(
    GPT_model_path,
    SoVITS_model_path,
    ref_audio_path,
    ref_text,
    ref_language,
    target_text,
    target_language,
    output_path,
):
    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # # if the input audio is mp3, call ffmpeg to convert it to wav
    # if ref_audio_path.endswith(".mp3"):
    #     os.system(f"ffmpeg -i {ref_audio_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {ref_audio_path[:-4]}.wav")
    #     ref_audio_path = ref_audio_path[:-4] + ".wav"

    i18n = I18nAuto()
    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language),
        top_p=1,
        temperature=1,
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        os.makedirs(output_path, exist_ok=True)
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    # fmt: off
    parser.add_argument("--gpt_model", default="GPT_SoVITS/pretrained_models/s1v3.ckpt", help="Path to the GPT model file")
    parser.add_argument("--sovits_model", default="GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth", help="Path to the SoVITS model file")
    parser.add_argument("--ref_audio", default="assets/sample.wav", help="Path to the reference audio file")
    parser.add_argument("--ref_text", default="希望你以后能够做的比我还好呦。", help="Reference text")
    parser.add_argument("--ref_language", default="中文", choices=["中文", "英文", "日文"], help="Language of the reference audio")
    parser.add_argument("--target_text", default="不对不对，是八号，也就是说还有八加十五天还有二十三天，还有三周多你才能见到我呢", help="Target text")
    parser.add_argument("--target_language", default="中文", help='Language: "中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"')
    parser.add_argument("--output_path", default="output", help="Path to the output directory")
    # fmt: on

    args = parser.parse_args()

    synthesize(
        args.gpt_model,
        args.sovits_model,
        args.ref_audio,
        args.ref_text,
        args.ref_language,
        args.target_text,
        args.target_language,
        args.output_path,
    )


if __name__ == "__main__":
    main()
