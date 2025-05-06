import os
import argparse
import soundfile as sf


from cli.cli_utils import load_gpt_weights, load_sovits_weights, get_tts_wav


def synthesize(ref_audio_path, ref_text, target_text, output_path, speed):
    # Change model weights
    load_gpt_weights("models/GPT_SoVITS_v4/s1v3.ckpt")
    load_sovits_weights("models/GPT_SoVITS_v4/s2Gv4.pth")

    # Synthesize audio
    # 默认中英文混合输入输出
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        text=target_text,
        speed=speed,
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
    parser.add_argument("--ref_audio", default="assets/sample.wav", help="Path to the reference audio file")
    parser.add_argument("--ref_text", default="希望你以后能够做的比我还好呦。", help="Reference text")
    parser.add_argument("--target_text", default="不对不对，是八号，也就是说还有八加十五天还有二十三天，还有三周多你才能见到我呢, Hi! Today is Monday.", help="Target text")
    parser.add_argument("--output_path", default="output", help="Path to the output directory")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the synthesized audio")
    # fmt: on

    args = parser.parse_args()

    synthesize(args.ref_audio, args.ref_text, args.target_text, args.output_path, args.speed)


if __name__ == "__main__":
    main()
