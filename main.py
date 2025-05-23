import os
import argparse
import soundfile as sf


from cli.GPTSovitsTTS import GPTSovitsTTS


def synthesize(ref_audio_path, ref_text, target_text, output_path, speed):
    tts = GPTSovitsTTS()

    # Synthesize audio
    ref_artifacts = tts.process_reference_audio(ref_audio_path, ref_text) # Example pause
    synthesis_result = tts.synthesize_target_speech(ref_artifacts, target_text)

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        os.makedirs(output_path, exist_ok=True)
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")
    else:
        print("Synthesis failed to produce audio.")


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    # fmt: off
    parser.add_argument("--ref_audio", default="assets/sample.wav", help="Path to the reference audio file")
    parser.add_argument("--ref_text", default="希望你以后能够做的比我还好呦。", help="Reference text")
    parser.add_argument("--target_text", default="Hi! Today is Monday. ", help="Target text")
    parser.add_argument("--output_path", default="output", help="Path to the output directory")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the synthesized audio")
    # fmt: on

    args = parser.parse_args()

    input_text = r"“在今天的大众媒体和图书市场上，到处充斥着关于潜能提升、心理操控、色彩星座、催眠读心等伪装成心理学的主题，\n更有一些伪心理学家、所谓的心理治疗师打着心理学的旗号欺世盗名，从中渔利。\n在浩如烟海、良莠不齐的心理学信息面前，如何拨除迷雾，去伪存真，成为一个明智的心理学信息的消费者呢？\n这本书将教给你科学实用的批判性思维技能，将真正的心理学研究从伪心理学中区分出来，告诉你什么才是真正的心理学。\n\n本书第1版出版于1983年，30多年来一直被奉为心理学入门经典，在全球顶尖大学中享有盛誉，现在呈现在读者面前的是第11版。\n这本书并不同于一般的心理学导论类教材，很多内容是心理学课堂上不曾讲授的，也是许多心理学教师在教学中感到只可意会不可言传的。\n作者正是从此初衷出发，以幽默生动的语言，结合一些妙趣横生、贴近生活的实例，深入浅出地介绍了可证伪性、操作主义、实证主义、安慰剂效应\n\nExcerpt From这才是心理学：看穿伪科学的批判性思维 (第11版)\n基思·斯坦诺维奇This material may be protected by copyright."
    args.target_text = input_text

    synthesize(args.ref_audio, args.ref_text, args.target_text, args.output_path, args.speed)


if __name__ == "__main__":
    main()
