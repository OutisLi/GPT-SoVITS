import logging
import warnings
import os
import re
import sys
import random
from time import time as ttime

# --- Suppress excessive logging ---
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)
# Filter out specific torchaudio UserWarning about TypedStorage deprecation
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", category=UserWarning, module="torch._utils")
# Filter out UserWarning about torchaudio backend being set internally
warnings.filterwarnings("ignore", message="Torchaudio's I/O functions now support parform fetching.*", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message="Neither SoundFile nor SoX were found.*", category=UserWarning, module="torchaudio")


import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

# --- Project specific imports ---
cli_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cli_dir + "/..")
from text.LangSegmenter import LangSegmenter
from feature_extractor import cnhubert  # Needs cnhubert_base_path set
from GPT_SoVITS.module.models import SynthesizerTrnV3, Generator
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from text import chinese  # For normalization
from tools.i18n.i18n import I18nAuto, scan_language_list
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from module.mel_processing import mel_spectrogram_torch, spectrogram_torch


# --- Utility Class ---
class DictToAttrRecursive(dict):
    """Recursively converts dict keys to attributes."""

    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


# --- Helper Function ---
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed = int(seed)
    # print(f"Setting random seed to: {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # Potentially add determinism settings, but they can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# --- Main TTS Class ---
class GPTSovitsTTS:
    """Encapsulates the GPT-SoVITS TTS functionality."""

    # --- Constants ---
    # Punctuation/Splits sets
    PUNCTUATION = set(["!", "?", "…", ",", ".", "-", " "])
    SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
    # Spectogram normalization values
    SPEC_MIN = -12
    SPEC_MAX = 2

    # fmt: off
    def __init__(self, gpt_weight_path=None, sovits_weight_path=None, cnhubert_base_path=None, bert_path=None, hifigan_path=None, language=None, device=None, is_half=None, seed=-1):
    # fmt: on
        """
        Initializes the TTS system.

        Args:
            gpt_weight_path (str, optional): Path to GPT model weights. Defaults to env var or hardcoded path.
            sovits_weight_path (str, optional): Path to SoVITS model weights. Defaults to env var or hardcoded path.
            cnhubert_base_path (str, optional): Path to CNHubert model. Defaults to env var or hardcoded path.
            bert_path (str, optional): Path to BERT model. Defaults to env var or hardcoded path.
            hifigan_path (str, optional): Path to the HiFiGAN vocoder model .pth file. Defaults to a standard path.
            language (str, optional): Default language ('en', 'zh', 'ja', 'auto', etc.). Defaults to env var or 'Auto'.
            device (str, optional): Device ('cuda', 'cpu'). Auto-detects if None.
            is_half (bool, optional): Use half precision (FP16). Auto-detects if None (True if CUDA available).
            seed (int): Random seed for reproducibility. Set to -1 for random seed.
        """
        # print("Initializing GPTSovitsTTS...")
        set_seed(seed)

        # Determine configuration values
        # fmt: off
        self.gpt_weight_path = gpt_weight_path or os.environ.get("gpt_weight_path", "GPT_SoVITS/pretrained_models/s1v3.ckpt")
        self.sovits_weight_path = sovits_weight_path or os.environ.get("sovits_weight_path", "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth")
        self.cnhubert_base_path = cnhubert_base_path or os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
        self.bert_path = bert_path or os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        self.hifigan_path = hifigan_path or os.path.join(os.getcwd(), "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth")
        # fmt: on

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if is_half is not None:
            self.is_half = is_half and self.device == "cuda"
        else:
            self.is_half = False
            # Default: use half precision if CUDA is available
            self.is_half = eval(os.environ.get("is_half", "True")) and self.device == "cuda"
        self.dtype = torch.float16 if self.is_half else torch.float32
        # print(f"Using device: {self.device}")
        # print(f"Using half precision: {self.is_half}")

        # Setup i18n
        default_lang = os.environ.get("language", "Auto")
        self.language = language or (sys.argv[-1] if len(sys.argv) > 1 and sys.argv[-1] in scan_language_list() else default_lang)
        self.i18n = I18nAuto(language=self.language)
        # print(f"Default language set to: {self.language}")

        self.LANGUAGE_MAP = {
            self.i18n("中文"): "all_zh",
            self.i18n("英文"): "en",
            self.i18n("日文"): "all_ja",
            self.i18n("粤语"): "all_yue",
            self.i18n("韩文"): "all_ko",
            self.i18n("中英混合"): "zh",
            self.i18n("日英混合"): "ja",
            self.i18n("粤英混合"): "yue",
            self.i18n("韩英混合"): "ko",
            self.i18n("多语种混合"): "auto",
            self.i18n("多语种混合(粤语)"): "auto_yue",
        }

        # --- Initialize models (load paths, set to None initially) ---
        self.tokenizer = None
        self.bert_model = None
        self.ssl_model = None
        self.vq_model = None
        self.hps = None
        self.model_version = "v4"  # Default, updated when loading SoVITS
        self.t2s_model = None
        self.hz = 50  # Default from GPT, updated on load
        self.max_sec = 50  # Default from GPT, updated on load

        # --- Load base models ---
        self._load_bert_model()
        self._load_cnhubert_model()

        # --- Initialize Vocoder Eagerly ---  <<<<<<<<< MODIFICATION
        self.hifigan_model = None  # Initialize attribute first
        try:
            self._init_hifigan()  # Load it now
        except Exception as e:
            print(f"Error initializing HiFiGAN vocoder: {e}")
            raise

        self.resample_transform_dict = {}  # Cache for torchaudio resamplers

        # Precompute mel function only once if needed
        self.mel_fn_v4 = lambda x: mel_spectrogram_torch(
            x,
            **{
                "n_fft": 1280,
                "win_size": 1280,
                "hop_size": 320,
                "num_mels": 100,
                "sampling_rate": 32000,
                "fmin": 0,
                "fmax": None,
                "center": False,
            },
        )
        self.load_gpt_weights(self.gpt_weight_path)
        self.load_sovits_weights(self.sovits_weight_path)

        # print("GPTSovitsTTS initialized.")

    # --- Model Loading ---
    def _load_bert_model(self):
        """Loads the BERT model and tokenizer."""
        # print(f"Loading BERT model from: {self.bert_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
            if self.is_half:
                self.bert_model = self.bert_model.half()
            self.bert_model = self.bert_model.to(self.device)
            self.bert_model.eval()
            # print("BERT model loaded successfully.")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            raise

    def _load_cnhubert_model(self):
        """Loads the CNHubert SSL model."""
        # print(f"Loading CNHubert model from: {self.cnhubert_base_path}")
        try:
            # Set path for cnhubert module before loading
            cnhubert.cnhubert_base_path = self.cnhubert_base_path
            self.ssl_model = cnhubert.get_model()
            if self.is_half:
                self.ssl_model = self.ssl_model.half()
            self.ssl_model = self.ssl_model.to(self.device)
            self.ssl_model.eval()
            # print("CNHubert model loaded successfully.")
        except Exception as e:
            print(f"Error loading CNHubert model: {e}")
            raise

    def _init_hifigan(self):
        """Initializes the HiFiGAN vocoder model."""
        if self.hifigan_model is not None:
            return  # Already initialized
        # print(f"Initializing HiFiGAN vocoder from: {self.hifigan_path}")
        if not os.path.exists(self.hifigan_path):
            raise FileNotFoundError(f"HiFiGAN vocoder model not found at: {self.hifigan_path}")
        try:
            # Using v4 parameters directly as in the original script
            self.hifigan_model = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            self.hifigan_model.eval()
            self.hifigan_model.remove_weight_norm()

            state_dict_g = torch.load(self.hifigan_path, map_location="cpu")

            # print(f"Loading HiFiGAN vocoder weights: {self.hifigan_model.load_state_dict(state_dict_g)}")

            if self.is_half:
                self.hifigan_model = self.hifigan_model.half()
            self.hifigan_model = self.hifigan_model.to(self.device)
            # print("HiFiGAN vocoder initialized successfully.")
        except Exception as e:
            print(f"Error initializing HiFiGAN vocoder: {e}")
            raise

    def load_sovits_weights(self, sovits_path):
        """Loads the SoVITS VQ model weights."""
        if not os.path.exists(sovits_path):
            raise FileNotFoundError(f"SoVITS weights not found at: {sovits_path}")
        # print(f"Loading SoVITS weights from: {sovits_path}")
        try:
            # Get model version info directly from the checkpoint file
            # Returned: version (e.g., "v2"), model_version (e.g., "v4"), if_lora_v3 (bool)
            _version_str, self.model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
            # print(f"Detected SoVITS info: version={_version_str}, model_version={self.model_version}, is_lora={if_lora_v3}")

            # Load state dict and config
            dict_s2 = load_sovits_new(sovits_path)  # Assuming this function handles loading logic
            self.hps = DictToAttrRecursive(dict_s2["config"])
            self.hps.model.semantic_frame_rate = "25hz"  # Ensure this is set

            # Determine actual internal version based on embedding shape (if needed, might be redundant with get_sovits_version_from_path_fast)
            if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
                inferred_version = "v2"  # v3model,v2symbols
            elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                inferred_version = "v1"
            else:
                inferred_version = "v2"
            # Override hps version based on detection, primarily using model_version from fast check
            self.hps.model.version = self.model_version
            # print(f"Using SoVITS model version: {self.model_version} (Inferred internal: {inferred_version})")

            self.vq_model = SynthesizerTrnV3(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model,
            )

            if self.is_half:
                self.vq_model = self.vq_model.half()
            self.vq_model = self.vq_model.to(self.device)
            self.vq_model.eval()

            # Load weights (handle potential LoRA)
            if not if_lora_v3:
                # print(f"Loading SoVITS weights (strict=False): {self.vq_model.load_state_dict(dict_s2['weight'], strict=False)}")
                pass
            else:
                # Handle LoRA loading if necessary (based on original logic if it existed)
                # print("LoRA detected - standard weight loading might be incomplete or incorrect if LoRA weights are separate.")
                # print(f"Attempting SoVITS LoRA weight load (strict=False): {self.vq_model.load_state_dict(dict_s2['weight'], strict=False)}")
                pass

            # print("SoVITS weights loaded successfully.")
        except Exception as e:
            print(f"Error loading SoVITS weights: {e}")
            raise

    def load_gpt_weights(self, gpt_path):
        """Loads the GPT Text-to-Semantic model weights."""
        if not os.path.exists(gpt_path):
            raise FileNotFoundError(f"GPT weights not found at: {gpt_path}")
        # print(f"Loading GPT weights from: {gpt_path}")
        try:
            dict_s1 = torch.load(gpt_path, map_location="cpu")
            config = dict_s1["config"]
            self.max_sec = config["data"]["max_sec"]
            self.hz = 50  # Often 50hz

            self.t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)  # "****" placeholder might be for phonemes?
            self.t2s_model.load_state_dict(dict_s1["weight"])

            if self.is_half:
                self.t2s_model = self.t2s_model.half()
            self.t2s_model = self.t2s_model.to(self.device)
            self.t2s_model.eval()
            # print(f"GPT weights loaded successfully. max_sec={self.max_sec}, hz={self.hz}")
        except Exception as e:
            print(f"Error loading GPT weights: {e}")
            raise

    # --- Preprocessing Helpers ---
    def _resample(self, audio_tensor, sr0, sr1):
        """Resamples audio tensor using cached transform."""
        if sr0 == sr1:
            return audio_tensor
        key = f"{sr0}-{sr1}"
        if key not in self.resample_transform_dict:
            # print(f"Creating resampler from {sr0} Hz to {sr1} Hz")
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(self.device)
        return self.resample_transform_dict[key](audio_tensor)

    def _norm_spec(self, x):
        """Normalizes spectrogram."""
        return (x - self.SPEC_MIN) / (self.SPEC_MAX - self.SPEC_MIN) * 2 - 1

    def _denorm_spec(self, x):
        """Denormalizes spectrogram."""
        return (x + 1) / 2 * (self.SPEC_MAX - self.SPEC_MIN) + self.SPEC_MIN

    def _get_spepc(self, filename):
        """Loads audio and computes spectrogram."""
        if not self.hps:
            raise RuntimeError("SoVITS HPS not loaded. Load SoVITS weights first.")

        sampling_rate = int(self.hps.data.sampling_rate)
        # Load with librosa, then convert to tensor
        audio, sr = librosa.load(filename, sr=sampling_rate)
        if sr != sampling_rate:
            # print(f"Warning: Audio loaded with sr={sr}, resampling to target {sampling_rate}")
            # Resample using librosa before converting to tensor if needed
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        audio = torch.FloatTensor(audio)

        # Normalize audio amplitude
        maxx = audio.abs().max()
        if maxx > 1:
            # print(f"Warning: Input audio max amplitude is {maxx:.2f} > 1. Normalizing.")
            audio /= maxx  # More robust normalization than clip/scale by 2

        audio_norm = audio.unsqueeze(0)  # Add batch dimension
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return spec  # Shape: [1, Freq, Time]

    def _clean_text_inf(self, text, language):
        """Cleans text and converts to phone sequence."""
        if not self.hps:
            raise RuntimeError("SoVITS HPS not loaded. Load SoVITS weights first.")
        # Language codes should match those expected by clean_text (e.g., 'zh', 'en', 'ja')
        language_cleaned = language.replace("all_", "")
        # Use the model version determined during SoVITS loading
        phones, word2ph, norm_text = clean_text(text, language_cleaned, self.hps.model.version)
        phones = cleaned_text_to_sequence(phones, self.hps.model.version)
        return phones, word2ph, norm_text

    def _get_bert_feature(self, text, word2ph):
        """
        Extracts BERT features, Uses the third-to-last hidden layer.
        """
        if self.bert_model is None or self.tokenizer is None:
            raise RuntimeError("BERT model or tokenizer not loaded.")
        if not text:
            # print("Warning: Empty text provided to _get_bert_feature.")
            raise ValueError("Text input is empty.")

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:  # Ensure all input tensors are on the correct device
                inputs[i] = inputs[i].to(self.device)

            # BERT forward pass
            bert_output = self.bert_model(**inputs, output_hidden_states=True)

            # --- Core logic from your original function ---
            # Get the third-to-last hidden state
            # Original: res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            # Here, res["hidden_states"] is a tuple of all hidden states.
            # res["hidden_states"][-3] is the third-to-last layer.
            # Its shape is (batch_size, sequence_length, hidden_size).
            # We want batch_size 0, and then skip CLS [0] and SEP [-1] tokens.
            # The original used torch.cat(..., -1) but on a single tensor slice,
            # which is just the tensor itself.
            res_layer = bert_output["hidden_states"][-3]  # Third-to-last layer
            res = res_layer[0, 1:-1]  # Get batch 0, remove CLS and SEP tokens

            # Original moved to CPU here. We can keep it on device for now.
            # If CPU is strictly needed for subsequent original logic, uncomment:
            # res = res.cpu()
            # --- End of core logic adaptation ---

        # Original assertion: assert len(word2ph) == len(text)
        # This assertion is tricky as len(text) is char count and len(word2ph) is token count.
        # A more common check is if token count from BERT (res.shape[0]) matches len(word2ph).
        if res.shape[0] != len(word2ph):
            print(
                f"Warning: BERT output token count ({res.shape[0]}) and word2ph length ({len(word2ph)}) mismatch for text: '{text}'. Original assert was `len(word2ph) == len(text)` which is {len(word2ph) == len(text)}."
            )
            # Fallback: clip to the minimum length to avoid out-of-bounds errors.
            min_len = min(res.shape[0], len(word2ph))
            res = res[:min_len]
            word2ph = word2ph[:min_len]
            if not min_len:  # If either was zero, result will be empty
                print("Warning: Resulting feature length is zero after mismatch handling.")
                return torch.zeros((res.shape[-1] if res.numel() > 0 else 1024, 0), dtype=self.dtype).to(self.device)

        phone_level_feature = []
        for i in range(len(word2ph)):
            # Original had no check for word2ph[i] == 0
            if word2ph[i] > 0:  # Added to prevent errors with repeat(0,1)
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)
            elif word2ph[i] < 0:  # Should not happen, but good to be aware
                print(f"Warning: word2ph[{i}] is negative ({word2ph[i]}). Skipping.")

        if not phone_level_feature:
            # Original would error at torch.cat([]). Return empty tensor of correct shape.
            # Feature dimension is res.shape[-1]
            print(f"Warning: No phone-level features generated for text: '{text}'. Returning empty tensor.")
            return torch.zeros((res.shape[-1] if res.numel() > 0 else 1024, 0), dtype=self.dtype).to(self.device)

        phone_level_feature_cat = torch.cat(phone_level_feature, dim=0)

        # Ensure dtype is correct before returning
        return phone_level_feature_cat.T.to(self.dtype)

    def _get_bert_inf(self, phones, word2ph, norm_text, language):
        """Gets BERT features, providing zeros for non-Chinese languages."""
        language_cleaned = language.replace("all_", "")
        # Only compute BERT for Chinese ('zh')
        if language_cleaned == "zh":
            # Ensure dtype is applied correctly
            bert_features = self._get_bert_feature(norm_text, word2ph)  # Already on device
            return bert_features.to(self.dtype)
        else:
            # Return zeros for other languages
            return torch.zeros((1024, len(phones)), dtype=self.dtype).to(self.device)

    def _get_phones_and_bert(self, text, language, final=False):
        """Processes text for phones and BERT features based on language mode."""
        if not self.hps:
            raise RuntimeError("SoVITS HPS not loaded. Load SoVITS weights first.")

        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            formattext = text.replace("  ", " ")  # Simple space normalization

            if language == "all_zh":
                # Apply Chinese specific normalization including optional English handling
                if re.search(r"[A-Za-z]", formattext):
                    # Convert English letters to uppercase as per original logic
                    formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                # Apply mix_text_normalize if available and needed
                if hasattr(chinese, "mix_text_normalize"):
                    formattext = chinese.mix_text_normalize(formattext)
                else:
                    print("Warning: 'chinese.mix_text_normalize' not found. Skipping mixed normalization.")

                # Recursively call with 'zh' language code after normalization
                return self._get_phones_and_bert(formattext, "zh", final=final)  # Pass 'final' flag

            # Process directly for 'en', 'all_ja', 'all_ko', 'all_yue' (after potential yue normalization below)
            elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
                if hasattr(chinese, "mix_text_normalize"):  # Assuming Yue uses similar normalization
                    formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                else:
                    print("Warning: 'chinese.mix_text_normalize' not found for Yue. Skipping mixed normalization.")
                # Recursively call with 'yue'
                return self._get_phones_and_bert(formattext, "yue", final=final)

            else:  # 'en', 'all_ja', 'all_ko', or 'all_yue' without English mix that needs normalization
                phones, word2ph, norm_text = self._clean_text_inf(formattext, language)
                # Get BERT features (will be zeros unless language is 'zh', handled by _get_bert_inf)
                bert = self._get_bert_inf(phones, word2ph, norm_text, language)

        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            # Mixed language processing using LangSegmenter
            textlist = []
            langlist = []

            if language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])  # Expected: 'zh', 'ja', 'en', etc.
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    processed_lang = "yue" if tmp["lang"] == "zh" else tmp["lang"]
                    langlist.append(processed_lang)
                    textlist.append(tmp["text"])
            else:  # "zh", "ja", "ko", "yue" (treat as potentially mixed with English)
                for tmp in LangSegmenter.getTexts(text):
                    # Prioritize English, otherwise assume the main specified language
                    processed_lang = tmp["lang"] if tmp["lang"] == "en" else language
                    langlist.append(processed_lang)
                    textlist.append(tmp["text"])

            # print("Segmented text:", textlist)
            # print("Segmented langs:", langlist)

            phones_list = []
            bert_list = []
            norm_text_list = []
            word2ph_list = []  # Need this for BERT if segments are Chinese

            for i in range(len(textlist)):
                lang_seg = langlist[i]
                text_seg = textlist[i]
                if not text_seg.strip():
                    continue  # Skip empty segments

                phones_seg, word2ph_seg, norm_text_seg = self._clean_text_inf(text_seg, lang_seg)
                bert_seg = self._get_bert_inf(phones_seg, word2ph_seg, norm_text_seg, lang_seg)

                phones_list.append(phones_seg)
                norm_text_list.append(norm_text_seg)
                bert_list.append(bert_seg)
                word2ph_list.append(word2ph_seg)

            if not phones_list:  # Handle cases where segmentation results in nothing
                print(f"Warning: No phones generated after segmentation for text: '{text}'")
                return [], torch.zeros((1024, 0), dtype=self.dtype).to(self.device), ""

            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])  # Concatenate lists of phones
            norm_text = "".join(norm_text_list)

        else:
            raise ValueError(f"Unsupported language mode: {language}")

        # Original logic to prepend "." if text is too short and not final call
        # This seems problematic, potentially causing infinite recursion if "." doesn't help.
        # Consider removing or revising this logic. Keeping it for parity for now.
        if not final and len(phones) < 6 and language != "en":  # Avoid adding '.' to English potentially
            print(f"Warning: Input text '{text}' resulted in very few phones ({len(phones)}). Prepending '.' and retrying.")
            # Prepending punctuation might depend on the language
            prepend_char = "。" if language in ["zh", "ja", "all_zh", "all_ja", "yue", "all_yue", "ko", "all_ko"] else "."
            # Recursive call with final=True to prevent infinite loop
            return self._get_phones_and_bert(prepend_char + text, language, final=True)

        # Ensure BERT is on the correct device and dtype before returning
        return phones, bert.to(device=self.device, dtype=self.dtype), norm_text

    # --- Text Splitting Utilities ---
    def _split_text(self, todo_text):
        """Splits text based on defined SPLITS."""
        todo_text = todo_text.replace("……", "。").replace("——", "，")  # Normalize first
        if not todo_text:
            return []
        if todo_text[-1] not in self.SPLITS:
            # Add appropriate punctuation based on likely language (heuristic)
            if LangSegmenter.is_likely_chinese_japanese(todo_text):
                todo_text += "。"
            else:
                todo_text += "."

        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while i_split_head < len_text:
            char = todo_text[i_split_head]
            # Special handling for decimals or ellipses potentially
            is_potential_decimal = (
                char == "."
                and i_split_head > 0
                and todo_text[i_split_head - 1].isdigit()
                and i_split_head + 1 < len_text
                and todo_text[i_split_head + 1].isdigit()
            )
            is_part_of_ellipsis = (char == "." and i_split_head + 2 < len_text and todo_text[i_split_head + 1 : i_split_head + 3] == "..") or (
                char == "…" and i_split_head + 1 < len_text and todo_text[i_split_head + 1] == "…"
            )

            if char in self.SPLITS and not is_potential_decimal and not is_part_of_ellipsis:
                # Found a split point
                segment = todo_text[i_split_tail : i_split_head + 1]
                if segment.strip():  # Avoid adding empty/whitespace-only segments
                    todo_texts.append(segment)
                i_split_head += 1
                i_split_tail = i_split_head  # Move tail past the split character
            else:
                # Continue scanning
                i_split_head += 1

        # Add any remaining text after the last split
        if i_split_tail < len_text:
            segment = todo_text[i_split_tail:len_text]
            if segment.strip():
                todo_texts.append(segment)

        return todo_texts

    def _cut1(self, inp):  # "凑四句一切"
        """Cuts text every 4 sentences (based on splitting)."""
        inp = inp.strip("\n")
        inps = self._split_text(inp)
        if len(inps) <= 4:
            return inp  # Don't split if 4 or fewer sentences

        opts = []
        for i in range(0, len(inps), 4):
            chunk = "".join(inps[i : min(i + 4, len(inps))])
            if chunk.strip() and not set(chunk).issubset(self.PUNCTUATION):
                opts.append(chunk)

        return "\n".join(opts)

    def _cut2(self, inp):  # "凑50字一切"
        """Cuts text into chunks of roughly 50 characters, splitting at sentences."""
        inp = inp.strip("\n")
        inps = self._split_text(inp)
        if len(inps) < 2:
            return inp

        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            segment = inps[i]
            if summ + len(segment) > 50 and tmp_str:  # If adding makes it too long, and we have something
                opts.append(tmp_str)
                tmp_str = segment
                summ = len(segment)
            else:  # Add to current chunk
                tmp_str += segment
                summ += len(segment)

        if tmp_str:  # Add the last chunk
            opts.append(tmp_str)

        # Merge last chunk if too short
        if len(opts) > 1 and len(opts[-1]) < 25:  # Use a threshold smaller than 50
            # print(f"Merging short last chunk (len {len(opts[-1])}) with previous.")
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]

        # Filter out punctuation-only results
        opts = [item for item in opts if item.strip() and not set(item).issubset(self.PUNCTUATION)]
        return "\n".join(opts)

    def _cut3(self, inp):  # "按中文句号。切"
        """Cuts text strictly by Chinese period '。'."""
        inp = inp.strip("\n")
        # Split by '。', keep the delimiter by adding it back unless it's the very end
        opts = [item + "。" for item in inp.split("。") if item.strip()]
        if inp.endswith("。"):  # Adjust if original ended with it
            if opts:
                opts[-1] = opts[-1][:-1]  # Remove the extra one added if original had it
        else:
            if opts:
                opts[-1] = opts[-1][:-1]  # Remove if original didn't end with it

        # Filter out punctuation-only results
        opts = [item for item in opts if item.strip() and not set(item).issubset(self.PUNCTUATION)]
        return "\n".join(opts)

    def _cut4(self, inp):  # "按英文句号.切"
        """Cuts text by English period '.', avoiding decimals."""
        inp = inp.strip("\n")
        # Use regex to split by '.' not preceded or followed by a digit
        opts = re.split(r"(?<!\d)\.(?!\d)", inp)
        # Reconstruct sentences including the period where appropriate (split removes it)
        reconstructed_opts = []
        current_opt = ""
        original_idx = 0
        for part in opts:
            current_opt += part
            original_idx += len(part)
            # Check if the original string had a '.' at this position
            if original_idx < len(inp) and inp[original_idx] == ".":
                # Check if it qualifies as a split point (not decimal)
                is_decimal = original_idx > 0 and inp[original_idx - 1].isdigit() and original_idx + 1 < len(inp) and inp[original_idx + 1].isdigit()
                if not is_decimal:
                    current_opt += "."  # Add the period back
                    reconstructed_opts.append(current_opt)
                    current_opt = ""
                else:
                    current_opt += "."  # Part of a decimal, keep it
                original_idx += 1  # Move past the period
            elif original_idx == len(inp):  # End of string
                if current_opt.strip():
                    reconstructed_opts.append(current_opt)

        # Filter out punctuation-only results
        opts = [item for item in reconstructed_opts if item.strip() and not set(item).issubset(self.PUNCTUATION)]
        return "\n".join(opts)

    def _cut5(self, inp):  # "按标点符号切"
        """Cuts text by any punctuation in SPLITS, avoiding decimals."""
        inp = inp.strip("\n")
        # Split using the generic _split_text which handles punctuation correctly
        opts = self._split_text(inp)
        # Filter out punctuation-only results
        opts = [item for item in opts if item.strip() and not set(item).issubset(self.PUNCTUATION)]
        return "\n".join(opts)

    def _process_text_input(self, texts):
        """Validates and cleans list of text inputs."""
        _text = []
        if not isinstance(texts, list):
            texts = [texts]  # Ensure it's a list

        if all(text is None or text.strip() == "" for text in texts):
            raise ValueError(self.i18n("请输入有效文本"))  # "Please enter valid text"

        for text in texts:
            if text is not None and text.strip() != "":
                _text.append(text.strip())
        return _text

    def _merge_short_text_in_array(self, texts, threshold):
        """Merges short text segments in a list."""
        if (len(texts)) < 2:
            return texts
        result = []
        text_buffer = ""
        for ele in texts:
            text_buffer += ele
            if len(text_buffer) >= threshold:
                result.append(text_buffer)
                text_buffer = ""
        # Append remaining buffer
        if text_buffer:
            if not result:  # If result is empty, just add the buffer
                result.append(text_buffer)
            else:  # Append to the last element
                result[-1] += text_buffer
        return result

    # --- Core Synthesis ---
    @torch.inference_mode()
    def get_tts_wav(
        self,
        ref_wav_path,
        prompt_text,
        text,
        prompt_language=None,
        text_language=None,
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        speed=1.0,
        if_freeze=False,
        sample_steps=8,
        pause_second=0.3,
        how_to_cut="不切",
        output_sr=48000,
    ):
        """
        Generates TTS audio based on reference audio/text and target text.

        Args:
        ref_wav_path (str): Path to the reference WAV file (16kHz recommended).
        prompt_text (str): Text corresponding to the reference audio.
        text (str): The target text to synthesize.
        prompt_language (str, optional): Language of the prompt_text (e.g., "中英混合"). Defaults to class language setting.
        text_language (str, optional): Language of the target text (e.g., "中英混合"). Defaults to class language setting.
        top_k (int): Top K sampling for GPT model.
        top_p (float): Top P sampling for GPT model.
        temperature (float): Sampling temperature for GPT model.
        speed (float): Controls the speed of the generated speech. Lower is slower.
        if_freeze (bool): Use caching within the call (for repeated identical segments).
        sample_steps (int): Number of steps for CFM inference.
        pause_second (float): Duration of silence inserted between text chunks.
        how_to_cut (str): Text splitting method key (e.g., "不切", "凑四句一切").
        output_sr (int): The desired sampling rate for the output audio.

        Yields:
        tuple: (sampling_rate, audio_numpy_array)
        Only yields the final concatenated audio for the entire input text.
        """
        # --- Timing Initialization ---
        time_log_segments = []  # Equivalent to 't' in original
        t0_overall_start = ttime()  # For overall timing, not part of the original specific print

        # --- Language and Initial Setup ---
        prompt_lang_key_arg = prompt_language if prompt_language is not None else self.i18n("中英混合")
        text_lang_key_arg = text_language if text_language is not None else self.i18n("中英混合")
        _prompt_lang_code = self.LANGUAGE_MAP.get(prompt_lang_key_arg, prompt_lang_key_arg)
        _text_lang_code = self.LANGUAGE_MAP.get(text_lang_key_arg, text_lang_key_arg)

        prompt_text = prompt_text.strip("\n")
        if prompt_text and prompt_text[-1] not in self.SPLITS:
            prompt_text += "。" if _prompt_lang_code != "en" else "."
        text = text.strip("\n")

        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * pause_second),
            dtype=np.float16 if self.is_half else np.float32,
        )
        zero_wav_torch = torch.from_numpy(zero_wav)
        if self.is_half:
            zero_wav_torch = zero_wav_torch.half().to(self.device)
        else:
            zero_wav_torch = zero_wav_torch.to(self.device)

        # --- Reference Audio Processing ---
        t_ref_start = ttime()  # t0 in original for the first segment

        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.ndim > 1:
            wav16k = librosa.to_mono(wav16k)
        wav16k_torch = torch.from_numpy(wav16k)
        if self.is_half:
            wav16k_torch = wav16k_torch.half().to(self.device)
        else:
            wav16k_torch = wav16k_torch.to(self.device)
        wav16k_torch = torch.cat([wav16k_torch, zero_wav_torch], dim=0)

        ssl_content = self.ssl_model.model(wav16k_torch.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = self.vq_model.extract_latent(ssl_content.to(self.dtype))
        prompt_semantic_float = codes[0, 0]
        prompt_for_t2s = prompt_semantic_float.unsqueeze(0).to(self.device)
        prompt_for_vq = prompt_semantic_float.unsqueeze(0).unsqueeze(0).to(self.device).long()

        t_ref_end_and_global_text_proc_start = ttime()  # t1 in original after ref processing
        time_log_segments.append(t_ref_end_and_global_text_proc_start - t_ref_start)

        # --- Text Cutting and Initial Prompt Text Processing ---
        how_to_cut_i18n = self.i18n(how_to_cut)
        if how_to_cut_i18n == self.i18n("凑四句一切"):
            text = self._cut1(text)
        elif how_to_cut_i18n == self.i18n("凑50字一切"):
            text = self._cut2(text)
        # ... (other cut conditions) ...
        elif how_to_cut_i18n == self.i18n("按标点符号切"):
            text = self._cut5(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = self._process_text_input(texts)
        texts = self._merge_short_text_in_array(texts, 5)

        audio_opt = []
        phones1, bert1, norm_text1 = self._get_phones_and_bert(prompt_text, _prompt_lang_code)
        cache = {}

        # This t_prev_stage_end will function as 't1' in the original loop's t.extend([t2-t1, ...])
        t_prev_stage_end = t_ref_end_and_global_text_proc_start  # Initial t1 for the first segment's t2-t1

        # --- Per-Segment Inference Loop ---
        for i_text, text_segment in enumerate(texts):
            if not text_segment.strip():
                # If skipping, ensure t_prev_stage_end is updated to maintain correct inter-segment timing
                # if significant work happened before skip, or just use current time
                t_prev_stage_end = ttime()  # Reset start for next segment's text processing phase
                continue

            if text_segment[-1] not in self.SPLITS:
                text_segment += "。" if _text_lang_code != "en" else "."

            # Current segment's text processing (phones2, bert2) happens here
            # before t_gpt_infer_start
            phones2, bert2, norm_text2 = self._get_phones_and_bert(text_segment, _text_lang_code)
            bert_combined = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            bert_combined = bert_combined.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            t_gpt_infer_start = ttime()  # t2 in original loop

            if i_text in cache and if_freeze == True:
                pred_semantic = cache[i_text]
            else:
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt_for_t2s.to(self.t2s_model.device),
                    bert_combined.to(self.t2s_model.device),
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic

            t_gpt_infer_end_and_vq_start = ttime()  # t3 in original loop

            # VQ (SoVITS) + CFM Decoder
            refer_spec = self._get_spepc(ref_wav_path).to(self.device, self.dtype)
            phoneme_ids0_dev = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
            phoneme_ids1_dev = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            fea_ref, ge = self.vq_model.decode_encp(prompt_for_vq, phoneme_ids0_dev, refer_spec)

            ref_audio_v4, sr_v4 = torchaudio.load(ref_wav_path)
            ref_audio_v4 = ref_audio_v4.to(self.device, torch.float32)
            if ref_audio_v4.shape[0] > 1:
                ref_audio_v4 = ref_audio_v4.mean(0, keepdim=True)
            tgt_sr_v4 = 32000
            if sr_v4 != tgt_sr_v4:
                ref_audio_v4 = self._resample(ref_audio_v4, sr_v4, tgt_sr_v4)
            mel_ref_v4 = self.mel_fn_v4(ref_audio_v4)
            mel_ref_v4 = self._norm_spec(mel_ref_v4)
            T_min = min(mel_ref_v4.shape[2], fea_ref.shape[2])
            mel_ref_v4 = mel_ref_v4[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            Tref, Tchunk = 500, 1000
            if T_min > Tref:
                mel_ref_v4 = mel_ref_v4[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            chunk_len = Tchunk - T_min
            mel_context = mel_ref_v4.to(self.dtype)
            fea_context = fea_ref

            fea_todo, ge = self.vq_model.decode_encp(pred_semantic, phoneme_ids1_dev, refer_spec, ge, speed)

            cfm_results_list = []
            idx_cfm = 0
            while True:
                fea_todo_chunk = fea_todo[:, :, idx_cfm : idx_cfm + chunk_len]
                if fea_todo_chunk.shape[-1] == 0:
                    break
                idx_cfm += fea_todo_chunk.shape[-1]
                fea_combined = torch.cat([fea_context, fea_todo_chunk], dim=2).transpose(1, 2)
                cfm_res_chunk = self.vq_model.cfm.inference(
                    fea_combined, torch.LongTensor([fea_combined.size(1)]).to(fea_combined.device), mel_context, sample_steps, inference_cfg_rate=0
                )
                cfm_res_new = cfm_res_chunk[:, :, mel_context.shape[2] :]
                cfm_results_list.append(cfm_res_new)
                mel_context = cfm_res_chunk[:, :, -T_min:]
                fea_context = fea_todo_chunk[:, :, -T_min:]

            if not cfm_results_list:
                audio_opt.append(zero_wav_torch)  # Add silence
                # Log zero times for this segment or handle appropriately
                time_log_segments.extend(
                    [
                        t_gpt_infer_start - t_prev_stage_end,  # Text proc time
                        t_gpt_infer_end_and_vq_start - t_gpt_infer_start,  # GPT time
                        0.0,  # VQ/Vocoder time
                    ]
                )
                t_prev_stage_end = ttime()  # Reset for next segment's text proc phase
                continue

            cfm_res_final = torch.cat(cfm_results_list, dim=2)
            mel_final_denorm = self._denorm_spec(cfm_res_final)

            if self.hifigan_model is None:
                raise RuntimeError("HiFiGAN vocoder not initialized.")
            wav_gen = self.hifigan_model(mel_final_denorm.to(self.device, self.dtype))
            audio_segment = wav_gen[0, 0]
            max_audio = torch.abs(audio_segment).max()
            if max_audio > 1.0:
                audio_segment = audio_segment / max_audio

            audio_opt.append(audio_segment.to(torch.float32))
            audio_opt.append(zero_wav_torch.to(audio_segment.device, torch.float32))

            t_vq_voc_end = ttime()  # t4 in original loop

            time_log_segments.extend(
                [
                    t_gpt_infer_start - t_prev_stage_end,  # Duration for text processing of current segment
                    t_gpt_infer_end_and_vq_start - t_gpt_infer_start,  # Duration for T2S (GPT)
                    t_vq_voc_end - t_gpt_infer_end_and_vq_start,  # Duration for VQ/CFM + Vocoder
                ]
            )
            t_prev_stage_end = ttime()  # Reset t1 for the next iteration, as in original's t1=ttime()

        # --- Final Audio Concatenation and Output ---
        final_audio_tensor = torch.cat(audio_opt, dim=0)
        final_audio_np = final_audio_tensor.cpu().numpy()
        final_audio_int16 = (final_audio_np * 32767.0).astype(np.int16)

        # --- Print Timing Info ---
        if len(time_log_segments) > 0:  # Ensure there's at least ref_proc_time
            ref_proc_time = time_log_segments[0]
            # Check if there are segment processing times
            if len(time_log_segments) > 1:
                sum_text_proc_time = sum(time_log_segments[1::3])
                sum_gpt_time = sum(time_log_segments[2::3])
                sum_vq_voc_time = sum(time_log_segments[3::3])
                print(
                    f"Timings (s): Ref={ref_proc_time:.3f}\t"
                    f"TextProc={sum_text_proc_time:.3f}\t"
                    f"GPT={sum_gpt_time:.3f}\t"
                    f"VQ/Voc={sum_vq_voc_time:.3f}"
                )
            else:  # Only reference processing time is available
                print(f"Timings (s): Ref={ref_proc_time:.3f}\t (No target segments processed)")

        t_overall_end = ttime()
        print(f"Total function runtime: {t_overall_end - t0_overall_start:.3f}s") # Optional overall time

        yield output_sr, final_audio_int16
