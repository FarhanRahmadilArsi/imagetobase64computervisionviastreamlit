import argparse
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LIGHTWEIGHT_VLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
QUESTION_LABELS = {
    "What is the main object in this image?": "Objek utama pada gambar",
    "What colors are most visible in this image?": "Warna yang paling dominan",
    "How many prominent objects are visible?": "Perkiraan jumlah objek yang menonjol",
    "What is the setting or background like?": "Kondisi latar atau lingkungan gambar",
    "What is happening in this image?": "Aktivitas atau kejadian utama",
    "Is this scene indoors or outdoors?": "Tipe lokasi",
    "Does this image look like a document, interface, or natural scene?": "Jenis tampilan gambar",
    "Is there a person visible in this image?": "Ada orang atau tidak",
    "What kind of document or screen is shown?": "Jenis dokumen atau layar",
    "What is the main purpose of this screen?": "Tujuan utama tampilan",
    "Is this image more like a street, office, or home setting?": "Kategori lingkungan",
}
PHRASE_TRANSLATIONS = {
    "a red circle on a blue background": "sebuah lingkaran merah di atas latar biru",
    "a person riding a bicycle on a city street with buildings in the background": "seseorang sedang mengendarai sepeda di jalan perkotaan dengan bangunan di latar belakang",
    "a man riding a bicycle on a street": "seorang pria sedang mengendarai sepeda di jalan",
    "bicycle": "sepeda",
    "circle": "lingkaran",
    "red and blue": "merah dan biru",
    "blue and gray": "biru dan abu-abu",
    "a city street": "jalan di area perkotaan",
    "street": "jalan",
    "no": "tidak",
    "yes": "ya",
    "riding a bike": "sedang mengendarai sepeda",
    "one": "satu",
    "two": "dua",
    "three": "tiga",
}
QUESTION_TRANSLATIONS = {
    "apa objek utama pada gambar ini?": "What is the main object in this image?",
    "apa objek utama pada gambar?": "What is the main object in this image?",
    "objek utama apa yang terlihat?": "What is the main object in this image?",
    "warna apa yang dominan pada gambar ini?": "What colors are most visible in this image?",
    "warna dominan apa yang terlihat?": "What colors are most visible in this image?",
    "ada berapa objek utama pada gambar ini?": "How many prominent objects are visible?",
    "berapa jumlah objek yang menonjol?": "How many prominent objects are visible?",
    "latar gambar ini seperti apa?": "What is the setting or background like?",
    "latar atau lingkungan gambar ini seperti apa?": "What is the setting or background like?",
    "apa yang sedang terjadi pada gambar ini?": "What is happening in this image?",
    "apa yang terlihat pada gambar ini?": "What is shown in this image?",
}
COMMON_ENGLISH_TO_INDONESIAN = {
    "what is happening in this image?": "Apa yang sedang terjadi pada gambar ini?",
    "what is shown in this image?": "Apa yang terlihat pada gambar ini?",
    "what is the main object in this image?": "Apa objek utama pada gambar ini?",
    "what colors are most visible in this image?": "Warna apa yang paling dominan pada gambar ini?",
    "how many prominent objects are visible?": "Ada berapa objek yang menonjol pada gambar ini?",
    "what is the setting or background like?": "Latar atau lingkungan gambar ini seperti apa?",
}
REASONING_LABELS = {
    "main_object": "objek utama",
    "dominant_colors": "warna dominan",
    "prominent_count": "jumlah objek menonjol",
    "background": "latar atau lingkungan",
    "activity": "aktivitas utama",
    "location_type": "jenis lokasi",
    "image_type": "jenis tampilan gambar",
    "person_presence": "keberadaan orang",
    "screen_type": "jenis dokumen atau layar",
    "screen_purpose": "tujuan tampilan",
    "environment_category": "kategori lingkungan",
}
PROFILE_FOLLOW_UPS = {
    "general": [
        "Apa objek yang paling penting pada gambar ini?",
        "Apa konteks utama dari gambar ini?",
        "Apakah ada detail yang tampak tidak biasa?",
    ],
    "surveillance": [
        "Apakah ada aktivitas yang perlu diperhatikan lebih lanjut?",
        "Apakah jumlah orang atau objek tampak wajar?",
        "Area mana yang paling relevan untuk dipantau ulang?",
    ],
    "document": [
        "Apakah dokumen ini terlihat formal atau informal?",
        "Bagian mana dari dokumen yang paling penting untuk dibaca?",
        "Apakah tampilannya lebih mirip formulir, laporan, atau surat?",
    ],
    "ui": [
        "Apa aksi utama yang bisa dilakukan pengguna pada layar ini?",
        "Bagian mana dari UI yang paling dominan?",
        "Apakah tampilan ini lebih cocok disebut dashboard, form, atau halaman detail?",
    ],
}
ANALYSIS_PROFILES = {
    "general": {
        "label": "Umum",
        "description": "Cocok untuk foto sehari-hari dan analisis visual umum.",
        "questions": [
            "What is the main object in this image?",
            "What colors are most visible in this image?",
            "How many prominent objects are visible?",
            "What is the setting or background like?",
            "What is happening in this image?",
            "Is this scene indoors or outdoors?",
        ],
    },
    "surveillance": {
        "label": "Pemantauan",
        "description": "Fokus pada subjek utama, aktivitas, dan konteks lingkungan.",
        "questions": [
            "What is the main object in this image?",
            "Is there a person visible in this image?",
            "What is happening in this image?",
            "How many prominent objects are visible?",
            "Is this image more like a street, office, or home setting?",
            "Is this scene indoors or outdoors?",
        ],
    },
    "document": {
        "label": "Dokumen",
        "description": "Fokus pada apakah gambar menyerupai dokumen atau layar.",
        "questions": [
            "Does this image look like a document, interface, or natural scene?",
            "What kind of document or screen is shown?",
            "What is the main purpose of this screen?",
            "What colors are most visible in this image?",
            "How many prominent objects are visible?",
        ],
    },
    "ui": {
        "label": "UI / Screen",
        "description": "Fokus pada layout, fungsi, dan jenis tampilan antarmuka.",
        "questions": [
            "Does this image look like a document, interface, or natural scene?",
            "What kind of document or screen is shown?",
            "What is the main purpose of this screen?",
            "What colors are most visible in this image?",
            "Is there a person visible in this image?",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Computer vision sederhana berbasis VLM untuk captioning dan tanya-jawab gambar."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path ke file gambar yang akan dianalisis.",
    )
    parser.add_argument(
        "--task",
        choices=("caption", "vqa", "both"),
        default="both",
        help="Jenis analisis yang dijalankan.",
    )
    parser.add_argument(
        "--question",
        default="Apa objek utama pada gambar ini?",
        help="Pertanyaan untuk mode VQA.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Format output akhir.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(ANALYSIS_PROFILES.keys()),
        default="general",
        help="Profil analisis untuk menyesuaikan pertanyaan otomatis dan reasoning.",
    )
    parser.add_argument(
        "--model-tier",
        choices=("auto", "strong", "light"),
        default="auto",
        help="Pilih tier model. `strong` untuk kualitas terbaik, `light` untuk fallback lebih ringan, `auto` memilih berdasarkan device.",
    )
    return parser.parse_args()


def validate_image(image_path: Path) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Path bukan file: {image_path}")
    return image_path


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


@lru_cache(maxsize=1)
def get_runtime_device() -> dict:
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "label": torch.cuda.get_device_name(0),
            "is_gpu": True,
        }
    return {
        "device": "cpu",
        "label": "CPU",
        "is_gpu": False,
    }


def resolve_model_name(model_tier: str) -> str:
    runtime = get_runtime_device()
    if model_tier == "strong":
        return VLM_MODEL
    if model_tier == "light":
        return LIGHTWEIGHT_VLM_MODEL
    return VLM_MODEL if runtime["is_gpu"] else LIGHTWEIGHT_VLM_MODEL


def should_fallback_to_light(error: Exception, model_name: str) -> bool:
    if model_name != VLM_MODEL:
        return False
    message = str(error).lower()
    fallback_markers = [
        "torchvision",
        "qwen2vlvideoprocessor",
        "video processor",
        "can't load processor",
        "cannot send a request",
        "timed out",
        "connection",
        "refused",
        "failed to establish",
    ]
    return any(marker in message for marker in fallback_markers)


def get_hf_cache_path(model_name: str) -> Path:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return cache_root / f"models--{model_name.replace('/', '--')}"


def cached_model_exists(model_name: str) -> bool:
    model_cache = get_hf_cache_path(model_name)
    return model_cache.exists() and any(model_cache.rglob("*.json"))


@lru_cache(maxsize=4)
def load_vlm_model(model_name: str) -> tuple[AutoProcessor, object]:
    processor_kwargs = {
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    }
    use_local_only = cached_model_exists(model_name)

    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=use_local_only,
            **processor_kwargs,
        )
    except OSError:
        if not cached_model_exists(model_name):
            raise
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, **processor_kwargs)

    runtime = get_runtime_device()
    if model_name == VLM_MODEL:
        model_kwargs = {
            "device_map": "auto",
            "attn_implementation": "sdpa",
        }
        model_kwargs["torch_dtype"] = torch.float16 if runtime["is_gpu"] else torch.float32
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=use_local_only,
                **model_kwargs,
            )
        except OSError:
            if not cached_model_exists(model_name):
                raise
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=True,
                **model_kwargs,
            )
    else:
        from transformers import AutoModelForImageTextToText

        model_kwargs = {}
        if runtime["is_gpu"]:
            model_kwargs["torch_dtype"] = torch.float16
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                local_files_only=use_local_only,
                **model_kwargs,
            )
        except OSError:
            if not cached_model_exists(model_name):
                raise
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                local_files_only=True,
                **model_kwargs,
            )
        if runtime["is_gpu"]:
            model = model.to("cuda")

    return processor, model


def load_vlm_model_with_fallback(model_name: str) -> tuple[AutoProcessor, object, str, list[str]]:
    warnings = []
    try:
        processor, model = load_vlm_model(model_name)
        return processor, model, model_name, warnings
    except Exception as error:
        if should_fallback_to_light(error, model_name):
            warnings.append(
                "Model strong tidak bisa dipakai di runtime ini. Sistem otomatis memakai model ringan."
            )
            processor, model = load_vlm_model(LIGHTWEIGHT_VLM_MODEL)
            return processor, model, LIGHTWEIGHT_VLM_MODEL, warnings
        raise


def generate_caption(
    image: Image.Image,
    processor: AutoProcessor,
    model: object,
    prompt: str,
    max_new_tokens: int = 120,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generated_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return generated_text.strip()


def estimate_answer_confidence(answer: str) -> float:
    normalized = normalize_text(answer)
    if not normalized:
        return 0.3
    if len(normalized.split()) >= 8:
        return 0.88
    if len(normalized.split()) >= 4:
        return 0.82
    if normalized in {"ya", "tidak", "yes", "no"}:
        return 0.72
    return 0.76

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def translate_question_to_model_language(question: str) -> str:
    normalized = normalize_text(question)
    direct_match = QUESTION_TRANSLATIONS.get(normalized)
    if direct_match:
        return direct_match

    if any(keyword in normalized for keyword in ["warna", "color", "colour"]):
        return "What colors are most visible in this image?"
    if any(keyword in normalized for keyword in ["berapa", "jumlah", "how many", "count"]):
        return "How many prominent objects are visible?"
    if any(keyword in normalized for keyword in ["latar", "background", "setting", "lingkungan"]):
        return "What is the setting or background like?"
    if any(keyword in normalized for keyword in ["objek", "object", "benda"]):
        return "What is the main object in this image?"
    if any(keyword in normalized for keyword in ["terjadi", "happening", "aktivitas", "activity"]):
        return "What is happening in this image?"
    if any(keyword in normalized for keyword in ["dokumen", "document", "interface", "ui", "screen", "layar"]):
        return "Does this image look like a document, interface, or natural scene?"
    if any(keyword in normalized for keyword in ["indoor", "outdoor", "ruangan", "luar", "dalam"]):
        return "Is this scene indoors or outdoors?"

    return question


def confidence_label(score: float) -> str:
    if score >= 0.9:
        return "sangat yakin"
    if score >= 0.7:
        return "cukup yakin"
    if score >= 0.5:
        return "masih tentatif"
    return "kurang yakin"


def run_vqa(
    image: Image.Image,
    question: str,
    processor: AutoProcessor,
    model: object,
    max_new_tokens: int = 48,
) -> dict:
    answer = generate_caption(
        image,
        processor,
        model,
        f"Answer the question about this image as precisely as possible: {question}",
        max_new_tokens=max_new_tokens,
    )
    score = estimate_answer_confidence(answer)

    return {
        "score": score,
        "score_source": "heuristic",
        "answer": answer,
    }


def run_caption_with_model(image: Image.Image, model_name: str, quick_mode: bool = False) -> dict:
    processor, model, resolved_model_name, warnings = load_vlm_model_with_fallback(model_name)

    short_caption = generate_caption(
        image,
        processor,
        model,
        "Describe this image briefly in one sentence.",
        max_new_tokens=40 if quick_mode else 60,
    )
    detailed_caption = ""
    if not quick_mode:
        detailed_caption = generate_caption(
            image,
            processor,
            model,
            "Describe this image in detail, including the main object, colors, setting, and any important activity.",
            max_new_tokens=100,
        )

    return {
        "short": short_caption,
        "detailed": detailed_caption,
        "_resolved_model": resolved_model_name,
        "_warnings": warnings,
    }


def run_detailed_vqa(
    image: Image.Image,
    question: str,
    profile: str,
    model_name: str,
    quick_mode: bool = False,
) -> dict:
    processor, model, resolved_model_name, warnings = load_vlm_model_with_fallback(model_name)
    normalized_question = translate_question_to_model_language(question)
    requested_answer = run_vqa(
        image,
        normalized_question,
        processor,
        model,
        max_new_tokens=32 if quick_mode else 48,
    )

    auto_analysis = []
    auto_questions = get_profile_questions(profile)
    if quick_mode:
        auto_questions = auto_questions[:2]

    for auto_question in auto_questions:
        auto_analysis.append(
            {
                "question": auto_question,
                "result": run_vqa(
                    image,
                    auto_question,
                    processor,
                    model,
                    max_new_tokens=24 if quick_mode else 48,
                ),
            }
        )

    return {
        "requested": {
            "question": question,
            "normalized_question": normalized_question,
            "result": requested_answer,
        },
        "auto_analysis": auto_analysis,
        "profile": profile,
        "_resolved_model": resolved_model_name,
        "_warnings": warnings,
    }


def sentence_case(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def humanize_phrase(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""

    translated = PHRASE_TRANSLATIONS.get(cleaned.lower(), cleaned)
    return translated


def humanize_question(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    return COMMON_ENGLISH_TO_INDONESIAN.get(cleaned.lower(), cleaned)


def get_profile_questions(profile: str) -> list[str]:
    return ANALYSIS_PROFILES.get(profile, ANALYSIS_PROFILES["general"])["questions"]


def get_signal_key(question: str) -> str:
    mapping = {
        "What is the main object in this image?": "main_object",
        "What colors are most visible in this image?": "dominant_colors",
        "How many prominent objects are visible?": "prominent_count",
        "What is the setting or background like?": "background",
        "What is happening in this image?": "activity",
        "Is this scene indoors or outdoors?": "location_type",
        "Does this image look like a document, interface, or natural scene?": "image_type",
        "Is there a person visible in this image?": "person_presence",
        "What kind of document or screen is shown?": "screen_type",
        "What is the main purpose of this screen?": "screen_purpose",
        "Is this image more like a street, office, or home setting?": "environment_category",
    }
    return mapping.get(question, question)


def find_auto_answer(auto_items: list[dict], question: str) -> str | None:
    for item in auto_items:
        if item["question"] == question:
            answer = item["result"]["answer"].strip()
            return answer or None
    return None


def find_auto_result(auto_items: list[dict], question: str) -> dict | None:
    for item in auto_items:
        if item["question"] == question:
            return item["result"]
    return None


def build_reasoning(caption: dict | None, requested: dict, auto_items: list[dict]) -> dict:
    observations = []
    evidence = {}
    issues = []

    if caption:
        short_caption = humanize_phrase(caption.get("short", ""))
        detailed_caption = humanize_phrase(caption.get("detailed", ""))
        if short_caption:
            observations.append(f"Deskripsi singkat model menunjukkan {short_caption}.")
            evidence["short_caption"] = short_caption
        if detailed_caption:
            observations.append(f"Deskripsi detail model menunjukkan {detailed_caption}.")
            evidence["detailed_caption"] = detailed_caption

    structured = {}
    for item in auto_items:
        question = item["question"]
        result = item["result"]
        key = get_signal_key(question)
        label = REASONING_LABELS.get(key, QUESTION_LABELS.get(question, question))
        answer = humanize_phrase(result["answer"])
        structured[key] = {
            "label": label,
            "question": question,
            "answer": answer,
            "score": result["score"],
            "confidence": confidence_label(result["score"]),
        }
        observations.append(
            f"Analisis {label} mengarah ke {answer} "
            f"dengan keyakinan {result['score']}."
        )

    if "person_presence" in structured and structured["person_presence"]["answer"] == "tidak":
        detailed_caption = evidence.get("detailed_caption", "")
        if any(word in detailed_caption.lower() for word in ["orang", "pria", "wanita", "seseorang"]):
            issues.append(
                "Caption mengindikasikan ada orang, tetapi sinyal keberadaan orang dari VQA menjawab tidak."
            )

    if "image_type" in structured and "screen_type" in structured:
        image_type = structured["image_type"]["answer"]
        screen_type = structured["screen_type"]["answer"]
        if image_type == "natural scene" and screen_type not in {"", "natural scene"}:
            issues.append(
                "Jenis tampilan umum dan jenis layar/dokumen tidak sepenuhnya konsisten."
            )

    main_object = structured.get("main_object", {}).get("answer")
    colors = structured.get("dominant_colors", {}).get("answer")
    background = structured.get("background", {}).get("answer")
    count = structured.get("prominent_count", {}).get("answer")
    activity = structured.get("activity", {}).get("answer")
    location_type = structured.get("location_type", {}).get("answer")
    image_type = structured.get("image_type", {}).get("answer")
    environment_category = structured.get("environment_category", {}).get("answer")

    requested_answer = humanize_phrase(requested["result"]["answer"])
    observations.append(
        f"Jawaban atas pertanyaan utama mengarah ke {requested_answer} "
        f"dengan keyakinan {requested['result']['score']}."
    )

    interpretation_parts = []
    if caption and caption.get("detailed"):
        interpretation_parts.append(sentence_case(humanize_phrase(caption["detailed"])))

    if main_object:
        interpretation_parts.append(f"Fokus visual utamanya tampak berada pada {main_object}.")
    if activity and activity != requested_answer:
        interpretation_parts.append(f"Aktivitas yang terdeteksi paling dekat dengan {activity}.")
    if colors:
        interpretation_parts.append(f"Warna yang paling menonjol terlihat berupa {colors}.")
    if background:
        interpretation_parts.append(f"Konteks lingkungannya paling dekat dengan {background}.")
    if location_type:
        interpretation_parts.append(f"Adegan ini cenderung berada di area {location_type}.")
    if environment_category:
        interpretation_parts.append(f"Kategori lingkungannya paling mirip {environment_category}.")
    if count:
        interpretation_parts.append(f"Jumlah elemen menonjol diperkirakan sekitar {count}.")
    if image_type:
        interpretation_parts.append(f"Jenis tampilan ini paling dekat dengan {image_type}.")

    conclusion = (
        f"Secara keseluruhan, sistem menyimpulkan bahwa jawaban paling masuk akal "
        f"untuk pertanyaan utama adalah {requested_answer}."
    )
    if issues:
        conclusion += " Namun ada beberapa sinyal yang tidak sepenuhnya konsisten."
    elif requested["result"]["score"] < 0.5:
        conclusion += " Namun keyakinannya rendah sehingga hasil ini sebaiknya dianggap indikatif saja."
    elif requested["result"]["score"] < 0.8:
        conclusion += " Keyakinannya menengah, jadi masih ada ruang untuk interpretasi lain."

    summary_grade = "kuat"
    if issues or requested["result"]["score"] < 0.7:
        summary_grade = "menengah"
    if requested["result"]["score"] < 0.5:
        summary_grade = "lemah"

    tags = []
    for signal in structured.values():
        answer = signal["answer"]
        if answer and answer not in tags and answer not in {"ya", "tidak"}:
            tags.append(answer)

    requested_question = normalize_text(requested.get("question", ""))
    profile = requested.get("profile", "general")
    follow_ups = PROFILE_FOLLOW_UPS.get(profile, PROFILE_FOLLOW_UPS["general"]).copy()
    if "dokumen" in requested_question or "document" in requested_question:
        follow_ups.insert(0, "Bagian mana dari dokumen atau layar yang paling penting untuk diperiksa?")
    if "objek" in requested_question or "object" in requested_question:
        follow_ups.insert(0, "Apa karakteristik visual utama dari objek tersebut?")

    return {
        "observations": observations,
        "evidence": evidence,
        "structured_signals": structured,
        "issues": issues,
        "tags": tags[:8],
        "follow_up_questions": follow_ups[:5],
        "interpretation": " ".join(interpretation_parts).strip(),
        "conclusion": conclusion,
        "assessment": {
            "overall_confidence": requested["result"]["score"],
            "overall_confidence_label": confidence_label(requested["result"]["score"]),
            "consistency": "bermasalah" if issues else "konsisten",
            "strength": summary_grade,
        },
    }


def build_summary_paragraph(caption: dict, auto_items: list[dict]) -> str:
    short_caption_raw = humanize_phrase(caption.get("short", ""))
    detailed_caption_raw = humanize_phrase(caption.get("detailed", ""))
    short_caption = sentence_case(short_caption_raw)
    detailed_caption = sentence_case(detailed_caption_raw)

    main_object = find_auto_answer(auto_items, "What is the main object in this image?")
    colors = find_auto_answer(auto_items, "What colors are most visible in this image?")
    background = find_auto_answer(auto_items, "What is the setting or background like?")

    parts = []
    if detailed_caption:
        parts.append(detailed_caption)
    elif short_caption:
        parts.append(short_caption)

    if main_object:
        main_object_text = humanize_phrase(main_object)
        if main_object_text.lower() not in detailed_caption.lower():
            parts.append(f"Objek yang paling menonjol tampaknya adalah {main_object_text}.")

    if colors:
        parts.append(f"Warna yang paling terlihat didominasi oleh {humanize_phrase(colors)}.")

    if background:
        parts.append(f"Latar gambar terlihat seperti {humanize_phrase(background)}.")

    return " ".join(parts).strip()


def build_detail_paragraph(auto_items: list[dict]) -> str:
    if not auto_items:
        return ""

    fragments = []
    for item in auto_items:
        result = item["result"]
        label = QUESTION_LABELS.get(item["question"], item["question"])
        fragments.append(
            f"{label.lower()} diperkirakan sebagai {humanize_phrase(result['answer'])} "
            f"dengan tingkat keyakinan {result['score']} ({confidence_label(result['score'])})"
        )

    return "Selain itu, " + ", ".join(fragments) + "."


def build_narrative(output: dict) -> str:
    paragraphs = [f"Hasil analisis untuk gambar `{output['image']}`."]

    caption = output.get("caption")
    analysis = output.get("analysis")
    auto_items = analysis.get("auto_analysis", []) if analysis else []
    metadata = output.get("meta", {})

    if caption:
        reasoning = analysis.get("reasoning") if analysis else None
        summary = reasoning.get("interpretation") if reasoning else ""
        if not summary:
            summary = build_summary_paragraph(caption, auto_items)
        if summary:
            paragraphs.append(summary)
        else:
            paragraphs.append("Sistem belum menghasilkan deskripsi ringkas yang cukup baik.")

    if analysis:
        requested = analysis["requested"]
        requested_result = requested["result"]
        display_question = humanize_question(requested["question"])
        normalized_question = requested.get("normalized_question")
        reasoning = analysis.get("reasoning", {})
        requested_sentence = (
            f"Untuk pertanyaan \"{display_question}\", sistem memperkirakan jawabannya adalah "
            f"\"{humanize_phrase(requested_result['answer'])}\" dengan tingkat keyakinan {requested_result['score']} "
            f"sehingga hasil ini tergolong {confidence_label(requested_result['score'])}."
        )
        if normalized_question and normalize_text(normalized_question) != normalize_text(requested["question"]):
            requested_sentence += (
                f" Pertanyaan tersebut dipetakan ke bentuk yang lebih sesuai untuk model, yaitu "
                f"\"{normalized_question}\"."
            )

        if reasoning.get("conclusion"):
            requested_sentence += f" {reasoning['conclusion']}"
        if reasoning.get("issues"):
            requested_sentence += " Catatan penting: " + " ".join(reasoning["issues"])

        if auto_items:
            detail_sentence = build_detail_paragraph(auto_items)
            paragraphs.append(f"{requested_sentence} {detail_sentence}")
        else:
            paragraphs.append(requested_sentence)

    if metadata:
        profile_label = ANALYSIS_PROFILES.get(metadata.get("profile", "general"), ANALYSIS_PROFILES["general"])["label"]
        paragraphs.append(
            f"Analisis ini dijalankan dengan profil {profile_label}, memproses {metadata.get('image_name', output['image'])}, "
            f"dan memerlukan sekitar {metadata.get('duration_seconds', 0)} detik."
        )
        if metadata.get("warnings"):
            paragraphs.append("Catatan runtime: " + " ".join(metadata["warnings"]))

    return "\n\n".join(paragraphs)


def analyze_image(
    image_path: Path,
    task: str,
    question: str,
    profile: str = "general",
    model_tier: str = "auto",
    quick_mode: bool = False,
    progress_callback=None,
) -> dict:
    started_at = time.perf_counter()
    image_path = validate_image(image_path)
    image = load_image(image_path)
    runtime = get_runtime_device()
    model_name = resolve_model_name(model_tier)

    output = {
        "image": str(image_path),
        "task": task,
        "profile": profile,
    }

    if progress_callback:
        progress_callback("Memeriksa runtime dan memuat gambar...", 0.05)

    if task in {"caption", "both"}:
        if progress_callback:
            progress_callback("Membuat caption gambar...", 0.3 if task == "both" else 0.6)
        output["caption"] = run_caption_with_model(image, model_name, quick_mode=quick_mode)

    if task in {"vqa", "both"}:
        if progress_callback:
            progress_callback("Menjalankan visual question answering...", 0.7 if task == "both" else 0.5)
        output["analysis"] = run_detailed_vqa(
            image,
            question,
            profile,
            model_name,
            quick_mode=quick_mode,
        )
        output["analysis"]["requested"]["profile"] = profile
        if progress_callback:
            progress_callback("Menyusun reasoning dan ringkasan...", 0.9)
        output["analysis"]["reasoning"] = build_reasoning(
            output.get("caption"),
            output["analysis"]["requested"],
            output["analysis"]["auto_analysis"],
        )

    output["meta"] = {
        "image_name": image_path.name,
        "profile": profile,
        "model_tier": model_tier,
        "duration_seconds": round(time.perf_counter() - started_at, 2),
        "runtime": runtime,
        "models": {
            "vlm": model_name,
        },
        "warnings": [],
    }
    resolved_models = []
    if output.get("caption", {}).get("_resolved_model"):
        resolved_models.append(output["caption"].pop("_resolved_model"))
    if output.get("analysis", {}).get("_resolved_model"):
        resolved_models.append(output["analysis"].pop("_resolved_model"))
    if output.get("caption", {}).get("_warnings"):
        output["meta"]["warnings"].extend(output["caption"].pop("_warnings"))
    if output.get("analysis", {}).get("_warnings"):
        output["meta"]["warnings"].extend(output["analysis"].pop("_warnings"))
    if resolved_models:
        output["meta"]["models"]["vlm"] = resolved_models[-1]
    if not runtime["is_gpu"] and model_name == VLM_MODEL:
        output["meta"]["warnings"].append(
            "Model strong berjalan di CPU. Inference bisa sangat lambat atau gagal karena resource tidak cukup."
        )
    if not runtime["is_gpu"] and model_name == LIGHTWEIGHT_VLM_MODEL:
        output["meta"]["warnings"].append(
            "GPU tidak terdeteksi. Sistem otomatis memakai model yang lebih ringan."
        )
    if quick_mode:
        output["meta"]["warnings"].append(
            "Mode cepat aktif. Caption detail dan sebagian pertanyaan otomatis dikurangi agar analisis CPU lebih cepat."
        )

    if progress_callback:
        progress_callback("Analisis selesai.", 1.0)

    return output


def main() -> None:
    args = parse_args()
    output = analyze_image(Path(args.image), args.task, args.question, args.profile, args.model_tier)

    if args.format == "json":
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(build_narrative(output))


if __name__ == "__main__":
    main()

# ========================
# HELPER UNTUK API
# ========================
def analyze_image_from_base64(base64_str: str):
    import base64
    import time
    from pathlib import Path

    filename = f"temp_{int(time.time())}.jpg"

    # decode base64 → file
    image_bytes = base64.b64decode(base64_str)

    with open(filename, "wb") as f:
        f.write(image_bytes)

    # pakai function utama kamu
    result = analyze_image(
        Path(filename),
        task="caption",  # bisa diganti "both"
        question="What is happening in this image?"
    )

    return result