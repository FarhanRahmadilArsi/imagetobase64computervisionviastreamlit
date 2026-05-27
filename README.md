# VLM Computer Vision Demo

Contoh project computer vision sederhana menggunakan Vision-Language Model (VLM) dengan Python.

Fitur:
- `caption`: membuat deskripsi singkat dan deskripsi yang lebih detail
- `vqa`: menjawab pertanyaan tentang gambar
- `both`: menjalankan keduanya sekaligus
- analisis otomatis tambahan: objek utama, warna dominan, jumlah objek, dan latar
- output bisa berupa narasi teks atau JSON
- pertanyaan bahasa Indonesia akan dipetakan otomatis ke bentuk yang lebih cocok untuk model
- model di-cache agar analisis berikutnya lebih cepat
- reasoning layer menyusun observasi, sinyal utama, interpretasi, dan kesimpulan
- versi Streamlit mendukung upload gambar dan snapshot dari kamera browser
- tersedia beberapa profil analisis seperti `general`, `surveillance`, `document`, dan `ui`
- Streamlit sekarang menampilkan reasoning, metadata, dan riwayat analisis singkat
- hasil reasoning sekarang punya `tags` dan `follow_up_questions`
- Streamlit mendukung unduh ringkasan teks dan laporan JSON lengkap

Model default:
- Unified VLM: `Qwen/Qwen2.5-VL-3B-Instruct`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Catatan:
- Saat run pertama, model akan diunduh dari Hugging Face.
- `torch` dapat memanfaatkan GPU bila environment Anda sudah mendukung CUDA.
- Karena model sekarang generatif, nilai `score` pada jawaban VQA bersifat heuristik, bukan probabilitas classifier.
- Model ini jauh lebih kuat daripada setup sebelumnya, tetapi juga jauh lebih berat. GPU sangat disarankan.

## Menjalankan

Versi browser dengan Streamlit:

```powershell
streamlit run .\streamlit_app.py
```

Lalu buka alamat lokal yang ditampilkan Streamlit di browser, biasanya:

```text
http://localhost:8501
```

Di halaman Streamlit, Anda bisa memilih:
- `Upload file` untuk mengunggah gambar dari komputer
- `Kamera` untuk mengambil snapshot langsung dari webcam browser

Catatan:
- mode kamera mengambil foto per snapshot, bukan analisis video real-time
- browser akan meminta izin akses kamera saat fitur ini dipakai

Captioning:

```powershell
python app.py --image .\contoh.jpg --task caption
```

Visual question answering:

```powershell
python app.py --image .\contoh.jpg --task vqa --question "Ada berapa orang di gambar ini?"
```

Keduanya:

```powershell
python app.py --image .\contoh.jpg --task both --question "What is happening in this image?" --profile general --model-tier auto
```

Kalau ingin JSON mentah:

```powershell
python app.py --image .\contoh.jpg --task both --question "What is happening in this image?" --profile surveillance --model-tier strong --format json
```

Pilihan profil:
- `general`: analisis umum untuk foto sehari-hari
- `surveillance`: fokus pada subjek, aktivitas, dan lingkungan
- `document`: fokus pada dokumen atau layar
- `ui`: fokus pada antarmuka dan fungsi tampilan

Pilihan tier model:
- `auto`: memakai model kuat jika GPU tersedia, kalau tidak fallback ke model ringan
- `strong`: memaksa `Qwen/Qwen2.5-VL-3B-Instruct`
- `light`: memaksa fallback `HuggingFaceTB/SmolVLM-500M-Instruct`

## Output

Default output sekarang berbentuk narasi teks berbahasa Indonesia, misalnya:

```text
Hasil analisis untuk gambar `contoh.jpg`.

Seseorang sedang mengendarai sepeda di jalan perkotaan dengan bangunan di latar belakang. Warna yang paling terlihat didominasi oleh biru dan abu-abu. Latar gambar terlihat seperti jalan di area perkotaan.

Untuk pertanyaan "Apa yang sedang terjadi pada gambar ini?", sistem memperkirakan jawabannya adalah "sedang mengendarai sepeda" dengan tingkat keyakinan 0.98 sehingga hasil ini tergolong sangat yakin. Pertanyaan tersebut dipetakan ke bentuk yang lebih sesuai untuk model, yaitu "What is happening in this image?". Selain itu, objek utama pada gambar diperkirakan sebagai sepeda dengan tingkat keyakinan 0.97 (sangat yakin), warna yang paling dominan diperkirakan sebagai biru dan abu-abu dengan tingkat keyakinan 0.88 (cukup yakin), kondisi latar atau lingkungan gambar diperkirakan sebagai jalan di area perkotaan dengan tingkat keyakinan 0.82 (cukup yakin).
```

Jika memakai `--format json`, output berbentuk JSON seperti ini:

```json
{
  "image": "contoh.jpg",
  "task": "both",
      "caption": {
        "short": "a person riding a bicycle on a street",
        "detailed": "a person is riding a bicycle on a city street with buildings visible in the background"
  },
  "analysis": {
    "requested": {
      "question": "What is happening in this image?",
      "result": {
        "score": 0.82,
        "score_source": "heuristic",
        "answer": "riding a bike"
      }
    },
    "auto_analysis": [
      {
        "question": "What is the main object in this image?",
        "result": {
          "score": 0.76,
          "score_source": "heuristic",
          "answer": "bicycle"
        }
      }
    ],
    "reasoning": {
      "observations": [
        "Deskripsi singkat model menunjukkan seorang pria sedang mengendarai sepeda di jalan."
      ],
      "structured_signals": {
        "main_object": {
          "label": "objek utama",
          "answer": "sepeda",
          "score": 0.97,
          "confidence": "sangat yakin"
        }
      },
      "tags": [
        "sepeda",
        "jalan di area perkotaan"
      ],
      "follow_up_questions": [
        "Apa objek yang paling penting pada gambar ini?",
        "Apa konteks utama dari gambar ini?"
      ],
      "interpretation": "Seseorang sedang mengendarai sepeda di jalan perkotaan dengan bangunan di latar belakang. Fokus visual utamanya tampak berada pada sepeda.",
      "conclusion": "Secara keseluruhan, sistem menyimpulkan bahwa jawaban paling masuk akal untuk pertanyaan utama adalah sedang mengendarai sepeda."
    }
  },
  "meta": {
    "image_name": "contoh.jpg",
    "profile": "general",
    "duration_seconds": 4.21,
    "models": {
      "vlm": "Qwen/Qwen2.5-VL-3B-Instruct"
    }
  }
}
```

## Pengembangan Lanjutan

Beberapa pengembangan yang masuk akal:
- ganti model ke VLM yang lebih kuat
- tambah mode batch untuk banyak gambar
- buat REST API dengan FastAPI
- tambah deteksi domain tertentu, misalnya helm, kendaraan, atau dokumen

# imagetobase64computervisionviastreamlit
