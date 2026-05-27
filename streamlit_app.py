import base64
import hashlib
import os
import time
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageOps

# ========================
# CONFIG
# ========================
DEFAULT_API_URL = "https://script.google.com/macros/s/AKfycbxrvLZ6cfBNHC1z_V5iSADK4nPqoVa2RLeafcmbIugcKQtrfjX8adKEUlNtKkM-rbkV/exec"
MAX_POST_BYTES = 45 * 1024 * 1024


def get_api_url() -> str:
    try:
        return st.secrets.get("API_URL") or os.getenv("API_URL") or DEFAULT_API_URL
    except st.errors.StreamlitSecretNotFoundError:
        return os.getenv("API_URL") or DEFAULT_API_URL


def make_upload_bytes(uploaded_file, compress: bool) -> tuple[bytes, str]:
    if not compress:
        return uploaded_file.getvalue(), uploaded_file.type or "image/png"

    image = Image.open(BytesIO(uploaded_file.getvalue()))
    image = ImageOps.exif_transpose(image)
    image.thumbnail((1280, 1280))

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    output = BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue(), "image/jpeg"


def response_to_json(response: requests.Response) -> dict | None:
    try:
        return response.json()
    except ValueError:
        return None


def add_upload_history(item: dict) -> None:
    history = st.session_state.setdefault("upload_history", [])
    history.insert(0, item)
    del history[5:]


def render_upload_history() -> None:
    history = st.session_state.get("upload_history", [])
    if not history:
        return

    st.divider()
    st.subheader("Riwayat upload")

    for item in history:
        with st.container(border=True):
            st.write(f"{item['time']} - {item['status']}")
            st.caption(f"SHA-256: {item['sha256'][:16]}... | Base64: {item['length']:,} karakter")
            if item.get("name"):
                st.write("File:", item["name"])
            if item.get("link"):
                st.markdown(f"[Buka file Drive]({item['link']})")


st.set_page_config(page_title="Base64 Upload to Drive", layout="centered")

st.title("Automate IMAGE TO BASE 64 BY FARHAN")
st.write(" BISMILLAH.")

api_url = get_api_url()
compress_image = st.checkbox("Kompres gambar sebelum encode", value=True)
input_source = st.radio("Pilih sumber gambar", ["Ambil foto", "Upload file"], horizontal=True)

if input_source == "Ambil foto":
    img = st.camera_input("Ambil gambar")
else:
    img = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

if img:
    st.image(img, caption="Preview", use_container_width=True)

    try:
        image_bytes, mime_type = make_upload_bytes(img, compress_image)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        base64_sha256 = hashlib.sha256(base64_str.encode("utf-8")).hexdigest()
        local_name = f"base64_{int(time.time())}.txt"

        payload = {
            "base64": base64_str,
            "length": len(base64_str),
            "sha256": base64_sha256,
            "mimeType": mime_type,
        }
        payload_size = len(str(payload).encode("utf-8"))

        st.write("Ukuran gambar:", f"{len(image_bytes):,} bytes")
        st.write("Panjang Base64:", f"{len(base64_str):,} karakter")
        st.write("Perkiraan ukuran request:", f"{payload_size:,} bytes")
        st.write("SHA-256 Base64:")
        st.code(base64_sha256)

        st.caption("Preview di bawah memang dipendekkan. Data lengkap ada di tombol download dan payload upload.")
        st.code(f"{base64_str[:120]} ... {base64_str[-120:]}")

        st.download_button(
            "Download Base64 lengkap",
            data=base64_str,
            file_name=local_name,
            mime="text/plain",
        )

        data_uri = f"data:{mime_type};base64,{base64_str}"
        with st.expander("Preview Data URI"):
            st.caption("Ini berguna untuk tes cepat apakah Base64 bisa langsung dibaca sebagai gambar.")
            st.code(f"{data_uri[:160]} ... {data_uri[-120:]}")
            st.download_button(
                "Download Data URI HTML",
                data=f'<img src="{data_uri}" style="max-width:100%;height:auto;">',
                file_name=f"preview_{int(time.time())}.html",
                mime="text/html",
            )

        if st.button("Simpan backup TXT lokal"):
            with open(local_name, "w", encoding="utf-8") as f:
                f.write(base64_str)
            st.success(f"Backup tersimpan: {local_name}")

        if payload_size > MAX_POST_BYTES:
            st.error("Payload terlalu besar untuk dikirim aman. Aktifkan kompresi atau ambil foto ulang.")
        elif st.session_state.get("last_uploaded_sha256") == base64_sha256:
            st.info("Foto ini sudah otomatis dikirim ke Google Drive.")
        else:
            with st.spinner("Mengirim ke Google Drive..."):
                response = requests.post(api_url, json=payload, timeout=60)

            if response.status_code != 200:
                st.error(f"HTTP Error: {response.status_code}")
                st.text(response.text)
                add_upload_history({
                    "time": time.strftime("%H:%M:%S"),
                    "status": f"HTTP Error {response.status_code}",
                    "sha256": base64_sha256,
                    "length": len(base64_str),
                })
            else:
                res = response_to_json(response)
                if res is None:
                    st.error("Server tidak mengembalikan JSON yang valid.")
                    st.text(response.text)
                    add_upload_history({
                        "time": time.strftime("%H:%M:%S"),
                        "status": "Response bukan JSON",
                        "sha256": base64_sha256,
                        "length": len(base64_str),
                    })
                elif res.get("status") == "success":
                    st.session_state["last_uploaded_sha256"] = base64_sha256
                    st.success("Berhasil upload ke Google Drive!")

                    if res.get("length") and int(res["length"]) != len(base64_str):
                        st.warning("Panjang Base64 di server berbeda. Cek kode Apps Script.")
                    if res.get("sha256") and res["sha256"] != base64_sha256:
                        st.warning("SHA-256 di server berbeda. Data kemungkinan berubah saat diterima.")

                    st.write("Nama file:")
                    st.code(res.get("name", "-"))

                    link = res.get("link") or res.get("url")
                    add_upload_history({
                        "time": time.strftime("%H:%M:%S"),
                        "status": "Berhasil",
                        "sha256": base64_sha256,
                        "length": len(base64_str),
                        "name": res.get("name"),
                        "link": link,
                    })
                    if link:
                        st.write("Link file:")
                        st.markdown(f"[Buka di Drive]({link})")
                    else:
                        st.info("Upload berhasil, tapi server tidak mengirim link file.")
                else:
                    st.error("Gagal dari server")
                    st.write(res)
                    add_upload_history({
                        "time": time.strftime("%H:%M:%S"),
                        "status": "Gagal dari server",
                        "sha256": base64_sha256,
                        "length": len(base64_str),
                    })

        render_upload_history()

    except Exception as e:
        st.error(f"Error: {str(e)}")
