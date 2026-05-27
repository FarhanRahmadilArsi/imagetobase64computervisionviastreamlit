from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="GAS Bridge API")

# URL yang kamu salin dari Deployment Google Apps Script
GAS_URL = "https://script.google.com/macros/s/AKfycbzXgPX1l1_H5pv6QEsLIuO_e4EuJJn9oFJIVSR3jIrrhchKAPNvMf_poTBQtevabd__lA/exec"

class ImageRequest(BaseModel):
    base64: str

@app.post("/analyze")
def analyze(data: ImageRequest):
    try:
        # Menyiapkan payload untuk dikirim ke Google Apps Script
        payload = {"base64": data.base64}
        
        # Mengirim data menggunakan metode POST
        response = requests.post(GAS_URL, json=payload)
        
        # Mengecek apakah koneksi ke Google Apps Script berhasil
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "message": "Foto berhasil masuk ke folder BASE64!",
                    "url": result.get("url")
                }
            else:
                return {
                    "status": "error", 
                    "message": f"Script Error: {result.get('message')}"
                }
        else:
            return {
                "status": "error", 
                "message": f"HTTP Error {response.status_code}: Gagal terhubung ke Script."
            }
            
    except Exception as e:
        return {
            "status": "error", 
            "message": f"System Error: {str(e)}"
        }

@app.get("/")
def root():
    return {"message": "API Bridge Google Apps Script sedang berjalan!"}    