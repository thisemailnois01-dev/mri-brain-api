"""FastAPI MRI Classifier — Railway ready"""
import io, base64, os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 384
MEAN, STD  = [0.485,0.456,0.406], [0.229,0.224,0.225]

class MRIClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False,
                                          num_classes=0, global_pool="avg")
        nf = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(nf,512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(dropout/2),
            nn.Linear(256,num_classes))
    def forward(self, x): return self.classifier(self.backbone(x))

ckpt        = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = ckpt["class_names"]
model       = MRIClassifier(ckpt["model_name"], ckpt["num_classes"]).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded | classes={CLASS_NAMES} | device={DEVICE}")

preprocess = A.Compose([A.Resize(IMG_SIZE,IMG_SIZE),
                         A.Normalize(mean=MEAN,std=STD), ToTensorV2()])

class GradCAM:
    def __init__(self, model, layer):
        self.model=model; self.grads=None; self.acts=None
        layer.register_forward_hook(lambda m,i,o: setattr(self,"acts",o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(self,"grads",go[0].detach()))
    def __call__(self, t, cls=None):
        t=t.to(DEVICE).unsqueeze(0).requires_grad_(True)
        lg=self.model(t)
        if cls is None: cls=lg.argmax(1).item()
        self.model.zero_grad(); lg[0,cls].backward()
        w=self.grads.mean([2,3],keepdim=True)
        cam=torch.relu((w*self.acts).sum(1)).squeeze().cpu().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cam, cls, torch.softmax(lg,1)[0].detach().cpu().numpy()

gradcam = GradCAM(model, model.backbone.blocks[-1])

app = FastAPI(title="MRI Brain Tumor Classifier", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path("templates/index.html")
    return p.read_text() if p.exists() else "<h1>MRI Classifier API</h1><a href='/docs'>Docs</a>"

@app.get("/health")
async def health(): return {"status":"ok","classes":CLASS_NAMES,"device":DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Need an image file.")
    try:
        data   = await file.read()
        arr    = np.frombuffer(data, np.uint8)
        bgr    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None: raise HTTPException(400, "Cannot decode image.")
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = preprocess(image=rgb)["image"]

        cam, pidx, probs = gradcam(tensor)
        h, w = rgb.shape[:2]
        camr = cv2.resize(cam, (w,h))
        heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*camr), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        over = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)

        def b64(img):
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf).decode()

        cls   = CLASS_NAMES[pidx]
        tumor = cls.lower() not in ["notumor","no_tumor","no tumor","normal"]
        return JSONResponse({
            "prediction":         cls,
            "is_tumor":           tumor,
            "confidence":         float(probs[pidx]),
            "confidence_pct":     f"{probs[pidx]*100:.1f}%",
            "all_probabilities":  {c:float(p) for c,p in zip(CLASS_NAMES,probs)},
            "original_b64":       b64(rgb),
            "heatmap_b64":        b64(heat),
            "gradcam_overlay_b64":b64(over),
            "explanation": (f"The model detected {cls.upper()} with "
                            f"{probs[pidx]*100:.1f}% confidence. "
                            f"Red/warm regions in the Grad-CAM map highlight "
                            f"the areas the model focused on.")
        })
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT",8000)))
