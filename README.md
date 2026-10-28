# Campus Eco & Waste Sorting Identifier (IIT Roorkee)

Upload a photo of a waste item. The site predicts its category with a MobileNetV3 fine-tuned on TrashNet
and shows every campus bin that accepts it on a map, nearest first if the user shares their location.

The model learns TrashNet's 6 classes, which the website groups into bins:

| Model label (TrashNet) | Website category |
|---|---|
| cardboard, paper | Paper / Cardboard |
| plastic | Plastics |
| glass | Glass |
| metal | Metal |
| trash | Landfill |

TrashNet has no food or electronics photos, so **Compost and E-Waste are not predicted by the model**.
They stay on the map (filters) and in the "No, it's…" correction menu, and the site tells users this.
Adding them later only needs new training folders plus an entry in `classes.json`.

## Project structure

```
campus-eco-sorter/
├── backend/
│   ├── main.py              FastAPI app: /api/predict, /api/bins, /api/feedback, serves the website
│   ├── model.py             Loads your PyTorch model, preprocessing, prediction (demo mode if no model)
│   ├── bins.py              Bin lookup + distance sorting
│   ├── data/
│   │   ├── classes.json     Model output order  <-- CHECK THIS
│   │   ├── categories.json  Names, colours, disposal tips per category
│   │   └── bins.json        Campus bin locations  <-- REPLACE WITH SURVEY DATA
│   ├── model/               Trained model (mobilenetv3_waste/, committed to git)
│   ├── feedback/            User-corrected photos are saved here for retraining
│   └── requirements.txt
├── frontend/
│   ├── index.html  style.css  app.js     Upload page + Leaflet/OpenStreetMap bin map
├── training/
│   ├── train_kaggle.ipynb   Kaggle notebook: train on TrashNet + evaluate (Week 1)
│   ├── evaluate.py          Optional: same metrics on your laptop
│   └── export_torchscript.py  Optional: convert .pth -> TorchScript .pt
├── render.yaml              Render deployment config
└── run.bat                  Windows: install + start the server
```

## Run it (Windows)

1. Install Python 3.10+.
2. Double-click `run.bat`, or in a terminal:
   ```
   cd backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```
3. Open http://localhost:8000. API docs (to test endpoints by hand) are at http://localhost:8000/docs.

Without a model file the site runs in **demo mode** (a banner says so, predictions are placeholders),
so the frontend team can work before the weights are ready.

## Train the model on Kaggle

1. Kaggle → **Create → New Notebook → File → Import Notebook** → choose `training/train_kaggle.ipynb`.
2. Settings: Accelerator **GPU T4**, Internet **On**. Then **Run All**.
   (No input needed: the notebook downloads TrashNet from GitHub if none is attached.)
4. From the **Output** tab download `mobilenetv3_waste.pth` (plus the charts for your report).

## Plug in your trained model

1. Copy the weights into `backend/model/`, e.g. `backend/model/mobilenetv3_waste.pth`.
   Accepted: a `state_dict`, a checkpoint dict with `model_state_dict`, a whole saved model, or TorchScript.
2. **Class order.** The notebook already uses the order in `backend/data/classes.json`
   (`cardboard, glass, metal, paper, plastic, trash`), so nothing to change. If you train a model some other way,
   make `model_output_order` match its `class_to_idx`; a wrong order gives confidently wrong answers.
3. If you fine-tuned `mobilenet_v3_small` (not large), set `MODEL_ARCH=small` before starting:
   `set MODEL_ARCH=small` (cmd) or `$env:MODEL_ARCH="small"` (PowerShell).
4. The code assumes you replaced only the final layer: `model.classifier[3] = nn.Linear(in_features, 6)`.
   If you changed the head differently, edit `build_mobilenet()` in `backend/model.py` to match.
5. Restart and visit http://localhost:8000/api/health. It should show `"demo_mode": false`.

Preprocessing is Resize(256) → CenterCrop(224) → ImageNet normalisation. If your validation transform was
different, change `build_transform()` in `model.py` to match it.

## Week 1: Evaluate the model

The Kaggle notebook already reports test accuracy (per TrashNet class and per website bin), per-class
precision / recall / F1, a confusion matrix and example mistakes.

To also measure how the model does on real phone photos (TrashNet is all plain white backgrounds),
put a few photos in `phone_test/<model label>/*.jpg` (folders `cardboard`, `glass`, `metal`, `paper`,
`plastic`, `trash`) and run on your laptop:

```
pip install -r training\requirements.txt
python training/evaluate.py --data path\to\phone_test --tag phone
```

Results go to `training/results/` (report, confusion matrix PNG, misclassified CSV), plus inference time
and model size.

## Weeks 2–3: Campus bin survey

`backend/data/bins.json` currently holds **sample placeholder locations** (landmark names with approximate
coordinates, all marked `"verified": false`, shown as "Unverified" on the site). For each real bin:

```json
{"id": "RJB-01", "name": "Rajendra Bhawan mess", "area": "Hostels",
 "lat": 29.86850, "lon": 77.89750, "accepts": ["compost", "plastics"], "verified": true}
```

Get lat/lon by long-pressing the spot in Google Maps. `accepts` uses the website category ids
(`plastics`, `paper`, `glass`, `metal`, `landfill`, `compost`, `ewaste`). No retraining is
needed, just restart the server.

## Week 4: Testing

- `/api/feedback` saves photos users corrected into `backend/feedback/<correct category>/` with a log at
  `feedback/feedback_log.csv`, ready to add to the training set.
- Low-confidence threshold (default 60%) can be changed with `CONF_THRESHOLD=0.7`.
- Test on phones: the upload button opens the camera on mobile. Location sorting needs HTTPS when deployed
  (localhost is fine for testing).

## Deploying (GitHub + Render, free)

1. Push this folder to a GitHub repository (the model folder is included; `.venv` and `feedback` are not).
2. On https://render.com sign in with GitHub → **New → Blueprint** → pick the repo → **Apply**.
   `render.yaml` tells Render everything: install `backend/requirements.txt` (CPU-only PyTorch),
   start `uvicorn main:app`, and check `/api/health`.
3. The first build takes ~5–10 minutes. You get a link like `https://campus-eco-sorter.onrender.com`.

Free plan notes: the site sleeps after 15 minutes without visitors and takes about a minute to wake up,
so open it a minute before a demo. Photos saved via "Was this correct?" are lost when Render restarts
(free plan has no permanent disk). Every `git push` redeploys automatically.
