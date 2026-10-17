# HVAC classifier — Angular UI step by step

Upload HVAC plan images (PNG/JPG). Later: C# API → Python inference → show symbols and metadata in the UI.

You write the code; we advance one step after each check.

**App folder:** `clients/hvac-classifier-ui/`

---

## Big picture (not built yet)

| Step | Layer | Job |
|------|--------|-----|
| A1–A4 | **Angular** | Pick a drawing, preview it, call API, show JSON / overlays |
| B1–B3 | **C#** | Receive `multipart/form-data`, forward to Python |
| C1–C3 | **FastAPI** | Run OpenCV / CNN on `projects/hvac/` pipeline, return detections |

This document covers **A1 only**.

---

## Step A1 — Run the empty Angular app

### Commands

```powershell
cd C:\Users\arash\projectWork\Training\clients\hvac-classifier-ui
npm start
```

Open **http://localhost:4200** (default Angular welcome page).

### Key files

| File | Role |
|------|------|
| `src/main.ts` | Boots Angular |
| `src/app/app.component.ts` | Root logic (`title = 'HVAC Classifier'`) |
| `src/app/app.component.html` | Root template |

---

## Checklist before Step A2

- [ ] `npm start` runs without errors  
- [ ] Browser shows the app at `http://localhost:4200`  

Reply **“A1 done”** or paste errors.

---

## Step A2 (next) — File input for HVAC drawing

`<input type="file" accept="image/*">`, preview with `URL.createObjectURL`, label the UI “Upload HVAC plan”.

## Step A4 (later) — POST to C#

`HttpClient` + `FormData` → `hvac-classifier-api` → show detection JSON in the UI.
