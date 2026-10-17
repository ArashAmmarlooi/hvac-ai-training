# Full-stack HVAC classifier (planned layout)

Angular + C# + Python for **HVAC drawing / symbol** workflows. Separate from the Python-only lessons under `projects/hvac/` and `projects/fruit/`.

```
clients/
  hvac-classifier-ui/      ← Angular upload UI (Step A1)
  hvac-classifier-api/     ← C# ASP.NET Core (later)
services/
  hvac-inference-api/      ← Python FastAPI (OpenCV / CNN later)
```

**Data flow (later):**

```
Browser (Angular)  →  upload plan image  →  C# API controller
                              →  FastAPI  →  masks / detections / class
                              →  C#  →  JSON  →  Angular shows results
```

**Lesson guide:** [`../guide/06_angular_step_by_step.md`](../guide/06_angular_step_by_step.md)
