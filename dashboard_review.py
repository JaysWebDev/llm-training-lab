"""
Dashboard Review API for proposals
Lists experiment proposals under experiments/ and allows marking as 'approved'.
This is a minimal FastAPI app intended to slot into your dashboard as a review endpoint.
Usage: uvicorn dashboard_review:APP --host 0.0.0.0 --port 8002
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import shutil
import os

APP = FastAPI()
EXP_DIR = Path("experiments")

@APP.get("/proposals")
def list_proposals():
    proposals = []
    if not EXP_DIR.exists():
        return {"proposals": []}
    for p in sorted(EXP_DIR.glob("run_*")):
        prop = p / "proposal.json"
        meta = p / "meta.json"
        proposals.append({
            "run_dir": str(p),
            "proposal": json.loads(prop.read_text()) if prop.exists() else None,
            "meta": json.loads(meta.read_text()) if meta.exists() else None
        })
    return {"proposals": proposals}

@APP.post("/approve")
def approve(run_dir: str):
    p = Path(run_dir)
    if not p.exists():
        raise HTTPException(status_code=404, detail="run_dir not found")
    approved_file = p / "approved.json"
    approved_file.write_text(json.dumps({"approved_at": __import__("datetime").datetime.utcnow().isoformat()}), encoding="utf-8")
    try:
        with open(p / "proposal.json", "r") as f:
            proposal = json.load(f)
    except Exception:
        proposal = None
    return JSONResponse({"status": "approved", "run_dir": str(p), "proposal": proposal})
