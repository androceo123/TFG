\
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import yaml
import uvicorn
from fastapi import FastAPI, Request, Response
import httpx

from waf_ml.features.http_features import extract_http_features


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


app = FastAPI()
cfg: Dict[str, Any] = {}
model_obj: Any = None


@app.on_event("startup")
async def _startup():
    global cfg, model_obj
    config_path = app.state.config_path
    cfg = load_config(config_path)

    model_path = cfg.get("model_path")
    if model_path:
        model_obj = joblib.load(model_path)
    else:
        model_obj = None

    log_path = Path(cfg.get("log_path", "logs/waf_events.jsonl"))
    ensure_parent(log_path)
    log_path.touch(exist_ok=True)


def _log_event(ev: Dict[str, Any]) -> None:
    log_path = Path(cfg.get("log_path", "logs/waf_events.jsonl"))
    ensure_parent(log_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _decision(features: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Return {allow: bool, reason: str, score/label...} based on loaded model.\"\"\"
    model_type = cfg.get("model_type", "ocsvm")
    if model_obj is None:
        return {"allow": True, "reason": "no_model_loaded"}

    # NOTE: This is a starter: you likely want a stable feature vector builder here
    # that matches your training scripts.
    # We take a subset and fill missing keys with 0.
    keys = [
        "uri_len","path_depth","query_len","n_query_params","max_param_value_len",
        "uri_pct_non_alnum_ratio","encoded","suspicious_tokens_count","has_suspicious_tokens",
        "uncommon_method","req_content_length","body_len",
        "method_GET","method_POST","method_HEAD","method_PUT","method_DELETE","method_PATCH","method_OPTIONS",
        "method_TRACE","method_CONNECT","method_OTHER",
    ]
    x = [[float(features.get(k, 0) or 0) for k in keys]]

    if model_type == "ocsvm":
        pred = model_obj.predict(x)[0]  # +1 normal, -1 anomaly
        anomaly = (pred == -1)
        allow = not (cfg.get("block_on_anomaly", True) and anomaly and cfg.get("mode") == "active")
        return {"allow": allow, "reason": "ocsvm", "pred": int(pred), "anomaly": bool(anomaly)}

    if model_type == "multiclass":
        pred = model_obj.predict(x)[0]
        allow = (cfg.get("mode") != "active") or (pred == "NORMAL")
        return {"allow": allow, "reason": "multiclass", "pred": str(pred)}

    if model_type == "multilabel":
        bundle = model_obj
        pipe = bundle["pipeline"]
        mlb = bundle["mlb"]
        Y = pipe.predict(x)
        labels = mlb.inverse_transform(Y)[0]
        allow = (cfg.get("mode") != "active") or (len(labels) == 0)
        return {"allow": allow, "reason": "multilabel", "labels": list(labels)}

    return {"allow": True, "reason": "unknown_model_type"}


@app.api_route("/{full_path:path}", methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS","HEAD"])
async def proxy(full_path: str, request: Request):
    upstream = cfg["upstream_base"].rstrip("/")
    url = f"{upstream}/{full_path}"

    body = await request.body()
    headers = dict(request.headers)

    feats = extract_http_features(
        method=request.method,
        uri=(request.url.path + (('?' + request.url.query) if request.url.query else '')),
        headers=headers,
        body=body,
    )
    decision = _decision(feats)

    ev = {
        "ts": time.time(),
        "method": request.method,
        "path": full_path,
        "url": str(request.url),
        "client": request.client.host if request.client else None,
        "decision": decision,
        "features": feats,
    }
    _log_event(ev)

    if not decision.get("allow", True) and cfg.get("mode") == "active":
        return Response(content="Blocked by WAF-ML", status_code=403)

    # forward request
    async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in headers.items() if k.lower() != "host"},
            content=body,
            params=dict(request.query_params),
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    app.state.config_path = args.config
    c = load_config(args.config)
    uvicorn.run(app, host=c.get("listen_host","0.0.0.0"), port=int(c.get("listen_port",8080)))


if __name__ == "__main__":
    main()
