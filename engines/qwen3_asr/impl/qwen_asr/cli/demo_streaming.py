# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Minimal web demo for Qwen3ASRModel Streaming Inference (vLLM backend).

Install:
  pip install qwen-asr[vllm]

Run:
  python streaming/demo_qwen3_asr_vllm_streaming.py
Open:
  http://127.0.0.1:7860
"""
import argparse
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from flask import Flask, Response, jsonify, request
from qwen_asr import Qwen3ASRModel


@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float


app = Flask(__name__)

global asr
global UNFIXED_CHUNK_NUM
global UNFIXED_TOKEN_NUM
global CHUNK_SIZE_SEC

SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60


def _gc_sessions():
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            asr.finish_streaming_transcribe(SESSIONS[sid].state)
        except Exception:
            pass
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Qwen3-ASR Streaming</title>
  <style>
    :root{
      --bg:#ffffff;
      --card:#ffffff;
      --muted:#5b6472;
      --text:#0f172a;
      --border:#e5e7eb;
      --ok:#059669;
      --warn:#d97706;
      --danger:#e11d48;
    }

    html, body { height: 100%; }

    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans";
      background: var(--bg);
      color:var(--text);
    }

    .wrap{
      height: 100vh;
      max-width: none;
      margin: 0;
      padding: 16px;
      box-sizing: border-box;
      display: flex;
    }

    .card{
      width: 100%;
      height: 100%;
      background: var(--card);
      border:1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-sizing: border-box;
      box-shadow: 0 10px 30px rgba(0,0,0,.06);

      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
    }

    h1{ font-size: 16px; margin: 0; letter-spacing:.2px;}

    .row{ display:flex; gap:12px; align-items:center; flex-wrap: wrap; }

    button{
      border:1px solid var(--border); border-radius: 12px;
      padding: 10px 14px; cursor:pointer; color:var(--text);
      background: #f8fafc;
      transition: transform .05s ease, background .15s ease, border-color .15s ease;
      font-weight: 700;
    }
    button:hover{ background: #f1f5f9; border-color:#cbd5e1; }
    button:active{ transform: translateY(1px); }
    button.primary{ border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    button.danger{ border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }
    button:disabled{ opacity:.5; cursor:not-allowed; }

    .pill{
      font-size: 12px; padding: 6px 10px; border-radius: 999px;
      border:1px solid var(--border); color: var(--muted);
      background: #f8fafc;
      user-select:none;
    }
    .pill.ok{ color: #065f46; border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    .pill.warn{ color: #92400e; border-color: rgba(217,119,6,.35); background: rgba(217,119,6,.10); }
    .pill.err{ color: #9f1239; border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }

    .panel{
      border:1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
      padding: 12px;
    }

    .panel.textpanel{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .label{ color:var(--muted); font-size: 12px; margin-bottom: 6px; }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }

    #text{
      flex: 1;
      min-height: 0;
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 15px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #f8fafc;
      overflow: auto;
    }

    a{ color: #2563eb; text-decoration:none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Qwen3-ASR Streaming</h1>

      <div class="row">
        <button id="btnStart" class="primary">Start / 开始</button>
        <button id="btnStop" class="danger" disabled>Stop / 停止</button>
        <span id="status" class="pill warn">Idle / 未开始</span>
        <a href="javascript:void(0)" id="btnClear" class="mono" style="margin-left:auto;">Clear / 清空</a>
      </div>

      <div class="panel">
        <div class="label">Language / 语言</div>
        <div id="lang" class="mono">—</div>
      </div>

      <div class="panel textpanel">
        <div class="label">Text / 文本</div>
        <div id="text"></div>
      </div>
    </div>
  </div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);

  const btnStart = $("btnStart");
  const btnStop  = $("btnStop");
  const btnClear = $("btnClear");
  const statusEl = $("status");
  const langEl   = $("lang");
  const textEl   = $("text");

  const CHUNK_MS = 500;
  const TARGET_SR = 16000;

  let audioCtx = null;
  let processor = null;
  let source = null;
  let mediaStream = null;

  let sessionId = null;
  let running = false;

  let buf = new Float32Array(0);
  let pushing = false;

  function setStatus(text, cls){
    statusEl.textContent = text;
    statusEl.className = "pill " + (cls || "");
  }

  function lockUI(on){
    btnStart.disabled = on;
    btnStop.disabled = !on;
  }

  function concatFloat32(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function resampleLinear(input, srcSr, dstSr){
    if (srcSr === dstSr) return input;
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++){
      const x = i / ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, input.length - 1);
      const t = x - x0;
      out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
  }

  async function apiStart(){
    const r = await fetch("/api/start", {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    const j = await r.json();
    sessionId = j.session_id;
  }

  async function apiPushChunk(float32_16k){
    const r = await fetch("/api/chunk?session_id=" + encodeURIComponent(sessionId), {
      method: "POST",
      headers: {"Content-Type":"application/octet-stream"},
      body: float32_16k.buffer
    });
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  async function apiFinish(){
    const r = await fetch("/api/finish?session_id=" + encodeURIComponent(sessionId), {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  btnClear.onclick = () => { textEl.textContent = ""; };

  async function stopAudioPipeline(){
    try{
      if (processor){ processor.disconnect(); processor.onaudioprocess = null; }
      if (source) source.disconnect();
      if (audioCtx) await audioCtx.close();
      if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    }catch(e){}
    processor = null; source = null; audioCtx = null; mediaStream = null;
  }

  btnStart.onclick = async () => {
    if (running) return;

    textEl.textContent = "";
    langEl.textContent = "—";
    buf = new Float32Array(0);

    try{
      setStatus("Starting… / 启动中…", "warn");
      lockUI(true);

      await apiStart();

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      source = audioCtx.createMediaStreamSource(mediaStream);

      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

      processor.onaudioprocess = (e) => {
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const resampled = resampleLinear(input, audioCtx.sampleRate, TARGET_SR);
        buf = concatFloat32(buf, resampled);
        if (!pushing) pump();
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      running = true;
      setStatus("Listening… / 识别中…", "ok");

    }catch(err){
      console.error(err);
      setStatus("Start failed / 启动失败: " + err.message, "err");
      lockUI(false);
      running = false;
      sessionId = null;
      await stopAudioPipeline();
    }
  };

  async function pump(){
    if (pushing) return;
    pushing = true;

    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

    try{
      while (running && buf.length >= chunkSamples){
        const chunk = buf.slice(0, chunkSamples);
        buf = buf.slice(chunkSamples);

        const j = await apiPushChunk(chunk);
        langEl.textContent = j.language || "—";
        textEl.textContent = j.text || "";
        if (running) setStatus("Listening… / 识别中…", "ok");
      }
    }catch(err){
      console.error(err);
      if (running) setStatus("Backend error / 后端错误: " + err.message, "err");
    }finally{
      pushing = false;
    }
  }

  btnStop.onclick = async () => {
    if (!running) return;

    running = false;
    setStatus("Finishing… / 收尾中…", "warn");
    lockUI(false);

    await stopAudioPipeline();

    try{
      if (sessionId){
        const j = await apiFinish();
        langEl.textContent = j.language || "—";
        textEl.textContent = j.text || "";
      }
      setStatus("Stopped / 已停止", "");
    }catch(err){
      console.error(err);
      setStatus("Finish failed / 收尾失败: " + err.message, "err");
    }finally{
      sessionId = null;
      buf = new Float32Array(0);
      pushing = false;
    }
  };
})();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.post("/api/start")
def api_start():
    session_id = uuid.uuid4().hex
    state = asr.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    return jsonify({"session_id": session_id})


@app.post("/api/chunk")
def api_chunk():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "expect application/octet-stream"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes length not multiple of 4"}), 400

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

    asr.streaming_transcribe(wav, s.state)

    return jsonify(
        {
            "language": getattr(s.state, "language", "") or "",
            "text": getattr(s.state, "text", "") or "",
        }
    )


@app.post("/api/finish")
def api_finish():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    asr.finish_streaming_transcribe(s.state)
    out = {
        "language": getattr(s.state, "language", "") or "",
        "text": getattr(s.state, "text", "") or "",
    }
    SESSIONS.pop(session_id, None)
    return jsonify(out)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR Streaming Web Demo (vLLM backend)")
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B", help="Model name or local path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")

    p.add_argument("--unfixed-chunk-num", type=int, default=4)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()

    global asr
    global UNFIXED_CHUNK_NUM
    global UNFIXED_TOKEN_NUM
    global CHUNK_SIZE_SEC

    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec

    asr = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=32,
    )
    print("Model loaded.")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()