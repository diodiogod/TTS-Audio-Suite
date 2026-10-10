"""Standalone recording controls for the Voice Capture frontend."""

import asyncio
import subprocess
from urllib.parse import urlsplit

from .voice_capture import capture_status, input_devices, start_capture


def register_voice_capture_routes(routes, web):
    def response(payload, status=200):
        return web.json_response(payload, status=status, headers={"Cache-Control": "no-store"})

    def same_origin(request):
        origin = urlsplit(request.headers.get("Origin", ""))
        return origin.scheme == request.scheme and origin.netloc == request.host

    @routes.get("/api/tts-audio-suite/voice-input-devices")
    async def devices(request):
        try:
            return response({"devices": await asyncio.to_thread(input_devices)})
        except subprocess.TimeoutExpired:
            return response({"devices": [], "error": "Microphone device lookup timed out."}, 504)
        except Exception as error:
            return response({"devices": [], "error": str(error)}, 500)

    @routes.post("/api/tts-audio-suite/voice-capture/start")
    async def start(request):
        if not same_origin(request):
            return response({"error": "Recording requires a same-origin request."}, 403)
        try:
            return response(start_capture(await request.json()))
        except (ValueError, TypeError) as error:
            return response({"error": str(error)}, 400)
        except RuntimeError as error:
            return response({"error": str(error)}, 409)

    @routes.get("/api/tts-audio-suite/voice-capture/status")
    async def status(request):
        try:
            return response(capture_status(request.query.get("recording", "")))
        except ValueError as error:
            return response({"error": str(error)}, 404)

    @routes.post("/api/tts-audio-suite/voice-capture/stop")
    async def stop(request):
        if not same_origin(request):
            return response({"error": "Recording requires a same-origin request."}, 403)
        try:
            data = await request.json()
            return response(capture_status(data.get("recording", ""), stop=True))
        except (ValueError, TypeError) as error:
            return response({"error": str(error)}, 400)
