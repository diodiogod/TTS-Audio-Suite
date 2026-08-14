"""HTTP route for the audio.cpp engine capability panel."""

from __future__ import annotations


def register_audio_cpp_capability_routes(routes, web) -> None:
    @routes.get("/api/tts-audio-suite/audio-cpp-capabilities")
    async def get_audio_cpp_capabilities(_request):
        try:
            from .capabilities import public_capabilities

            return web.json_response(public_capabilities())
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get("/api/tts-audio-suite/audio-cpp-status")
    async def get_audio_cpp_status(_request):
        try:
            from .session import audio_cpp_session_statuses

            return web.json_response({"sessions": audio_cpp_session_statuses()})
        except Exception as exc:
            return web.json_response({"sessions": [], "error": str(exc)}, status=500)

    @routes.post("/api/tts-audio-suite/audio-cpp-stop")
    async def stop_audio_cpp_session(request):
        try:
            from .session import stop_owned_audio_cpp_session

            data = await request.json()
            stopped = stop_owned_audio_cpp_session(str(data.get("session_id", "")))
            if not stopped:
                return web.json_response({"error": "audio.cpp session not found"}, status=404)
            return web.json_response({"status": "stopped"})
        except PermissionError as exc:
            return web.json_response({"error": str(exc)}, status=403)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
