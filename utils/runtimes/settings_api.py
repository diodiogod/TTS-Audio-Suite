"""Preferences and runtime status only; HTTP requests never install packages."""

from urllib.parse import urlsplit

from .bootstrap import resolve_runtime_dir, runtime_is_ready
from .profiles import get_runtime_profile
from .settings import read_settings, remember_user_directory, save_settings, settings_path


def runtime_status():
    profile = get_runtime_profile("vibevoice_transformers4_shared")
    return {
        **read_settings(),
        "settings_file": str(settings_path()),
        "runtime_folder": str(resolve_runtime_dir(profile)),
        "runtime_ready": runtime_is_ready(profile),
    }


def register_runtime_settings_routes(routes, web):
    remember_user_directory()

    @routes.get("/api/tts-audio-suite/runtime-settings")
    async def get_settings(request):
        try:
            return web.json_response(runtime_status(), headers={"Cache-Control": "no-store"})
        except (ValueError, OSError, RuntimeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    @routes.post("/api/tts-audio-suite/runtime-settings")
    async def update_settings(request):
        # A remote web page must not change the local installation preference.
        origin = urlsplit(request.headers.get("Origin", ""))
        if origin.scheme != request.scheme or origin.netloc != request.host:
            return web.json_response({"error": "Runtime settings require a same-origin request."}, status=403)
        try:
            save_settings(await request.json())
            return web.json_response(runtime_status(), headers={"Cache-Control": "no-store"})
        except (ValueError, OSError, RuntimeError) as error:
            return web.json_response({"error": str(error)}, status=400)
