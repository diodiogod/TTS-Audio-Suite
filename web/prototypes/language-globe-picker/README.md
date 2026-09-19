# Language Globe Picker Prototype

Standalone visual test for the TTS Audio Suite language picker. It has no ComfyUI imports or backend requirements.

## Run

From the repository root:

```powershell
python -m http.server 8765 --directory web
```

Open `http://127.0.0.1:8765/prototypes/language-globe-picker/`.

Do not open `index.html` directly as a `file://` URL; browsers block its ES module import.

## Reuse

Copy `web/language-swap-globe.js` into the target project and import:

```js
import {
  createLanguageGlobe,
  languageEntries,
  languageMeta,
  languageTileCode,
} from "./language-swap-globe.js";
```

The host must be a positioned element (`position: relative` or equivalent). Create the globe with:

```js
const globe = createLanguageGlobe(host, "en", {
  durationScale: 1,
  opacity: 0.82,
});

button.addEventListener("pointerenter", () => globe.focus("fr", button));
button.addEventListener("focus", () => globe.focus("fr", button));
```

The button argument lets the renderer choose a visible landmark position away from the active tile. Call `globe.destroy()` when removing the host.

Language tiles may be generated directly from backend codes. Unknown codes remain valid UI choices: `languageMeta()` returns their uppercase code with no invented geographic coordinates, so the globe stays put and hides its landmark until metadata is added. Add a row to `META` when a trustworthy display name and representative location are chosen.

## Coastline data

The globe uses the public-domain [Natural Earth 1:110m land layer](https://www.naturalearthdata.com/downloads/110m-physical-vectors/), simplified into a small JavaScript asset. The outlines are real geographic data, not hand-drawn approximations.

To regenerate with the GitHub CLI and a simplification tolerance of `0.7` degrees:

```powershell
gh api repos/nvkelso/natural-earth-vector/contents/geojson/ne_110m_land.geojson `
  -H "Accept: application/vnd.github.raw+json" |
  node web/prototypes/language-globe-picker/build-land-data.mjs 0.7
```

## Files

- `language-swap-globe.js`: reusable renderer, language metadata, and public API.
- `natural-earth-land-110m.js`: generated, simplified public-domain coastline polygons.
- `build-land-data.mjs`: reproducible Natural Earth simplification tool.
- `index.html`, `prototype.css`, `prototype.js`: dependency-free test harness.
