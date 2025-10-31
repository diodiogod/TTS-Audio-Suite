/**
 * 🌍 Language Constants
 * Single source of truth for supported languages - fetched from backend API
 */

let SUPPORTED_LANGUAGES = ["en", "de", "fr", "ja", "es", "it", "pt", "th", "no"]; // fallback

export async function loadSupportedLanguages() {
    try {
        const response = await fetch("/api/tts-audio-suite/available-languages");
        if (response.ok) {
            const data = await response.json();
            if (data.languages && Array.isArray(data.languages)) {
                SUPPORTED_LANGUAGES = data.languages;
            }
        }
    } catch (err) {
        console.warn("Could not load supported languages from API, using fallback:", err);
    }
    return SUPPORTED_LANGUAGES;
}

export function getSupportedLanguages() {
    return SUPPORTED_LANGUAGES;
}

export function isLanguageCode(code) {
    return SUPPORTED_LANGUAGES.includes(code.toLowerCase());
}
