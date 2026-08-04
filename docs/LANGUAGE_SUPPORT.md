# Language Support by Engine

## Language Support by Engine

| Language       | Code | F5-TTS | ChatterBox | ChatterBox 23L | VibeVoice | Higgs Audio 2 | Higgs Audio v3 | IndexTTS-2 | CosyVoice3 | Qwen3-TTS | Granite ASR | Step Audio EditX | Echo-TTS | Fish Audio S2 Pro | Dots TTS | DramaBox | OmniVoice | VoxCPM | MOSS-TTS | MOSS-SoundEffect v2 | RVC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 🇺🇸 **English** | EN   | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Tier 1 | ✅ | ✅ Official model is English-only | ✅ | ✅ All variants | ✅ | ✅ Officially demonstrated prompt language | ✅ |
| 🇨🇳 **Chinese** | ZH   | ❌ | ❌ | ✅ | ✅ | ✅ (Mandarin) | ✅ | ✅ | ✅ + 18 dialects | ✅ | ❌ | ✅ (Mandarin + Sichuanese, Cantonese) | ❌ | ✅ Tier 1 | ✅ (Mandarin; official also exposes YUE separately outside this matrix) | ❌ | ✅ | ✅ All variants; VoxCPM2 also supports nine documented Chinese dialects | ✅ | ✅ Officially demonstrated prompt language | ✅ |
| 🇩🇪 **German**  | DE   | ✅ | ✅ (×3) | ✅ | ✅ (Kugel) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ Tier 2 | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇪🇸 **Spanish** | ES   | ✅ | ❌ | ✅ | ✅ (Kugel) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ Tier 2 | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇫🇷 **French**  | FR   | ✅ | ✅ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ Tier 2 | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇮🇹 **Italian** | IT   | ✅ | ✅ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇯🇵 **Japanese** | JA   | ✅ | ✅ | ✅ | ✅ (Kugel) | ❌ | ✅ | ✅ ? | ✅ | ✅ | ✅ (4.1-2b supports Japanese; plus variant does not) | ✅ | ❌ | ✅ Tier 1 | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇰🇷 **Korean**  | KO   | ❌ | ✅ | ✅ | ✅ (Kugel) | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ Tier 2 | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇷🇺 **Russian** | RU   | ❌ | ✅ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ Tier 2 | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇧🇷 **Portuguese** | PT   | ✅ (BR) | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ✅ (EU/BR*) | ✅ | ❌ | ❌ | ✅ Tier 2 | ✅ (Official PT tag is generic; upstream does not expose separate PT-BR/PT-PT tags and it may lean more European Portuguese than Brazilian Portuguese) | ❌ | ✅ (generic PT; official language space is much broader than this matrix) | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇵🇱 **Polish**  | PL   | ✅ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇮🇳 **Hindi**   | HI   | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇪🇬 **Arabic**  | AR   | ❌ | ❌ | ✅ (Egyptian) | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Tier 2 | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇹🇷 **Turkish** | TR   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇹🇭 **Thai**    | TH   | ✅ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇳🇴 **Norwegian** | NO   | ❌ | ✅ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ VoxCPM2 | ❌ | ❌ | ✅ |
| 🇻🇳 **Vietnamese** | VI   | ❌ | ❌ | ✅ (Viterbox) | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇦🇲 **Armenian** | HY   | ❌ | ✅ | ❌ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| 🇬🇪 **Georgian** | KA   | ❌ | ✅ | ❌ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| 🇩🇰 **Danish**  | DA   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇫🇮 **Finnish** | FI   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇬🇷 **Greek**   | EL   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇮🇱 **Hebrew**  | HE   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇲🇾 **Malay**   | MS   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇳🇱 **Dutch**   | NL   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇸🇪 **Swedish** | SV   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ VoxCPM2 | ✅ | ❌ | ✅ |
| 🇰🇪 **Swahili** | SW   | ❌ | ❌ | ✅ | ✅ (Kugel) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ VoxCPM2 | ✅ (v1.5) | ❌ | ✅ |
| 🇿🇦 **Afrikaans** | AF   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇦🇱 **Albanian** | SQ   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Assamese** | AS   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇪🇸 **Asturian** | AST  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇦🇿 **Azerbaijani** | AZ   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇷🇺 **Bashkir** | BA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇪🇸 **Basque**  | EU   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇧🇾 **Belarusian** | BE   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇧🇩 **Bengali** | BN   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇧🇦 **Bosnian** | BS   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇧🇬 **Bulgarian** | BG   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇪🇸 **Catalan** | CA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇵🇭 **Cebuano** | CEB  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇶 **Central Kurdish** | CKB  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇭🇷 **Croatian** | HR   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇨🇿 **Czech**   | CS   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇷🇺 **Eastern Mari** | MHR  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🌍 **Esperanto** | EO   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇪🇪 **Estonian** | ET   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇪🇸 **Galician** | GL   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Gujarati** | GU   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇭🇹 **Haitian Creole** | HT   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇳🇬 **Hausa**   | HA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇭🇺 **Hungarian** | HU   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇩 **Indonesian** | ID   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇩 **Javanese** | JV   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Kannada** | KN   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇰🇿 **Kazakh**  | KK   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇷🇼 **Kinyarwanda** | RW   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇰🇬 **Kyrgyz**  | KY   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇱🇻 **Latvian** | LV   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇨🇩 **Lingala** | LN   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇱🇹 **Lithuanian** | LT   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇰🇪 **Luo**     | LUO  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇲🇰 **Macedonian** | MK   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Malayalam** | ML   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇲🇹 **Maltese** | MT   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇳🇿 **Māori**   | MI   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Marathi** | MR   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇲🇳 **Mongolian** | MN   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇳🇵 **Nepali**  | NE   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇫🇷 **Occitan** | OC   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇷 **Persian** | FA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇷🇴 **Romanian** | RO   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇿🇦 **Sepedi**  | NSO  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇷🇸 **Serbian** | SR   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇿🇼 **Shona**   | SN   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇸🇰 **Slovak**  | SK   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇸🇮 **Slovene** | SL   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇵🇭 **Tagalog** | TL   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇹🇯 **Tajik**   | TG   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Tamil**   | TA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Telugu**  | TE   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇺🇦 **Ukrainian** | UK   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇵🇰 **Urdu**    | UR   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇨🇳 **Uyghur**  | UG   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇺🇿 **Uzbek**   | UZ   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇿🇦 **Xhosa**   | XH   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇿🇦 **Zulu**    | ZU   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (polished) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇲🇼 **Chichewa/Nyanja** | NY   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇳 **Eastern Punjabi** | PA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇺🇬 **Ganda**   | LG   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇸 **Icelandic** | IS   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇮🇪 **Irish**   | GA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇩🇿 **Kabyle**  | KAB  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇨🇻 **Kabuverdianu** | KEA  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇰🇪 **Kamba**   | KAM  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇻🇦 **Latin**   | LA   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇱🇺 **Luxembourgish** | LB   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇪🇹 **Oromo**   | OM   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇦🇫 **Pashto**  | PS   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇵🇰 **Sindhi**  | SD   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇸🇴 **Somali**  | SO   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🇦🇴 **Umbundu** | UMB  | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 🏴 **Welsh**    | CY   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (usable) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

**Notes:**

- **CosyVoice3 Chinese**: Includes 18+ dialects (Cantonese, Sichuan, Dongbei, Shanghai, etc.)
- **Higgs Audio 2**: Trained on EN, ZH (Mandarin), KO, DE, ES (English majority) - 10M hours AudioVerse dataset
- **Higgs Audio v3**: Official model card lists 102 languages total: 85 polished-production languages and 17 usable but less polished languages.
- **IndexTTS-2**: Trained on 55K+ hours - ZH, EN, JA primary.
- **MOSS-TTS**: Official list includes ZH, EN, DE, ES, FR, JA, IT, HU, KO, RU, FA, AR, PL, PT, CS, DA, SV, EL, TR; HU/FA/CS are not separate columns in this matrix.
- **OmniVoice**: Official model supports 600+ languages; this matrix only shows the suite's comparison subset.
- **VoxCPM**: VoxCPM2 supports 30 languages without language tags; VoxCPM1.5 and the legacy 0.5B model support English and Chinese.
- **RVC**: Language-agnostic voice conversion/post-processing; it is marked supported for every language row.
- Qwen3-TTS: Supports pt-BR only with instruction (voice design or harcoded voices). Base can't do instructions, so it will always ouput pt-PT.