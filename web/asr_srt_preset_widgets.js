import { app } from "../../scripts/app.js";
import { setupSrtAdvancedOptionsPanel } from "./asr_srt_advanced_options_panel_compact.js";

// SRT Advanced Options UI Control
// Presets seed recommended values and switch to Custom when the user diverges.
// Heuristic language profiles seed editable text fields and preserve manual overrides.

const SRT_FIELDS = [
    "srt_max_chars_per_line",
    "srt_max_lines",
    "srt_max_duration",
    "srt_min_duration",
    "srt_min_gap",
    "srt_max_cps",
    "tts_ready_mode",
    "tts_ready_paragraph_mode",
];

const HEURISTIC_FIELDS = [
    "merge_dangling_tail_allowlist",
    "merge_incomplete_keywords",
];

const PRESET_VALUES = {
    "Netflix-Standard": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 7.0,
        srt_min_duration: 0.85,
        srt_min_gap: 0.2,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "Broadcast": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "Fast speech": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 0.8,
        srt_min_gap: 0.4,
        srt_max_cps: 20.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "Mobile": {
        srt_max_chars_per_line: 32,
        srt_max_lines: 2,
        srt_max_duration: 5.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "TTS-Ready": {
        srt_max_chars_per_line: 240,
        srt_max_lines: 1,
        srt_max_duration: 12.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.8,
        srt_max_cps: 17.0,
        tts_ready_mode: true,
        tts_ready_paragraph_mode: false,
    },
    "TTS-Ready (Paragraphs)": {
        srt_max_chars_per_line: 320,
        srt_max_lines: 1,
        srt_max_duration: 24.0,
        srt_min_duration: 1.2,
        srt_min_gap: 1.0,
        srt_max_cps: 15.0,
        tts_ready_mode: true,
        tts_ready_paragraph_mode: true,
    },
};

const HEURISTIC_PROFILE_VALUES = {
    "Auto": {
        merge_dangling_tail_allowlist: "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
        merge_incomplete_keywords: "what,why,how,where,who,which,when",
    },
    "English": {
        merge_dangling_tail_allowlist: "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
        merge_incomplete_keywords: "what,why,how,where,who,which,when",
    },
    "Portuguese (Brazil)": {
        merge_dangling_tail_allowlist: "o,a,os,as,um,uma,uns,umas,de,do,da,dos,das,e,ou,mas,se,que,como,quando,onde,quem,para,pra,por,com,sem,em,no,na,nos,nas,ao,aos,pelo,pela,pelos,pelas",
        merge_incomplete_keywords: "o que,por que,porque,como,onde,quem,qual,quais,quando",
    },
    "Spanish": {
        merge_dangling_tail_allowlist: "el,la,los,las,un,una,unos,unas,de,del,y,o,pero,si,que,como,cuando,donde,quien,para,por,con,sin,en,al,a,lo",
        merge_incomplete_keywords: "lo que,por qué,por que,porque,como,cómo,donde,dónde,quien,quién,cual,cuál,cuales,cuáles,cuando,cuándo",
    },
    "French": {
        merge_dangling_tail_allowlist: "le,la,les,un,une,des,de,du,et,ou,mais,si,que,comme,quand,qui,pour,par,avec,sans,en,dans,sur,au,aux,à",
        merge_incomplete_keywords: "ce que,pourquoi,comment,où,qui,quel,quelle,quels,quelles,quand",
    },
    "Italian": {
        merge_dangling_tail_allowlist: "il,lo,la,i,gli,le,un,una,di,del,della,e,o,ma,se,che,come,quando,dove,chi,per,con,senza,in,su,a,da",
        merge_incomplete_keywords: "che cosa,perché,perche,come,dove,chi,quale,quali,quando",
    },
    "German": {
        merge_dangling_tail_allowlist: "der,die,das,ein,eine,und,oder,aber,wenn,dass,wie,wann,wo,wer,für,mit,ohne,in,an,auf,zu,von,aus,bei,nach",
        merge_incomplete_keywords: "was,warum,wieso,wie,wo,wer,welche,welcher,welches,wann",
    },
    "Dutch": {
        merge_dangling_tail_allowlist: "de,het,een,en,of,maar,als,dat,hoe,wanneer,waar,wie,voor,met,zonder,in,op,aan,van,bij,naar,uit",
        merge_incomplete_keywords: "wat,waarom,hoe,waar,wie,welke,wanneer",
    },
    "Russian": {
        merge_dangling_tail_allowlist: "и,а,но,да,или,что,чтобы,как,если,то,в,во,на,за,к,ко,от,до,из,у,о,об,обо,с,со,по,для,без,при",
        merge_incomplete_keywords: "что,как,почему,зачем,где,когда,кто,какой,какая,какие,который",
    },
    "Romanian": {
        merge_dangling_tail_allowlist: "și,sau,dar,dacă,că,cum,când,unde,cine,ce,pentru,cu,fără,în,pe,la,de,din,un,o,niște",
        merge_incomplete_keywords: "ce,de ce,cum,unde,cine,care,când",
    },
    "Indonesian": {
        merge_dangling_tail_allowlist: "dan,atau,tapi,kalau,yang,karena,bahwa,seperti,ketika,di,ke,dari,untuk,dengan,tanpa,pada,dalam,seorang,sebuah,itu,ini",
        merge_incomplete_keywords: "apa,kenapa,mengapa,bagaimana,di mana,siapa,yang mana,kapan",
    },
    "Malay": {
        merge_dangling_tail_allowlist: "dan,atau,tetapi,kalau,yang,kerana,bahawa,seperti,apabila,di,ke,dari,untuk,dengan,tanpa,pada,dalam,seorang,sebuah,itu,ini",
        merge_incomplete_keywords: "apa,kenapa,mengapa,bagaimana,di mana,siapa,yang mana,bila",
    },
    "Turkish": {
        merge_dangling_tail_allowlist: "ve,veya,ama,eğer,ki,çünkü,gibi,için,ile,olmadan,da,de,bu,şu,o,bir",
        merge_incomplete_keywords: "ne,neden,niye,nasıl,nerede,kim,hangi,ne zaman",
    },
    "Polish": {
        merge_dangling_tail_allowlist: "i,lub,ale,że,żeby,jak,kiedy,gdzie,kto,co,dla,z,bez,w,na,do,od,po,przy,o,u",
        merge_incomplete_keywords: "co,dlaczego,jak,gdzie,kto,który,która,które,kiedy",
    },
    "Czech": {
        merge_dangling_tail_allowlist: "a,nebo,ale,že,aby,jak,když,kde,kdo,co,pro,s,bez,v,na,do,od,po,u,o",
        merge_incomplete_keywords: "co,proč,jak,kde,kdo,který,která,které,kdy",
    },
    "Swedish": {
        merge_dangling_tail_allowlist: "och,eller,men,om,att,som,när,var,vem,för,med,utan,i,på,till,från,av,en,ett,den,det",
        merge_incomplete_keywords: "vad,varför,hur,var,vem,vilken,vilka,när",
    },
    "Danish": {
        merge_dangling_tail_allowlist: "og,eller,men,hvis,at,som,når,hvor,hvem,for,med,uden,i,på,til,fra,af,en,et,den,det",
        merge_incomplete_keywords: "hvad,hvorfor,hvordan,hvor,hvem,hvilken,hvilke,hvornår",
    },
    "Finnish": {
        merge_dangling_tail_allowlist: "ja,tai,mutta,jos,että,kuten,kun,missä,kuka,mikä,sekä,eli",
        merge_incomplete_keywords: "mikä,miksi,miten,missä,kuka,kumpi,milloin",
    },
    "Greek": {
        merge_dangling_tail_allowlist: "και,ή,αλλά,αν,ότι,πως,όπως,όταν,όπου,ποιος,για,με,χωρίς,σε,από,προς,στο,στη,στον,στην,το,τη,τον,την,ένα,μια",
        merge_incomplete_keywords: "τι,γιατί,πώς,πού,ποιος,ποια,ποιο,ποιοι,πότε",
    },
};

function isSrtOptionsNode(node) {
    return node.comfyClass === "SRTAdvancedOptionsNode";
}

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function setWidgetValue(widget, value) {
    if (!widget) {
        return;
    }
    widget.value = value;
    widget.callback?.(widget.value);
}

function normalizeComparableValue(widget, value) {
    if (typeof value === "boolean") {
        return Boolean(value);
    }
    if (typeof value === "number") {
        return Number(value);
    }
    if (typeof widget?.value === "boolean") {
        return Boolean(value);
    }
    if (typeof widget?.value === "number") {
        const numeric = Number(value);
        return Number.isNaN(numeric) ? value : numeric;
    }
    return value;
}

function widgetValueMatches(widget, expectedValue) {
    const actual = normalizeComparableValue(widget, widget?.value);
    const expected = normalizeComparableValue(widget, expectedValue);
    if (typeof actual === "number" && typeof expected === "number") {
        return Math.abs(actual - expected) < 1e-9;
    }
    return actual === expected;
}

function bindWidgetValueHandler(widget, onChange) {
    if (!widget) {
        return;
    }

    if (!widget.__ttsAudioSuiteValueHandlers) {
        widget.__ttsAudioSuiteValueHandlers = [];
    }
    if (!widget.__ttsAudioSuiteValueHandlers.includes(onChange)) {
        widget.__ttsAudioSuiteValueHandlers.push(onChange);
    }

    if (widget.__ttsAudioSuiteValueBound) {
        return;
    }

    const originalCallback = widget.callback;
    widget.callback = function (...args) {
        const result = originalCallback ? originalCallback.apply(this, args) : undefined;
        for (const handler of widget.__ttsAudioSuiteValueHandlers || []) {
            try {
                handler(widget.value);
            } catch (error) {
                console.warn("SRT Advanced Options widget handler failed:", error);
            }
        }
        return result;
    };

    widget.__ttsAudioSuiteValueBound = true;
}

function applyPreset(node, preset) {
    if (!preset || preset === "Custom") {
        return;
    }

    const values = PRESET_VALUES[preset];
    if (!values) {
        return;
    }

    node.__srtLastPresetBaseline = preset;
    node.__applyingSrtPreset = true;
    try {
        for (const field of Object.keys(values)) {
            const widget = findWidgetByName(node, field);
            if (!widget || values[field] === undefined) {
                continue;
            }
            setWidgetValue(widget, values[field]);
        }
    } finally {
        node.__applyingSrtPreset = false;
    }
}

function applyPresetState(node) {
    if (!isSrtOptionsNode(node)) {
        return;
    }

    const presetWidget = findWidgetByName(node, "srt_preset");
    if (!presetWidget) {
        return;
    }

    const preset = presetWidget.value;
    if (preset && preset !== "Custom") {
        applyPreset(node, preset);
    }
}

function applyTtsReadyFieldState(node) {
    if (node.__applyingSrtPreset) {
        return;
    }

    const ttsReadyWidget = findWidgetByName(node, "tts_ready_mode");
    const isTtsReady = Boolean(ttsReadyWidget && ttsReadyWidget.value);

    const paragraphWidget = findWidgetByName(node, "tts_ready_paragraph_mode");
    const maxLinesWidget = findWidgetByName(node, "srt_max_lines");

    if (paragraphWidget) {
        paragraphWidget.disabled = !isTtsReady;
    }

    if (maxLinesWidget) {
        maxLinesWidget.disabled = isTtsReady;
    }

    if (isTtsReady) {
        if (maxLinesWidget && maxLinesWidget.value !== 1) {
            node.__applyingTtsReadyNormalization = true;
            try {
                setWidgetValue(maxLinesWidget, 1);
            } finally {
                node.__applyingTtsReadyNormalization = false;
            }
        }
    }
}

function matchesPresetValues(node, preset) {
    const values = PRESET_VALUES[preset];
    if (!values) {
        return false;
    }

    for (const field of Object.keys(values)) {
        const widget = findWidgetByName(node, field);
        if (!widget) {
            return false;
        }

        if (!widgetValueMatches(widget, values[field])) {
            return false;
        }
    }

    return true;
}

function srtPresetHandler(node) {
    if (!isSrtOptionsNode(node)) {
        return;
    }

    const presetWidget = findWidgetByName(node, "srt_preset");
    if (!presetWidget) {
        return;
    }

    applyPresetState(node);
    applyTtsReadyFieldState(node);
}

function srtFieldEdited(node) {
    if (!isSrtOptionsNode(node) || node.__applyingSrtPreset || node.__applyingTtsReadyNormalization) {
        return;
    }

    const presetWidget = findWidgetByName(node, "srt_preset");
    if (!presetWidget || presetWidget.value === "Custom") {
        return;
    }

    if (matchesPresetValues(node, presetWidget.value)) {
        return;
    }

    node.__srtLastPresetBaseline = presetWidget.value;
    setWidgetValue(presetWidget, "Custom");
}

function applyHeuristicProfile(node, profile) {
    if (!profile || profile === "Custom") {
        return;
    }

    const values = HEURISTIC_PROFILE_VALUES[profile] || HEURISTIC_PROFILE_VALUES["English"];
    node.__applyingHeuristicProfile = true;
    try {
        for (const field of HEURISTIC_FIELDS) {
            const widget = findWidgetByName(node, field);
            if (!widget || values[field] === undefined) {
                continue;
            }
            setWidgetValue(widget, values[field]);
        }
    } finally {
        node.__applyingHeuristicProfile = false;
    }
}

function heuristicProfileHandler(node) {
    if (!isSrtOptionsNode(node)) {
        return;
    }

    const profileWidget = findWidgetByName(node, "heuristic_language_profile");
    if (!profileWidget) {
        return;
    }

    const profile = profileWidget.value;

    if (profile !== "Custom") {
        applyHeuristicProfile(node, profile);
    }
}

function heuristicFieldEdited(node) {
    if (!isSrtOptionsNode(node) || node.__applyingHeuristicProfile) {
        return;
    }
}

app.registerExtension({
    name: "tts-audio-suite.srt-preset.widgets",
    nodeCreated(node) {
        if (!isSrtOptionsNode(node)) {
            return;
        }

        const presetWidget = findWidgetByName(node, "srt_preset");
        node.__srtLastPresetBaseline = presetWidget && presetWidget.value !== "Custom" ? presetWidget.value : null;

        const profileWidget = findWidgetByName(node, "heuristic_language_profile");
        if (profileWidget && profileWidget.value === "Custom") {
            setWidgetValue(profileWidget, "Auto");
        }

        applyPresetState(node);
        heuristicProfileHandler(node);
        applyTtsReadyFieldState(node);

        bindWidgetValueHandler(findWidgetByName(node, "srt_preset"), () => {
            const currentPreset = findWidgetByName(node, "srt_preset")?.value;
            if (currentPreset && currentPreset !== "Custom") {
                node.__srtLastPresetBaseline = currentPreset;
            }
            srtPresetHandler(node);
        });
        bindWidgetValueHandler(findWidgetByName(node, "heuristic_language_profile"), () => heuristicProfileHandler(node));
        bindWidgetValueHandler(findWidgetByName(node, "tts_ready_mode"), () => applyTtsReadyFieldState(node));

        for (const field of SRT_FIELDS) {
            bindWidgetValueHandler(findWidgetByName(node, field), () => srtFieldEdited(node));
        }

        for (const field of HEURISTIC_FIELDS) {
            bindWidgetValueHandler(findWidgetByName(node, field), () => heuristicFieldEdited(node));
        }

        setupSrtAdvancedOptionsPanel(node);
    }
});
