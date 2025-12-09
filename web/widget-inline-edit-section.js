/**
 * 🏷️ Widget Inline Edit Section
 * Builds the inline edit tag controls for Step Audio EditX post-processing
 * Provides UI for paralinguistic, emotion, style, speed, and restore tags
 */

export function buildInlineEditSection(state, storageKey) {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "8px";
    container.style.overflowY = "auto";
    container.style.flex = "1";
    container.style.paddingRight = "5px";

    // ==================== PARALINGUISTIC TAGS ====================
    const paraSection = document.createElement("div");
    paraSection.style.marginBottom = "8px";
    paraSection.style.paddingBottom = "8px";
    paraSection.style.borderBottom = "1px solid #444";

    const paraLabel = document.createElement("div");
    paraLabel.textContent = "Paralinguistic";
    paraLabel.style.fontWeight = "bold";
    paraLabel.style.marginBottom = "5px";
    paraLabel.style.fontSize = "11px";
    paraLabel.style.color = "#00ffff";

    const paraSelect = document.createElement("select");
    paraSelect.style.width = "100%";
    paraSelect.style.marginBottom = "4px";
    paraSelect.style.padding = "3px";
    paraSelect.style.fontSize = "10px";
    paraSelect.style.background = "#2a2a2a";
    paraSelect.style.color = "#eee";
    paraSelect.style.border = "1px solid #444";
    paraSelect.innerHTML = `
        <option value="">Select sound...</option>
        <option value="Laughter">Laughter</option>
        <option value="Breathing">Breathing</option>
        <option value="Sigh">Sigh</option>
        <option value="Uhm">Uhm</option>
        <option value="Surprise-oh">Surprise (oh)</option>
        <option value="Surprise-ah">Surprise (ah)</option>
        <option value="Surprise-wa">Surprise (wa)</option>
        <option value="Confirmation-en">Confirmation (en)</option>
        <option value="Question-ei">Question (ei)</option>
        <option value="Dissatisfaction-hnn">Dissatisfaction (hnn)</option>
    `;

    // Restore saved selection
    if (state.lastParalinguisticType) {
        paraSelect.value = state.lastParalinguisticType;
    }

    paraSelect.addEventListener("change", () => {
        state.lastParalinguisticType = paraSelect.value;
        state.saveToLocalStorage(storageKey);
    });

    const paraIterLabel = document.createElement("div");
    const paraIterValue = state.lastParalinguisticIter || "1";
    paraIterLabel.textContent = `Iterations: ${paraIterValue}`;
    paraIterLabel.style.fontSize = "9px";
    paraIterLabel.style.marginBottom = "2px";
    paraIterLabel.style.color = "#999";

    const paraIterSlider = document.createElement("input");
    paraIterSlider.type = "range";
    paraIterSlider.min = "1";
    paraIterSlider.max = "5";
    paraIterSlider.value = paraIterValue;
    paraIterSlider.style.width = "100%";
    paraIterSlider.style.marginBottom = "4px";

    paraIterSlider.addEventListener("input", () => {
        paraIterLabel.textContent = `Iterations: ${paraIterSlider.value}`;
        state.lastParalinguisticIter = paraIterSlider.value;
        state.saveToLocalStorage(storageKey);
    });

    const addParaBtn = document.createElement("button");
    addParaBtn.textContent = "Add Paralinguistic";
    addParaBtn.title = "Insert paralinguistic tag at cursor";
    addParaBtn.style.width = "100%";
    addParaBtn.style.padding = "4px";
    addParaBtn.style.cursor = "pointer";
    addParaBtn.style.fontSize = "10px";
    addParaBtn.style.background = "#3a3a3a";
    addParaBtn.style.color = "#eee";
    addParaBtn.style.border = "1px solid #555";
    addParaBtn.style.borderRadius = "2px";

    paraSection.appendChild(paraLabel);
    paraSection.appendChild(paraSelect);
    paraSection.appendChild(paraIterLabel);
    paraSection.appendChild(paraIterSlider);
    paraSection.appendChild(addParaBtn);

    // ==================== EMOTION TAGS ====================
    const emotionSection = document.createElement("div");
    emotionSection.style.marginBottom = "8px";
    emotionSection.style.paddingBottom = "8px";
    emotionSection.style.borderBottom = "1px solid #444";

    const emotionLabel = document.createElement("div");
    emotionLabel.textContent = "Emotion";
    emotionLabel.style.fontWeight = "bold";
    emotionLabel.style.marginBottom = "5px";
    emotionLabel.style.fontSize = "11px";
    emotionLabel.style.color = "#ffaa00";

    const emotionSelect = document.createElement("select");
    emotionSelect.style.width = "100%";
    emotionSelect.style.marginBottom = "4px";
    emotionSelect.style.padding = "3px";
    emotionSelect.style.fontSize = "10px";
    emotionSelect.style.background = "#2a2a2a";
    emotionSelect.style.color = "#eee";
    emotionSelect.style.border = "1px solid #444";
    emotionSelect.innerHTML = `
        <option value="">Select emotion...</option>
        <option value="happy">Happy</option>
        <option value="sad">Sad</option>
        <option value="angry">Angry</option>
        <option value="excited">Excited</option>
        <option value="calm">Calm</option>
        <option value="fearful">Fearful</option>
        <option value="surprised">Surprised</option>
        <option value="disgusted">Disgusted</option>
        <option value="confusion">Confusion</option>
        <option value="empathy">Empathy</option>
        <option value="embarrass">Embarrass</option>
        <option value="depressed">Depressed</option>
        <option value="coldness">Coldness</option>
        <option value="admiration">Admiration</option>
    `;

    // Restore saved selection
    if (state.lastEmotionType) {
        emotionSelect.value = state.lastEmotionType;
    }

    emotionSelect.addEventListener("change", () => {
        state.lastEmotionType = emotionSelect.value;
        state.saveToLocalStorage(storageKey);
    });

    const emotionIterLabel = document.createElement("div");
    const emotionIterValue = state.lastEmotionIter || "1";
    emotionIterLabel.textContent = `Iterations: ${emotionIterValue}`;
    emotionIterLabel.style.fontSize = "9px";
    emotionIterLabel.style.marginBottom = "2px";
    emotionIterLabel.style.color = "#999";

    const emotionIterSlider = document.createElement("input");
    emotionIterSlider.type = "range";
    emotionIterSlider.min = "1";
    emotionIterSlider.max = "5";
    emotionIterSlider.value = emotionIterValue;
    emotionIterSlider.style.width = "100%";
    emotionIterSlider.style.marginBottom = "4px";

    emotionIterSlider.addEventListener("input", () => {
        emotionIterLabel.textContent = `Iterations: ${emotionIterSlider.value}`;
        state.lastEmotionIter = emotionIterSlider.value;
        state.saveToLocalStorage(storageKey);
    });

    const addEmotionBtn = document.createElement("button");
    addEmotionBtn.textContent = "Add Emotion";
    addEmotionBtn.title = "Insert emotion tag at cursor";
    addEmotionBtn.style.width = "100%";
    addEmotionBtn.style.padding = "4px";
    addEmotionBtn.style.cursor = "pointer";
    addEmotionBtn.style.fontSize = "10px";
    addEmotionBtn.style.background = "#3a3a3a";
    addEmotionBtn.style.color = "#eee";
    addEmotionBtn.style.border = "1px solid #555";
    addEmotionBtn.style.borderRadius = "2px";

    emotionSection.appendChild(emotionLabel);
    emotionSection.appendChild(emotionSelect);
    emotionSection.appendChild(emotionIterLabel);
    emotionSection.appendChild(emotionIterSlider);
    emotionSection.appendChild(addEmotionBtn);

    // ==================== STYLE TAGS ====================
    const styleSection = document.createElement("div");
    styleSection.style.marginBottom = "8px";
    styleSection.style.paddingBottom = "8px";
    styleSection.style.borderBottom = "1px solid #444";

    const styleLabel = document.createElement("div");
    styleLabel.textContent = "Style";
    styleLabel.style.fontWeight = "bold";
    styleLabel.style.marginBottom = "5px";
    styleLabel.style.fontSize = "11px";
    styleLabel.style.color = "#ff5555";

    const styleSelect = document.createElement("select");
    styleSelect.style.width = "100%";
    styleSelect.style.marginBottom = "4px";
    styleSelect.style.padding = "3px";
    styleSelect.style.fontSize = "10px";
    styleSelect.style.background = "#2a2a2a";
    styleSelect.style.color = "#eee";
    styleSelect.style.border = "1px solid #444";
    styleSelect.innerHTML = `
        <option value="">Select style...</option>
        <option value="whisper">Whisper</option>
        <option value="serious">Serious</option>
        <option value="child">Child</option>
        <option value="older">Older</option>
        <option value="girl">Girl</option>
        <option value="pure">Pure</option>
        <option value="sister">Sister</option>
        <option value="sweet">Sweet</option>
        <option value="exaggerated">Exaggerated</option>
        <option value="ethereal">Ethereal</option>
        <option value="generous">Generous</option>
        <option value="recite">Recite</option>
        <option value="act_coy">Act Coy</option>
        <option value="warm">Warm</option>
        <option value="shy">Shy</option>
        <option value="comfort">Comfort</option>
        <option value="authority">Authority</option>
        <option value="chat">Chat</option>
        <option value="radio">Radio</option>
        <option value="soulful">Soulful</option>
        <option value="gentle">Gentle</option>
        <option value="story">Story</option>
        <option value="vivid">Vivid</option>
        <option value="program">Program</option>
        <option value="news">News</option>
        <option value="advertising">Advertising</option>
        <option value="roar">Roar</option>
        <option value="murmur">Murmur</option>
        <option value="shout">Shout</option>
        <option value="deeply">Deeply</option>
        <option value="loudly">Loudly</option>
        <option value="arrogant">Arrogant</option>
        <option value="friendly">Friendly</option>
    `;

    // Restore saved selection
    if (state.lastStyleType) {
        styleSelect.value = state.lastStyleType;
    }

    styleSelect.addEventListener("change", () => {
        state.lastStyleType = styleSelect.value;
        state.saveToLocalStorage(storageKey);
    });

    const styleIterLabel = document.createElement("div");
    const styleIterValue = state.lastStyleIter || "1";
    styleIterLabel.textContent = `Iterations: ${styleIterValue}`;
    styleIterLabel.style.fontSize = "9px";
    styleIterLabel.style.marginBottom = "2px";
    styleIterLabel.style.color = "#999";

    const styleIterSlider = document.createElement("input");
    styleIterSlider.type = "range";
    styleIterSlider.min = "1";
    styleIterSlider.max = "5";
    styleIterSlider.value = styleIterValue;
    styleIterSlider.style.width = "100%";
    styleIterSlider.style.marginBottom = "4px";

    styleIterSlider.addEventListener("input", () => {
        styleIterLabel.textContent = `Iterations: ${styleIterSlider.value}`;
        state.lastStyleIter = styleIterSlider.value;
        state.saveToLocalStorage(storageKey);
    });

    const addStyleBtn = document.createElement("button");
    addStyleBtn.textContent = "Add Style";
    addStyleBtn.title = "Insert style tag at cursor";
    addStyleBtn.style.width = "100%";
    addStyleBtn.style.padding = "4px";
    addStyleBtn.style.cursor = "pointer";
    addStyleBtn.style.fontSize = "10px";
    addStyleBtn.style.background = "#3a3a3a";
    addStyleBtn.style.color = "#eee";
    addStyleBtn.style.border = "1px solid #555";
    addStyleBtn.style.borderRadius = "2px";

    styleSection.appendChild(styleLabel);
    styleSection.appendChild(styleSelect);
    styleSection.appendChild(styleIterLabel);
    styleSection.appendChild(styleIterSlider);
    styleSection.appendChild(addStyleBtn);

    // ==================== SPEED TAGS ====================
    const speedSection = document.createElement("div");
    speedSection.style.marginBottom = "8px";
    speedSection.style.paddingBottom = "8px";
    speedSection.style.borderBottom = "1px solid #444";

    const speedLabel = document.createElement("div");
    speedLabel.textContent = "Speed";
    speedLabel.style.fontWeight = "bold";
    speedLabel.style.marginBottom = "5px";
    speedLabel.style.fontSize = "11px";
    speedLabel.style.color = "#66ff66";

    const speedSelect = document.createElement("select");
    speedSelect.style.width = "100%";
    speedSelect.style.marginBottom = "4px";
    speedSelect.style.padding = "3px";
    speedSelect.style.fontSize = "10px";
    speedSelect.style.background = "#2a2a2a";
    speedSelect.style.color = "#eee";
    speedSelect.style.border = "1px solid #444";
    speedSelect.innerHTML = `
        <option value="">Select speed...</option>
        <option value="faster">Faster</option>
        <option value="slower">Slower</option>
        <option value="more_faster">More Faster</option>
        <option value="more_slower">More Slower</option>
    `;

    // Restore saved selection
    if (state.lastSpeedType) {
        speedSelect.value = state.lastSpeedType;
    }

    speedSelect.addEventListener("change", () => {
        state.lastSpeedType = speedSelect.value;
        state.saveToLocalStorage(storageKey);
    });

    const speedIterLabel = document.createElement("div");
    const speedIterValue = state.lastSpeedIter || "1";
    speedIterLabel.textContent = `Iterations: ${speedIterValue}`;
    speedIterLabel.style.fontSize = "9px";
    speedIterLabel.style.marginBottom = "2px";
    speedIterLabel.style.color = "#999";

    const speedIterSlider = document.createElement("input");
    speedIterSlider.type = "range";
    speedIterSlider.min = "1";
    speedIterSlider.max = "5";
    speedIterSlider.value = speedIterValue;
    speedIterSlider.style.width = "100%";
    speedIterSlider.style.marginBottom = "4px";

    speedIterSlider.addEventListener("input", () => {
        speedIterLabel.textContent = `Iterations: ${speedIterSlider.value}`;
        state.lastSpeedIter = speedIterSlider.value;
        state.saveToLocalStorage(storageKey);
    });

    const addSpeedBtn = document.createElement("button");
    addSpeedBtn.textContent = "Add Speed";
    addSpeedBtn.title = "Insert speed tag at cursor";
    addSpeedBtn.style.width = "100%";
    addSpeedBtn.style.padding = "4px";
    addSpeedBtn.style.cursor = "pointer";
    addSpeedBtn.style.fontSize = "10px";
    addSpeedBtn.style.background = "#3a3a3a";
    addSpeedBtn.style.color = "#eee";
    addSpeedBtn.style.border = "1px solid #555";
    addSpeedBtn.style.borderRadius = "2px";

    speedSection.appendChild(speedLabel);
    speedSection.appendChild(speedSelect);
    speedSection.appendChild(speedIterLabel);
    speedSection.appendChild(speedIterSlider);
    speedSection.appendChild(addSpeedBtn);

    // ==================== RESTORE TAGS ====================
    const restoreSection = document.createElement("div");
    restoreSection.style.marginBottom = "8px";

    const restoreLabel = document.createElement("div");
    restoreLabel.textContent = "Voice Restoration";
    restoreLabel.style.fontWeight = "bold";
    restoreLabel.style.marginBottom = "5px";
    restoreLabel.style.fontSize = "11px";
    restoreLabel.style.color = "#ffcc33";

    const restorePassLabel = document.createElement("div");
    const restorePassValue = state.lastRestorePasses || "1";
    restorePassLabel.textContent = `VC Passes: ${restorePassValue}`;
    restorePassLabel.style.fontSize = "9px";
    restorePassLabel.style.marginBottom = "2px";
    restorePassLabel.style.color = "#999";

    const restorePassSlider = document.createElement("input");
    restorePassSlider.type = "range";
    restorePassSlider.min = "1";
    restorePassSlider.max = "5";
    restorePassSlider.value = restorePassValue;
    restorePassSlider.style.width = "100%";
    restorePassSlider.style.marginBottom = "4px";

    restorePassSlider.addEventListener("input", () => {
        restorePassLabel.textContent = `VC Passes: ${restorePassSlider.value}`;
        state.lastRestorePasses = restorePassSlider.value;
        state.saveToLocalStorage(storageKey);
    });

    const restoreRefLabel = document.createElement("div");
    restoreRefLabel.textContent = "Reference Iteration (optional)";
    restoreRefLabel.style.fontSize = "9px";
    restoreRefLabel.style.marginBottom = "2px";
    restoreRefLabel.style.color = "#999";

    const restoreRefInput = document.createElement("input");
    restoreRefInput.type = "number";
    restoreRefInput.placeholder = "Leave empty for original";
    restoreRefInput.min = "1";
    restoreRefInput.max = "10";
    restoreRefInput.value = state.lastRestoreRefIter || "";
    restoreRefInput.style.width = "100%";
    restoreRefInput.style.padding = "3px";
    restoreRefInput.style.fontSize = "10px";
    restoreRefInput.style.background = "#2a2a2a";
    restoreRefInput.style.color = "#eee";
    restoreRefInput.style.border = "1px solid #444";
    restoreRefInput.style.marginBottom = "4px";

    restoreRefInput.addEventListener("change", () => {
        state.lastRestoreRefIter = restoreRefInput.value;
        state.saveToLocalStorage(storageKey);
    });

    const addRestoreBtn = document.createElement("button");
    addRestoreBtn.textContent = "Add Restore";
    addRestoreBtn.title = "Insert voice restoration tag at cursor";
    addRestoreBtn.style.width = "100%";
    addRestoreBtn.style.padding = "4px";
    addRestoreBtn.style.cursor = "pointer";
    addRestoreBtn.style.fontSize = "10px";
    addRestoreBtn.style.background = "#3a3a3a";
    addRestoreBtn.style.color = "#eee";
    addRestoreBtn.style.border = "1px solid #555";
    addRestoreBtn.style.borderRadius = "2px";

    restoreSection.appendChild(restoreLabel);
    restoreSection.appendChild(restorePassLabel);
    restoreSection.appendChild(restorePassSlider);
    restoreSection.appendChild(restoreRefLabel);
    restoreSection.appendChild(restoreRefInput);
    restoreSection.appendChild(addRestoreBtn);

    // Assemble all sections
    container.appendChild(paraSection);
    container.appendChild(emotionSection);
    container.appendChild(styleSection);
    container.appendChild(speedSection);
    container.appendChild(restoreSection);

    return {
        inlineEditSection: container,
        paraSelect,
        paraIterSlider,
        addParaBtn,
        emotionSelect,
        emotionIterSlider,
        addEmotionBtn,
        styleSelect,
        styleIterSlider,
        addStyleBtn,
        speedSelect,
        speedIterSlider,
        addSpeedBtn,
        restorePassSlider,
        restoreRefInput,
        addRestoreBtn
    };
}
