import { app } from "../../scripts/app.js";

const TARGET_CLASSES = new Set(["RVCDatasetPrepNode"]);
const SUPPORTED_AUDIO_EXTENSIONS = new Set([".wav", ".flac", ".mp3"]);
const SUPPORTED_DATASET_EXTENSIONS = new Set([".zip", ...SUPPORTED_AUDIO_EXTENSIONS]);
const AUDIO_INPUT_PREFIX = "opt_audio";
const AUDIO_INPUT_TYPE = "AUDIO";

function isTargetNode(nodeOrData) {
    const comfyClass = nodeOrData?.comfyClass || nodeOrData?.name;
    return TARGET_CLASSES.has(comfyClass);
}

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
}

function getExtension(filename) {
    const dotIndex = filename.lastIndexOf(".");
    return dotIndex >= 0 ? filename.slice(dotIndex).toLowerCase() : "";
}

function sanitizePathSegment(value) {
    const cleaned = String(value || "")
        .trim()
        .replace(/[\\/:*?"<>|]+/g, "_")
        .replace(/\s+/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "");
    return cleaned || `dataset_${Date.now()}`;
}

function setDatasetSource(node, value) {
    const datasetWidget = findWidgetByName(node, "dataset_source");
    if (!datasetWidget) {
        return;
    }
    datasetWidget.value = value;
    if (datasetWidget.callback) {
        datasetWidget.callback(value);
    }
    app.graph.setDirtyCanvas(true);
}

function isDynamicAudioInput(input) {
    return Boolean(input?.name && String(input.name).startsWith(AUDIO_INPUT_PREFIX));
}

function getDynamicAudioInputs(node) {
    return (node.inputs || []).filter(isDynamicAudioInput);
}

function renumberDynamicAudioInputs(node) {
    let nextIndex = 1;
    for (const input of node.inputs || []) {
        if (!isDynamicAudioInput(input)) {
            continue;
        }
        input.name = `${AUDIO_INPUT_PREFIX}${nextIndex}`;
        input.label = input.name;
        nextIndex += 1;
    }
}

function ensureTrailingDynamicAudioInput(node) {
    const audioInputs = getDynamicAudioInputs(node);
    if (!audioInputs.length || audioInputs[audioInputs.length - 1].link != null) {
        const newIndex = audioInputs.length + 1;
        node.addInput(`${AUDIO_INPUT_PREFIX}${newIndex}`, AUDIO_INPUT_TYPE);
    }
}

function syncDynamicAudioInputs(node) {
    if (!node || node.__ttsRvcSyncingAudioInputs) {
        return;
    }

    node.__ttsRvcSyncingAudioInputs = true;
    try {
        for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
            const input = node.inputs[index];
            if (isDynamicAudioInput(input) && input.link == null) {
                node.removeInput(index);
            }
        }

        renumberDynamicAudioInputs(node);
        ensureTrailingDynamicAudioInput(node);
        renumberDynamicAudioInputs(node);

        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            node.setSize(node.computeSize());
        }
        app.graph.setDirtyCanvas(true, true);
    } finally {
        node.__ttsRvcSyncingAudioInputs = false;
    }
}

async function uploadFileToComfyUI(file, subfolder) {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("type", "input");
    formData.append("subfolder", subfolder);

    const response = await fetch("/upload/image", {
        method: "POST",
        body: formData,
    });
    if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
}

function buildDatasetFolderName(node, preferredName = "") {
    const modelNameWidget = findWidgetByName(node, "model_name");
    const modelName = modelNameWidget?.value || "";
    const baseName = sanitizePathSegment(preferredName || modelName || "dataset");
    return `${baseName}_${Date.now()}`;
}

function filterSupportedFiles(fileList, allowZip = true) {
    const allowed = allowZip ? SUPPORTED_DATASET_EXTENSIONS : SUPPORTED_AUDIO_EXTENSIONS;
    return Array.from(fileList || []).filter((file) => allowed.has(getExtension(file.name)));
}

async function uploadZipOrFiles(node, fileButton, folderButton) {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".zip,.wav,.flac,.mp3";

    input.onchange = async () => {
        try {
            const files = filterSupportedFiles(input.files, true);
            if (!files.length) {
                return;
            }

            fileButton.name = "Uploading Dataset...";
            if (folderButton) {
                folderButton.name = "Upload Folder";
            }
            app.graph.setDirtyCanvas(true);

            if (files.length === 1 && getExtension(files[0].name) === ".zip") {
                const datasetFolder = buildDatasetFolderName(node, files[0].name.replace(/\.zip$/i, ""));
                const result = await uploadFileToComfyUI(files[0], `datasets/tts_audio_suite/${datasetFolder}`);
                setDatasetSource(node, `tts_audio_suite/${datasetFolder}/${result.name}`);
            } else {
                const datasetFolder = buildDatasetFolderName(node);
                const targetSubfolder = `datasets/tts_audio_suite/${datasetFolder}`;
                for (const file of files) {
                    await uploadFileToComfyUI(file, targetSubfolder);
                }
                setDatasetSource(node, `tts_audio_suite/${datasetFolder}`);
            }
        } catch (error) {
            console.error("RVC Dataset Prep upload failed:", error);
            fileButton.name = "Upload Failed";
            app.graph.setDirtyCanvas(true);
            setTimeout(() => {
                fileButton.name = "Upload Files / Zip";
                app.graph.setDirtyCanvas(true);
            }, 1500);
        } finally {
            fileButton.name = "Upload Files / Zip";
            app.graph.setDirtyCanvas(true);
            document.body.removeChild(input);
        }
    };

    document.body.appendChild(input);
    input.click();
}

async function uploadFolder(node, fileButton, folderButton) {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.webkitdirectory = true;

    input.onchange = async () => {
        try {
            const files = filterSupportedFiles(input.files, false);
            if (!files.length) {
                return;
            }

            const firstRelativePath = files[0].webkitRelativePath || "";
            const rootFolderName = firstRelativePath ? firstRelativePath.split("/")[0] : "";
            const datasetFolder = buildDatasetFolderName(node, rootFolderName);

            folderButton.name = "Uploading Folder...";
            if (fileButton) {
                fileButton.name = "Upload Files / Zip";
            }
            app.graph.setDirtyCanvas(true);

            for (const file of files) {
                const relativePath = file.webkitRelativePath || file.name;
                const parts = relativePath.split("/").filter(Boolean);
                const nestedDir = parts.slice(1, -1).join("/");
                const subfolder = nestedDir
                    ? `datasets/tts_audio_suite/${datasetFolder}/${nestedDir}`
                    : `datasets/tts_audio_suite/${datasetFolder}`;
                await uploadFileToComfyUI(file, subfolder);
            }

            setDatasetSource(node, `tts_audio_suite/${datasetFolder}`);
        } catch (error) {
            console.error("RVC Dataset Prep folder upload failed:", error);
            folderButton.name = "Upload Failed";
            app.graph.setDirtyCanvas(true);
            setTimeout(() => {
                folderButton.name = "Upload Folder";
                app.graph.setDirtyCanvas(true);
            }, 1500);
        } finally {
            folderButton.name = "Upload Folder";
            app.graph.setDirtyCanvas(true);
            document.body.removeChild(input);
        }
    };

    document.body.appendChild(input);
    input.click();
}

function ensureDatasetUploadWidgets(node) {
    if (node.__ttsRvcDatasetUploadWidgetsInitialized) {
        return;
    }
    node.__ttsRvcDatasetUploadWidgetsInitialized = true;

    const datasetWidget = findWidgetByName(node, "dataset_source");
    if (!datasetWidget) {
        return;
    }

    const fileButton = node.addWidget(
        "button",
        "Upload Files / Zip",
        "",
        () => uploadZipOrFiles(node, fileButton, folderButton),
        { serialize: false }
    );

    const folderButton = node.addWidget(
        "button",
        "Upload Folder",
        "",
        () => uploadFolder(node, fileButton, folderButton),
        { serialize: false }
    );

    node.__ttsRvcDatasetUploadFileButton = fileButton;
    node.__ttsRvcDatasetUploadFolderButton = folderButton;
}

app.registerExtension({
    name: "TTS_Audio_Suite.RVCDatasetPrepUpload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNode(nodeData)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setTimeout(() => {
                ensureDatasetUploadWidgets(this);
                syncDynamicAudioInputs(this);
                app.graph.setDirtyCanvas(true);
            }, 100);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo) {
            const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            const input = this.inputs?.[index];
            if (type === 2 || !isDynamicAudioInput(input)) {
                return result;
            }

            const stackTrace = new Error().stack || "";
            if (stackTrace.includes("loadGraphData") || stackTrace.includes("pasteFromClipboard")) {
                setTimeout(() => syncDynamicAudioInputs(this), 0);
                return result;
            }

            if (!linkInfo && !connected) {
                setTimeout(() => syncDynamicAudioInputs(this), 0);
                return result;
            }

            syncDynamicAudioInputs(this);
            return result;
        };
    },
});
