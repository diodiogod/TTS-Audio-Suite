import { createLanguageGlobe, languageEntries, languageTileCode } from "../../language-swap-globe.js";

const picker=document.querySelector("#picker");
const grid=document.querySelector("#language-grid");
const status=document.querySelector("#status");
let savedPickerSize=null;
try{savedPickerSize=JSON.parse(localStorage.getItem("tts-language-globe-prototype-size")||"null");}catch{/* use default */}
if(savedPickerSize){
    const size=Math.max(300,Math.min(savedPickerSize.width||720,savedPickerSize.height||720));
    picker.style.width=`${size}px`;
    picker.style.height=`${size}px`;
}
const resizeHandle=document.createElement("div");
resizeHandle.className="resize-handle";
resizeHandle.setAttribute("aria-label","Resize language picker");
picker.append(resizeHandle);
let resizeDrag=null;
resizeHandle.addEventListener("pointerdown",event=>{
    event.preventDefault(); event.stopPropagation();
    resizeDrag={x:event.clientX,y:event.clientY,size:picker.getBoundingClientRect().width};
    resizeHandle.setPointerCapture(event.pointerId);
});
resizeHandle.addEventListener("pointermove",event=>{
    if(!resizeDrag||!resizeHandle.hasPointerCapture(event.pointerId))return;
    event.preventDefault(); event.stopPropagation();
    const dx=event.clientX-resizeDrag.x,dy=event.clientY-resizeDrag.y;
    const delta=Math.abs(dx)>Math.abs(dy)?dx:dy;
    const box=picker.getBoundingClientRect();
    const maxSize=Math.max(300,Math.min(innerWidth-box.left-8,innerHeight-box.top-8));
    const size=Math.max(300,Math.min(maxSize,resizeDrag.size+delta));
    picker.style.width=`${size}px`;picker.style.height=`${size}px`;
});
const finishResize=event=>{
    if(!resizeDrag)return;
    event.preventDefault();event.stopPropagation();
    const size=Math.round(picker.getBoundingClientRect().width);
    localStorage.setItem("tts-language-globe-prototype-size",JSON.stringify({width:size,height:size}));
    resizeDrag=null;
    if(resizeHandle.hasPointerCapture(event.pointerId))resizeHandle.releasePointerCapture(event.pointerId);
};
resizeHandle.addEventListener("pointerup",finishResize);
resizeHandle.addEventListener("pointercancel",finishResize);
const entries=languageEntries()
    .filter(entry=>!["pt-pt","zh-cn"].includes(entry.code))
    .sort((a,b)=>languageTileCode(a.code).localeCompare(languageTileCode(b.code)));
const globe=createLanguageGlobe(picker,"en");
const buttons=new Map();
let selectedCode="";
let page=0;

function visit(entry,button) {
    globe.focus(entry.code,button);
    status.textContent=`${entry.name} · ${entry.country} · ${entry.code}`;
}

function visibleEntries() {
    if (!document.querySelector("#paged").checked) return entries;
    return entries.filter(entry=>page===0 ? languageTileCode(entry.code)<="M" : languageTileCode(entry.code)>"M");
}

function renderGrid() {
    grid.replaceChildren(); buttons.clear();
    for (const entry of visibleEntries()) {
        const button=document.createElement("button");
        button.type="button";
        button.textContent=languageTileCode(entry.code);
        button.setAttribute("aria-label",`${entry.name}, ${entry.country}`);
        button.classList.toggle("is-selected",entry.code===selectedCode);
        button.addEventListener("pointerenter",()=>visit(entry,button));
        button.addEventListener("focus",()=>visit(entry,button));
        button.addEventListener("click",()=>{
            selectedCode=entry.code;
            grid.querySelector(".is-selected")?.classList.remove("is-selected");
            button.classList.add("is-selected");
        });
        grid.append(button); buttons.set(entry.code,button);
    }
}
renderGrid();
grid.addEventListener("pointerleave",()=>{
    const fallbackCode=selectedCode||"en";
    const entry=entries.find(item=>item.code===fallbackCode);
    const button=buttons.get(fallbackCode);
    if (entry&&button) visit(entry,button);
});

const speed=document.querySelector("#speed");
speed.addEventListener("input",()=>{
    // Higher UI speed means a shorter duration.
    globe.setDurationScale(1/Number(speed.value));
    document.querySelector("#speed-value").value=`${Number(speed.value).toFixed(1)}×`;
});
const opacity=document.querySelector("#opacity");
opacity.addEventListener("input",()=>{
    globe.setOpacity(opacity.value);
    document.querySelector("#opacity-value").value=`${Math.round(Number(opacity.value)*100)}%`;
});
const paged=document.querySelector("#paged"),pager=document.querySelector("#pager"),pageLabel=document.querySelector("#page-label");
function updatePage() {
    const enabled=paged.checked;
    grid.classList.toggle("is-paged",enabled);
    pager.hidden=!enabled;
    pageLabel.textContent=`${page===0?"A–M":"N–Z"} · ${page+1} / 2`;
    document.querySelector("#paged-value").value=enabled?"On":"Off";
    renderGrid();
}
paged.addEventListener("change",()=>{page=0;updatePage();});
document.querySelector("#previous-page").addEventListener("click",()=>{page=page?0:1;updatePage();});
document.querySelector("#next-page").addEventListener("click",()=>{page=page?0:1;updatePage();});

document.querySelector("#tour").addEventListener("click",async()=>{
    for (const code of ["en","pt-br","ja","ar","fr","zu"]) {
        const entry=entries.find(item=>item.code===code);
        if (paged.checked) { page=languageTileCode(code)<="M"?0:1; updatePage(); }
        const button=buttons.get(code);
        visit(entry,button); button.focus();
        await new Promise(resolve=>setTimeout(resolve,1500));
    }
});

requestAnimationFrame(()=>visit(entries.find(entry=>entry.code==="en"),buttons.get("en")));
