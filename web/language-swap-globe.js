import { NATURAL_EARTH_LAND } from "./prototypes/language-globe-picker/natural-earth-land-110m.js";

const META = {
    af:["Afrikaans","South Africa",-30.56,22.94], ar:["Arabic","Saudi Arabia",24.71,46.68], as:["Assamese","India",26.14,91.74],
    bg:["Bulgarian","Bulgaria",42.70,23.32], bn:["Bengali","Bangladesh",23.81,90.41], cs:["Czech","Czechia",50.08,14.44],
    da:["Danish","Denmark",55.68,12.57], de:["German","Germany",52.52,13.41], el:["Greek","Greece",37.98,23.73],
    en:["English","United Kingdom",51.51,-0.13], es:["Spanish","Spain",40.42,-3.70], et:["Estonian","Estonia",59.44,24.75],
    fa:["Persian","Iran",35.69,51.39], fi:["Finnish","Finland",60.17,24.94], fr:["French","France",48.86,2.35],
    gu:["Gujarati","India",23.02,72.57], he:["Hebrew","Israel",31.77,35.21], hi:["Hindi","India",28.61,77.21],
    hr:["Croatian","Croatia",45.81,15.98], hu:["Hungarian","Hungary",47.50,19.04], id:["Indonesian","Indonesia",-6.21,106.85],
    is:["Icelandic","Iceland",64.15,-21.94], it:["Italian","Italy",41.90,12.50], ja:["Japanese","Japan",35.68,139.69],
    kn:["Kannada","India",12.97,77.59], ko:["Korean","South Korea",37.57,126.98], lt:["Lithuanian","Lithuania",54.69,25.28],
    lv:["Latvian","Latvia",56.95,24.11], ml:["Malayalam","India",8.52,76.94], mr:["Marathi","India",19.08,72.88],
    ms:["Malay","Malaysia",3.14,101.69], nl:["Dutch","Netherlands",52.37,4.90], no:["Norwegian","Norway",59.91,10.75],
    or:["Odia","India",20.30,85.82], pa:["Punjabi","India",31.63,74.87], pl:["Polish","Poland",52.23,21.01],
    pt:["Portuguese","Portugal",38.72,-9.14], "pt-br":["Portuguese","Brazil",-15.79,-47.88], "pt-pt":["Portuguese","Portugal",38.72,-9.14],
    ro:["Romanian","Romania",44.43,26.10], ru:["Russian","Russia",55.76,37.62], sk:["Slovak","Slovakia",48.15,17.11],
    sl:["Slovenian","Slovenia",46.06,14.51], sr:["Serbian","Serbia",44.82,20.46], sv:["Swedish","Sweden",59.33,18.07],
    sw:["Swahili","Tanzania",-6.79,39.21], ta:["Tamil","India",13.08,80.27], te:["Telugu","India",17.39,78.49],
    th:["Thai","Thailand",13.76,100.50], tl:["Tagalog","Philippines",14.60,120.98], tr:["Turkish","Türkiye",39.93,32.86],
    ur:["Urdu","Pakistan",33.69,73.04], vi:["Vietnamese","Vietnam",21.03,105.85], zh:["Chinese","China",39.90,116.41],
    "zh-cn":["Chinese","China",39.90,116.41], "zh-tw":["Chinese","Taiwan",25.03,121.57], zu:["Zulu","South Africa",-29.86,31.02],
};

const LAND = NATURAL_EARTH_LAND;

export function languageMeta(code) {
    const normalized = String(code || "").toLowerCase();
    return META[normalized] || [normalized.toUpperCase(), "", null, null];
}

export function languageTileCode(code) {
    const normalized = String(code || "").toLowerCase();
    if (normalized === "pt-br") return "BR";
    if (normalized === "zh-tw") return "TW";
    return normalized.slice(0, 2).toUpperCase();
}

export function languageEntries() {
    return Object.entries(META).map(([code,[name,country,lat,lon]]) => ({ code,name,country,lat,lon }));
}

export function createLanguageGlobe(host, initialCode, options={}) {
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;opacity:.82";
    host.prepend(canvas);
    const marker = document.createElement("div");
    marker.setAttribute("aria-hidden", "true");
    marker.style.cssText = "position:absolute;z-index:2;width:8px;height:8px;margin:-4px 0 0 -4px;border-radius:50%;pointer-events:none;opacity:0;background:#ffad2f;box-shadow:0 0 0 5px #ffad2f22,0 0 14px #ff9d00;transition:opacity .12s";
    const markerLabel=document.createElement("span");
    markerLabel.style.cssText="position:absolute;top:-7px;padding:3px 6px;white-space:nowrap;color:#ffe1a8;background:rgba(7,12,18,.78);border:1px solid rgba(255,173,47,.34);border-radius:4px;font:600 11px/1.2 system-ui,sans-serif;letter-spacing:.02em;text-shadow:0 1px 2px #000";
    marker.append(markerLabel);
    host.append(marker);
    const ctx = canvas.getContext("2d");
    // Motion is the purpose of this picker, so do not turn the journey into an
    // instant cut when the OS-wide reduced-motion preference is enabled.
    const reduceMotion = false;
    let [, , selectedLat, selectedLon] = languageMeta(initialCode);
    let current = { lat:selectedLat, lon:selectedLon };
    let target = { ...current };
    let animationStart = { ...current };
    let animationStartedAt = 0;
    let animationDuration = 750;
    let durationScale = Number(options.durationScale) || 1;
    if (options.opacity !== undefined) canvas.style.opacity=String(options.opacity);

    const buildLandCells = () => {
        const width=720,height=360,step=3;
        const mask=document.createElement("canvas"); mask.width=width; mask.height=height;
        const maskContext=mask.getContext("2d",{willReadFrequently:true});
        maskContext.fillStyle="#fff";
        for (const polygon of LAND) for (const offset of [-width,0,width]) {
            maskContext.beginPath();
            polygon.forEach(([pointLat,pointLon],index)=>{
                const x=(pointLon+180)/360*width+offset, y=(90-pointLat)/180*height;
                if (index) maskContext.lineTo(x,y); else maskContext.moveTo(x,y);
            });
            maskContext.closePath(); maskContext.fill();
        }
        const pixels=maskContext.getImageData(0,0,width,height).data;
        const cells=[];
        for (let cellLat=-90+step/2;cellLat<90;cellLat+=step) for (let cellLon=-180+step/2;cellLon<180;cellLon+=step) {
            const x=Math.floor((cellLon+180)/360*width), y=Math.floor((90-cellLat)/180*height);
            if (pixels[(y*width+x)*4+3]>0) cells.push([cellLat,cellLon,step/2]);
        }
        return cells;
    };
    const landCells=buildLandCells();
    let frame = 0;

    const projectFactory = (cx, cy, radius) => {
        const lonRotation = -current.lon * Math.PI / 180;
        const latRotation = current.lat * Math.PI / 180;
        return (pointLat, pointLon) => {
            const phi = pointLat * Math.PI / 180;
            const theta = pointLon * Math.PI / 180 + lonRotation;
            const x = Math.cos(phi) * Math.sin(theta);
            const y = Math.sin(phi), z = Math.cos(phi) * Math.cos(theta);
            return { x:cx+radius*x, y:cy-radius*(y*Math.cos(latRotation)-z*Math.sin(latRotation)), z:y*Math.sin(latRotation)+z*Math.cos(latRotation) };
        };
    };

    const draw = () => {
        const box = host.getBoundingClientRect();
        const ratio = Math.min(devicePixelRatio || 1, 2);
        if (canvas.width !== Math.round(box.width*ratio) || canvas.height !== Math.round(box.height*ratio)) {
            canvas.width = Math.round(box.width*ratio); canvas.height = Math.round(box.height*ratio);
            ctx.setTransform(ratio,0,0,ratio,0,0);
        }
        ctx.clearRect(0,0,box.width,box.height);
        const radius = Math.min(box.width*.40, box.height*.43);
        const cx = box.width*.5, cy = box.height*.52;
        const project = projectFactory(cx,cy,radius);
        ctx.save(); ctx.beginPath(); ctx.arc(cx,cy,radius,0,Math.PI*2); ctx.clip();
        ctx.fillStyle="rgba(7,20,35,.58)"; ctx.fillRect(cx-radius,cy-radius,radius*2,radius*2);
        const strokeVisiblePath = points => {
            ctx.beginPath(); let drawing=false;
            for (const point of points) {
                const projected=project(point[0],point[1]);
                if (projected.z <= 0) { drawing=false; continue; }
                if (!drawing) { ctx.moveTo(projected.x,projected.y); drawing=true; }
                else ctx.lineTo(projected.x,projected.y);
            }
            ctx.stroke();
        };
        ctx.strokeStyle="rgba(75,164,255,.20)"; ctx.lineWidth=.8;
        for (let latLine=-60;latLine<=60;latLine+=30) {
            strokeVisiblePath(Array.from({length:73},(_,index)=>[latLine,-180+index*5]));
        }
        for (let lonLine=-180;lonLine<180;lonLine+=30) {
            strokeVisiblePath(Array.from({length:37},(_,index)=>[-90+index*5,lonLine]));
        }
        ctx.fillStyle="rgba(42,121,202,.27)";
        for (const [cellLat,cellLon,half] of landCells) {
            const corners=[[cellLat-half,cellLon-half],[cellLat-half,cellLon+half],[cellLat+half,cellLon+half],[cellLat+half,cellLon-half]].map(([pLat,pLon])=>project(pLat,pLon));
            if (corners.some(point=>point.z<=.015)) continue;
            ctx.beginPath(); corners.forEach((point,index)=>index?ctx.lineTo(point.x,point.y):ctx.moveTo(point.x,point.y)); ctx.closePath(); ctx.fill();
        }
        for (const polygon of LAND) {
            const detailed=[];
            for (let index=0;index<polygon.length;index++) {
                const from=polygon[index], to=polygon[(index+1)%polygon.length];
                detailed.push(from);
                const longitudeDelta=((to[1]-from[1]+540)%360)-180;
                for (let step=1;step<3;step++) detailed.push([from[0]+(to[0]-from[0])*step/3,from[1]+longitudeDelta*step/3]);
            }
            ctx.strokeStyle="rgba(79,169,255,.78)"; ctx.lineWidth=1.15;
            strokeVisiblePath([...detailed,detailed[0]]);
        }
        ctx.restore();
        ctx.beginPath(); ctx.arc(cx,cy,radius,0,Math.PI*2); ctx.strokeStyle="rgba(67,157,250,.5)"; ctx.lineWidth=1.2; ctx.stroke();
        const landmark = project(selectedLat,selectedLon);
        marker.style.left = `${landmark.x}px`;
        marker.style.top = `${landmark.y}px`;
        marker.style.opacity = landmark.z > .05 ? "1" : "0";
        if (landmark.x>cx) { markerLabel.style.right="14px"; markerLabel.style.left="auto"; }
        else { markerLabel.style.left="14px"; markerLabel.style.right="auto"; }
    };

    const chooseSafeCenter = activeButton => {
        const hostRect = host.getBoundingClientRect();
        const activeRect = activeButton?.getBoundingClientRect();
        const buttons = [...host.querySelectorAll("button")].filter(button => button.offsetParent !== null);
        if (!activeRect || buttons.length < 2) return { x:hostRect.width*.5, y:hostRect.height*.52 };
        const rects = buttons.map(button => button.getBoundingClientRect());
        const columns = [...new Set(rects.map(rect => Math.round(rect.left-hostRect.left)))].sort((a,b)=>a-b);
        const rows = [...new Set(rects.map(rect => Math.round(rect.top-hostRect.top)))].sort((a,b)=>a-b);
        const width = rects[0]?.width || 40, height = rects[0]?.height || 38;
        const xs = columns.slice(0,-1).map((left,index) => (left+width+columns[index+1])/2);
        const ys = rows.slice(0,-1).map((top,index) => (top+height+rows[index+1])/2);
        const globeX = hostRect.width*.5, globeY = hostRect.height*.52;
        const globeRadius = Math.min(hostRect.width*.40,hostRect.height*.43);
        let best = { x:globeX, y:globeY, score:-Infinity };
        for (const x of xs) for (const y of ys) {
            const globeDistance = Math.hypot(x-globeX,y-globeY);
            if (globeDistance > globeRadius*.72) continue;
            const activeX=activeRect.left-hostRect.left+activeRect.width/2;
            const activeY=activeRect.top-hostRect.top+activeRect.height/2;
            const distance=Math.hypot(x-activeX,y-activeY);
            const requiredClearance=Math.hypot(activeRect.width,activeRect.height)*.5+22;
            if (distance<requiredClearance) continue;
            // Prefer the closest safe grid gap to the globe center. Moving
            // farther than necessary makes nearby countries feel misplaced.
            const score=-globeDistance;
            if (score > best.score) best = { x,y,score };
        }
        return { x:best.x, y:best.y, globeX, globeY, globeRadius };
    };

    const orientationFor = (lat,lon,point) => {
        const phi = lat*Math.PI/180;
        const screenX = Math.max(-.72,Math.min(.72,(point.x-point.globeX)/point.globeRadius));
        const screenY = Math.max(-.72,Math.min(.72,(point.globeY-point.y)/point.globeRadius));
        const cosPhi = Math.max(.08,Math.cos(phi));
        const theta = Math.asin(Math.max(-.98,Math.min(.98,screenX/cosPhi)));
        const a = Math.sin(phi), b = Math.cos(phi)*Math.cos(theta);
        const amplitude = Math.max(.001,Math.hypot(a,b));
        const phase = Math.atan2(b,a);
        const base = Math.acos(Math.max(-1,Math.min(1,screenY/amplitude)));
        const candidates = [base-phase,-base-phase];
        let rotation = candidates[0];
        for (const candidate of candidates) {
            const depth = a*Math.sin(candidate)+b*Math.cos(candidate);
            if (depth > 0) { rotation=candidate; break; }
        }
        return { lat:rotation*180/Math.PI, lon:lon-theta*180/Math.PI };
    };

    const tick = now => {
        const progress = reduceMotion ? 1 : Math.min(1,(now-animationStartedAt)/animationDuration);
        const eased = progress < .5 ? 4*progress*progress*progress : 1-Math.pow(-2*progress+2,3)/2;
        const lonDelta = ((target.lon-animationStart.lon+540)%360)-180;
        current.lat = animationStart.lat+(target.lat-animationStart.lat)*eased;
        current.lon = animationStart.lon+lonDelta*eased;
        draw();
        if (progress < 1) frame=requestAnimationFrame(tick); else frame=0;
    };
    const observer = new ResizeObserver(draw); observer.observe(host); draw();
    return {
        focus(code, activeButton) {
            const next=languageMeta(code);
            markerLabel.textContent=next[0];
            if (!Number.isFinite(next[2]) || !Number.isFinite(next[3])) {
                marker.style.opacity="0";
                return;
            }
            selectedLat=next[2]; selectedLon=next[3];
            animationStart={...current};
            target=orientationFor(selectedLat,selectedLon,chooseSafeCenter(activeButton));
            const travel=Math.hypot(target.lat-animationStart.lat,((target.lon-animationStart.lon+540)%360)-180);
            animationDuration=Math.max(650,Math.min(1250,520+travel*5))*durationScale;
            animationStartedAt=performance.now();
            if (frame) cancelAnimationFrame(frame);
            frame=0;
            if (!frame) frame=requestAnimationFrame(tick);
        },
        setDurationScale(value) { durationScale=Math.max(.25,Math.min(3,Number(value)||1)); },
        setOpacity(value) { canvas.style.opacity=String(Math.max(.1,Math.min(1,Number(value)||.82))); },
        destroy() { observer.disconnect(); if (frame) cancelAnimationFrame(frame); marker.remove(); },
    };
}
