import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const SOURCE = "https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_land.geojson";
const EPSILON = Number(process.argv[2] || 0.7);

function perpendicularDistance(point,start,end) {
    const dx=end[0]-start[0], dy=end[1]-start[1];
    if (!dx && !dy) return Math.hypot(point[0]-start[0],point[1]-start[1]);
    const t=Math.max(0,Math.min(1,((point[0]-start[0])*dx+(point[1]-start[1])*dy)/(dx*dx+dy*dy)));
    return Math.hypot(point[0]-(start[0]+t*dx),point[1]-(start[1]+t*dy));
}

function simplify(points,epsilon) {
    if (points.length<=3) return points;
    let max=0,index=0;
    for (let i=1;i<points.length-1;i++) {
        const distance=perpendicularDistance(points[i],points[0],points.at(-1));
        if (distance>max) { max=distance; index=i; }
    }
    if (max<=epsilon) return [points[0],points.at(-1)];
    return [...simplify(points.slice(0,index+1),epsilon).slice(0,-1),...simplify(points.slice(index),epsilon)];
}

function unwrapLongitudes(points) {
    if (!points.length) return points;
    const result=[points[0].slice()];
    for (let index=1;index<points.length;index++) {
        const previous=result.at(-1)[0];
        const raw=points[index][0];
        const delta=((raw-previous+540)%360)-180;
        result.push([previous+delta,points[index][1]]);
    }
    return result;
}

const input=await new Promise((resolve,reject)=>{
    let value="";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data",chunk=>value+=chunk);
    process.stdin.on("end",()=>resolve(value));
    process.stdin.on("error",reject);
});
const geojson=JSON.parse(input);
const rings=[];
for (const feature of geojson.features) {
    const geometry=feature.geometry;
    const polygons=geometry.type==="Polygon" ? [geometry.coordinates] : geometry.type==="MultiPolygon" ? geometry.coordinates : [];
    for (const polygon of polygons) {
        const exterior=polygon[0];
        const open=exterior.length>1 && exterior[0][0]===exterior.at(-1)[0] && exterior[0][1]===exterior.at(-1)[1] ? exterior.slice(0,-1) : exterior;
        const reduced=simplify(unwrapLongitudes(open),EPSILON).map(([lon,lat])=>[Number(lat.toFixed(2)),Number(lon.toFixed(2))]);
        if (reduced.length>=3) rings.push(reduced);
    }
}
const output=`// Generated from Natural Earth 1:110m land polygons (public domain).\n// Source: ${SOURCE}\n// Simplification tolerance: ${EPSILON} degrees. Regenerate with build-land-data.mjs.\nexport const NATURAL_EARTH_LAND = ${JSON.stringify(rings)};\n`;
const here=path.dirname(fileURLToPath(import.meta.url));
fs.writeFileSync(path.join(here,"natural-earth-land-110m.js"),output,"utf8");
console.log(`Wrote ${rings.length} land polygons with ${rings.reduce((sum,ring)=>sum+ring.length,0)} vertices.`);
