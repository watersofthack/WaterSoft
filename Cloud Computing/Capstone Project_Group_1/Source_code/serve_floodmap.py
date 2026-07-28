#!/usr/bin/env python3
"""
Multi-source flood-map API (FloodOps cloud).

Serves the real flood map retrieved from several official sources and highlights
where the sources DIFFER. No synthetic reports -- just the retrieved data.

Endpoints (port 8000):
  GET /flood_map.geojson   the full combined map (all sources), as GeoJSON
  GET /sources             a JSON summary: how many features per source, and a
                           simple "discrepancy" count where sources disagree
  GET /                    same as /sources (quick human-readable check)

Sources tagged in properties.source:
  nws_fim     NWS-FIM real inundation extent (polygon)
  nws_alert   NWS flood-warning area (polygon)
  fema_nfhl   FEMA floodplain / flood-prone zone (polygon)
  nwps_gauge  river gauge reading flood (point)

Stdlib only.  Run:  python3 serve_floodmap.py   (binds 0.0.0.0:8000)
"""

import json
import os
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
GEOJSON = os.path.join(HERE, "flood_map.geojson")
REFRESH_STATUS = os.path.join(HERE, "refresh_status.json")
SOURCE_KEYS = ("nws_fim", "nws_alert", "fema_nfhl", "nwps_gauge")
APP_VERSION = os.environ.get("APP_VERSION", "v1.0.12")


def load_map():
    try:
        with open(GEOJSON, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"type": "FeatureCollection", "features": []}

def iso_from_mtime(path):
    try:
        return datetime.fromtimestamp(
            os.path.getmtime(path), timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        return None


def load_refresh_status():
    """Return refresh timing, with GeoJSON metadata/mtime as a safe fallback."""
    fc = load_map()
    retrieved_at = fc.get("retrieved_at") or iso_from_mtime(GEOJSON)
    sources = fc.get("source_retrieved_at") or {}
    status = {}
    try:
        with open(REFRESH_STATUS, encoding="utf-8") as f:
            status = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    recorded_sources = status.get("sources") or {}
    status["sources"] = {
        key: recorded_sources.get(key) or sources.get(key) or retrieved_at
        for key in SOURCE_KEYS
    }
    status.setdefault("last_success_at", retrieved_at)
    status.setdefault("refresh_interval_seconds",
                      int(os.environ.get("REFRESH_SECONDS", "3600")))
    status.setdefault("status", "waiting")

    # Older refresh_status.json files may not contain next_refresh_at. Derive it
    # from the last successful retrieval so the map can still show a countdown.
    if not status.get("next_refresh_at") and status.get("last_success_at"):
        try:
            last_success = datetime.fromisoformat(
                status["last_success_at"].replace("Z", "+00:00")
            )
            interval = int(status["refresh_interval_seconds"])
            status["next_refresh_at"] = (
                last_success + timedelta(seconds=interval)
            ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError, OverflowError):
            pass
    return status


def point_in_ring(lat, lon, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def rings_of(geom):
    t = geom.get("type")
    if t == "Polygon":
        return list(geom["coordinates"])
    if t == "MultiPolygon":
        return [ring for poly in geom["coordinates"] for ring in poly]
    return []


def summarize(fc):
    """Counts per source + a simple cross-source discrepancy metric."""
    feats = fc.get("features", [])
    by_source = {}
    poly_rings = []          # all flood-AREA rings (fim/alert/fema)
    gauges = []              # gauge points that read flood
    for f in feats:
        p = f.get("properties") or {}
        src = p.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        geom = f.get("geometry") or {}
        if src in ("nws_fim", "nws_alert", "fema_nfhl"):
            poly_rings.extend(rings_of(geom))
        elif src == "nwps_gauge" and geom.get("type") == "Point":
            lon, lat = geom["coordinates"][:2]
            gauges.append((lat, lon))

    # Discrepancy: gauges reading flood that fall OUTSIDE every mapped flood area.
    # (A gauge in flood with no polygon around it = the sources disagree there.)
    outside = 0
    for lat, lon in gauges:
        if not any(point_in_ring(lat, lon, ring) for ring in poly_rings):
            outside += 1

    return {
        "total_features": len(feats),
        "features_by_source": by_source,
        "gauges_in_flood": len(gauges),
        "gauges_outside_any_flood_polygon": outside,
        "note": ("Gauges reading flood but sitting outside every NWS-FIM / "
                 "NWS-warning / FEMA polygon are places where the official "
                 "sources disagree."),
    }


# Interactive Leaflet map. It fetches /flood_map.geojson live, so it always shows
# whatever the service currently serves (no rebuild needed to refresh the view).
MAP_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FloodOps - live multi-source flood map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<style>
  html,body,#map{height:100%;margin:0}
  .legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:13px sans-serif;line-height:1.7}
  .legend .sw{display:inline-block;width:13px;height:13px;margin-right:6px;vertical-align:middle;border:1px solid rgba(0,0,0,.35)}
  .legend .poly{border-radius:2px;opacity:.7}
  .legend .pt{border-radius:50%}
  .map-title{position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:1001;margin:0;background:rgba(255,255,255,.96);padding:9px 22px;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.3);font:700 25px/1.2 sans-serif;color:#12344d;text-align:center;white-space:nowrap}
  .banner{position:absolute;top:62px;left:55px;right:10px;z-index:1000;background:#fff;padding:7px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:13px sans-serif}
  .leaflet-control-layers{font:13px sans-serif}
  .leaflet-control-home a{font:700 20px/30px sans-serif;color:#173b57;text-align:center;text-decoration:none}
  .leaflet-control-home a:hover{background:#eef6fb;color:#087f8c}
  .refresh-box{position:absolute;top:107px;right:10px;z-index:1000;min-width:260px;background:#fff;padding:9px 11px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:12px sans-serif;line-height:1.45}
  .refresh-box table{border-collapse:collapse;width:100%;margin-top:4px}
  .refresh-box td{padding:1px 0}
  .refresh-box td:last-child{text-align:right;padding-left:12px;color:#444}
  .refresh-box .countdown{font-weight:700;color:#0b7285}
  .refresh-box .waiting{font-weight:700;color:#1976d2}
  .refresh-box .ok{font-weight:700;color:#238636}
  .refresh-box .error{color:#b42318}
  .version-badge{position:absolute;bottom:22px;left:50%;transform:translateX(-50%);z-index:999;background:rgba(255,255,255,.94);padding:5px 9px;border-radius:5px;box-shadow:0 1px 4px rgba(0,0,0,.25);font:600 11px sans-serif;color:#345}
  .feature-popup{font:12px/1.45 sans-serif;min-width:245px}
  .feature-popup table{border-collapse:collapse;width:100%;margin-top:5px}
  .feature-popup td{padding:2px 4px;border-bottom:1px solid #eee;vertical-align:top}
  .feature-popup td:first-child{font-weight:600;color:#345;width:42%}
  .feature-popup .hint{margin-top:6px;color:#667;font-size:11px}
  .flood-flag-wrap{background:transparent;border:0}
  .flood-flag{position:relative;width:30px;height:43px;filter:drop-shadow(2px 4px 2px rgba(0,0,0,.38));transform-origin:50% 100%;transition:transform .16s ease}
  .flood-flag:hover{transform:translateY(-3px) scale(1.12)}
  .flood-flag .pole{position:absolute;left:6px;top:2px;width:3px;height:37px;background:linear-gradient(90deg,#fff,#697884);border-radius:2px;box-shadow:0 0 0 1px rgba(22,42,58,.55)}
  .flood-flag .cloth{position:absolute;left:9px;top:3px;width:20px;height:14px;background:var(--flag-color);border:2px solid #fff;border-left-width:1px;clip-path:polygon(0 0,100% 0,78% 50%,100% 100%,0 100%);box-shadow:0 1px 2px rgba(0,0,0,.35)}
  .flood-flag .foot{position:absolute;left:2px;bottom:1px;width:12px;height:5px;background:#263f52;border:1px solid #fff;border-radius:50%}
  .marker-cluster-small div,.marker-cluster-medium div,.marker-cluster-large div{background:#d9480f;color:#fff;font:700 13px/30px sans-serif;box-shadow:0 1px 5px rgba(0,0,0,.35)}
  .marker-cluster-small,.marker-cluster-medium,.marker-cluster-large{background:rgba(255,255,255,.82);box-shadow:0 0 0 2px #d9480f}
  @media(max-width:650px){
    .map-title{top:8px;font-size:18px;padding:8px 12px;white-space:normal;width:calc(100% - 110px)}
    .banner{top:67px;left:10px}
    .refresh-box{top:116px;left:10px;right:10px;min-width:0}
  }
</style>
</head>
<body>
<div id="map"></div>
<h1 class="map-title">FloodOps — Live Multi-Source Flood Map</h1>
<div class="banner" id="banner">Loading flood map...</div>
<div class="refresh-box" id="refreshBox">Loading refresh times...</div>
<div class="version-badge">Map image __APP_VERSION__</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
var COLORS={nws_fim:"#0b7285",nws_alert:"#e31a1c",fema_nfhl:"#6a3d9a",nwps_gauge:"#1f78b4"};
var LABELS={nws_fim:"NWS-FIM inundation (real flood area)",nws_alert:"NWS flood-warning area",fema_nfhl:"FEMA floodplain (flood-prone)",nwps_gauge:"River gauge in flood"};
var SHORT_LABELS={nws_fim:"NWS-FIM",nws_alert:"NWS alerts",fema_nfhl:"FEMA NFHL (static)",nwps_gauge:"NWPS gauges"};
var CONUS_BOUNDS=L.latLngBounds([[24.396308,-124.848974],[49.384358,-66.885444]]);
var map=L.map('map',{
  zoomControl:true,
  minZoom:3,
  maxBounds:L.latLngBounds([[18,-137],[57,-55]]),
  maxBoundsViscosity:0.65
});
function goHome(){map.fitBounds(CONUS_BOUNDS,{paddingTopLeft:[45,95],paddingBottomRight:[45,45]});}
goHome();
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OpenStreetMap &copy; CARTO',maxZoom:19}).addTo(map);
map.createPane("conusBoundary");
map.getPane("conusBoundary").style.zIndex=390;
map.getPane("conusBoundary").style.pointerEvents="none";
map.createPane("outsideConusMask");
map.getPane("outsideConusMask").style.zIndex=380;
map.getPane("outsideConusMask").style.pointerEvents="none";
map.createPane("floodAreas");
map.getPane("floodAreas").style.zIndex=410;
map.createPane("floodGauges");
map.getPane("floodGauges").style.zIndex=430;
map.createPane("floodFlags");
map.getPane("floodFlags").style.zIndex=650;

var HomeControl=L.Control.extend({
  options:{position:"topleft"},
  onAdd:function(){
    var box=L.DomUtil.create("div","leaflet-bar leaflet-control leaflet-control-home");
    var a=L.DomUtil.create("a","",box);
    a.href="#";
    a.title="Return to CONUS view";
    a.setAttribute("aria-label","Return to CONUS view");
    a.innerHTML="⌂";
    L.DomEvent.disableClickPropagation(box);
    L.DomEvent.on(a,"click",L.DomEvent.preventDefault).on(a,"click",goHome);
    return box;
  }
});
new HomeControl().addTo(map);

// Merge the contiguous states so only the OUTER CONUS boundary is drawn.
// A translucent inverse mask de-emphasizes Canada, Mexico, oceans, and all
// other areas outside CONUS without hiding the basemap or flood information.
// Internal state borders remain dissolved to keep flood polygons clear.
fetch("https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json")
  .then(function(r){if(!r.ok){throw new Error("HTTP "+r.status);}return r.json();})
  .then(function(us){
    var conusStates=us.objects.states.geometries.filter(function(g){
      var id=String(g.id).padStart(2,"0");
      return id!=="02" && id!=="15" && id!=="72";
    });
    var conusOutline=topojson.merge(us,conusStates);
    var polygons=conusOutline.type==="MultiPolygon"
      ? conusOutline.coordinates
      : [conusOutline.coordinates];
    var worldRing=[
      [-180,-85],[-180,85],[180,85],[180,-85],[-180,-85]
    ];
    var inverseMask=[worldRing];
    polygons.forEach(function(polygon){
      if(polygon && polygon[0]){inverseMask.push(polygon[0]);}
    });
    L.geoJSON({
      type:"Feature",
      properties:{purpose:"outside-CONUS emphasis mask"},
      geometry:{type:"Polygon",coordinates:inverseMask}
    },{
      pane:"outsideConusMask",
      interactive:false,
      style:{
        stroke:false,
        fill:true,
        fillColor:"#d9e0e5",
        fillOpacity:0.52,
        fillRule:"evenodd"
      }
    }).addTo(map);
    L.geoJSON(conusOutline,{
      pane:"conusBoundary",
      interactive:false,
      style:{color:"#31556f",weight:1.6,opacity:0.70,fill:false}
    }).addTo(map);
  }).catch(function(){/* The basemap remains available if the outline CDN fails. */});

function styleFn(f){
  var s=(f.properties||{}).source,c=COLORS[s]||"#ff7f00";
  var styles={
    nws_fim:{weight:2.2,fillOpacity:0.42},
    nws_alert:{weight:2.1,fillOpacity:0.25,dashArray:"7 4"},
    fema_nfhl:{weight:1.35,fillOpacity:0.16}
  };
  var x=styles[s]||{weight:1.8,fillOpacity:0.30};
  return {pane:"floodAreas",color:c,weight:x.weight,opacity:0.95,
    fillColor:c,fillOpacity:x.fillOpacity,dashArray:x.dashArray||null};
}
function esc(v){return String(v===undefined||v===null?"":v).replace(/[&<>"']/g,function(c){
  return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});}
function row(label,value){
  return value===undefined||value===null||value==="" ? "" :
    "<tr><td>"+esc(label)+"</td><td>"+esc(value)+"</td></tr>";
}
function rings(geom){
  if(!geom){return [];}
  if(geom.type==="Polygon"){return geom.coordinates||[];}
  if(geom.type==="MultiPolygon"){
    return (geom.coordinates||[]).reduce(function(a,p){return a.concat(p);},[]);
  }
  return [];
}
function ringAreaSqM(ring){
  if(!ring||ring.length<3){return 0;}
  var total=0,rad=Math.PI/180,R=6378137;
  for(var i=0;i<ring.length;i++){
    var a=ring[i],b=ring[(i+1)%ring.length];
    total+=(b[0]-a[0])*rad*(2+Math.sin(a[1]*rad)+Math.sin(b[1]*rad));
  }
  return Math.abs(total*R*R/2);
}
function polygonAreaSqM(geom){
  if(!geom){return 0;}
  var polys=geom.type==="Polygon"?[geom.coordinates]:(geom.type==="MultiPolygon"?geom.coordinates:[]);
  return polys.reduce(function(sum,poly){
    if(!poly||!poly.length){return sum;}
    var area=ringAreaSqM(poly[0]);
    for(var i=1;i<poly.length;i++){area-=ringAreaSqM(poly[i]);}
    return sum+Math.max(0,area);
  },0);
}
function haversineKm(a,b){
  var rad=Math.PI/180,R=6371,dLat=(b[1]-a[1])*rad,dLon=(b[0]-a[0])*rad;
  var q=Math.sin(dLat/2)**2+Math.cos(a[1]*rad)*Math.cos(b[1]*rad)*Math.sin(dLon/2)**2;
  return 2*R*Math.asin(Math.sqrt(q));
}
function perimeterKm(geom){
  return rings(geom).reduce(function(total,ring){
    for(var i=1;i<ring.length;i++){total+=haversineKm(ring[i-1],ring[i]);}
    return total;
  },0);
}
function centerOfFeature(f){
  var g=f.geometry||{};
  if(g.type==="Point"){return g.coordinates.slice(0,2);}
  var pts=[];
  rings(g).forEach(function(r){r.forEach(function(p){pts.push(p);});});
  if(!pts.length){return null;}
  return [pts.reduce(function(a,p){return a+p[0];},0)/pts.length,
          pts.reduce(function(a,p){return a+p[1];},0)/pts.length];
}
function pointInRing2D(point,ring){
  var x=point[0],y=point[1],inside=false;
  for(var i=0,j=ring.length-1;i<ring.length;j=i++){
    var xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1];
    var crosses=((yi>y)!==(yj>y)) &&
      (x<(xj-xi)*(y-yi)/((yj-yi)||Number.EPSILON)+xi);
    if(crosses){inside=!inside;}
  }
  return inside;
}
function pointInPolygon2D(point,polygon){
  if(!polygon||!polygon.length||!pointInRing2D(point,polygon[0])){return false;}
  for(var i=1;i<polygon.length;i++){
    if(pointInRing2D(point,polygon[i])){return false;}
  }
  return true;
}
function pointSegmentDistance2(point,a,b){
  var x=a[0],y=a[1],dx=b[0]-x,dy=b[1]-y;
  if(dx!==0||dy!==0){
    var t=((point[0]-x)*dx+(point[1]-y)*dy)/(dx*dx+dy*dy);
    if(t>1){x=b[0];y=b[1];}
    else if(t>0){x+=dx*t;y+=dy*t;}
  }
  dx=point[0]-x;dy=point[1]-y;
  return dx*dx+dy*dy;
}
function distanceToPolygonEdges2(point,polygon){
  var best=Infinity;
  polygon.forEach(function(ring){
    for(var i=0,j=ring.length-1;i<ring.length;j=i++){
      best=Math.min(best,pointSegmentDistance2(point,ring[j],ring[i]));
    }
  });
  return best;
}
function polygonInteriorPoint(polygon){
  var outer=polygon&&polygon[0];
  if(!outer||outer.length<3){return null;}
  var minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  outer.forEach(function(p){
    minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);
    minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);
  });
  var candidates=[
    [(minX+maxX)/2,(minY+maxY)/2],
    outer.reduce(function(a,p){return [a[0]+p[0]/outer.length,a[1]+p[1]/outer.length];},[0,0])
  ];
  var best=null,bestDistance=-1;
  function consider(point){
    if(pointInPolygon2D(point,polygon)){
      var d=distanceToPolygonEdges2(point,polygon);
      if(d>bestDistance){best=point;bestDistance=d;}
    }
  }
  candidates.forEach(consider);
  // Successively finer grids find a well-inside point even for narrow,
  // concave river corridors and polygons containing holes.
  [12,28,56].forEach(function(steps){
    var width=maxX-minX,height=maxY-minY;
    if(width===0||height===0){return;}
    for(var ix=0;ix<steps;ix++){
      for(var iy=0;iy<steps;iy++){
        consider([minX+(ix+0.5)*width/steps,minY+(iy+0.5)*height/steps]);
      }
    }
  });
  // A non-degenerate valid polygon always has an interior. This fallback
  // nudges candidate vertices toward their neighboring vertices if a very
  // thin polygon was missed by the grids.
  if(!best){
    for(var i=0;i<outer.length-1&&!best;i++){
      var prev=outer[(i-1+outer.length-1)%(outer.length-1)];
      var next=outer[(i+1)%(outer.length-1)];
      consider([(outer[i][0]*8+prev[0]+next[0])/10,
                (outer[i][1]*8+prev[1]+next[1])/10]);
    }
  }
  return best;
}
function interiorPointOfFeature(f){
  var g=f.geometry||{},polygons=g.type==="Polygon"?[g.coordinates]:
    (g.type==="MultiPolygon"?g.coordinates.slice():[]);
  if(!polygons.length){return centerOfFeature(f);}
  // For multipart features, use the largest component so the flag is not
  // placed in empty space between separate polygons.
  polygons.sort(function(a,b){
    return ringAreaSqM((b&&b[0])||[])-ringAreaSqM((a&&a[0])||[]);
  });
  for(var i=0;i<polygons.length;i++){
    var point=polygonInteriorPoint(polygons[i]);
    if(point){return point;}
  }
  return null;
}
function flagIcon(source){
  var color=source==="nws_alert"?"#e31a1c":
    (source==="fema_nfhl"?"#7b2cbf":"#f08c00");
  return L.divIcon({
    className:"flood-flag-wrap",
    html:"<div class='flood-flag' style='--flag-color:"+color+"'>"+
      "<span class='pole'></span><span class='cloth'></span><span class='foot'></span></div>",
    iconSize:[30,43],iconAnchor:[8,41],popupAnchor:[7,-39],
    tooltipAnchor:[8,-34]
  });
}
function fmtArea(m2){
  var km2=m2/1e6;
  return km2>=1 ? km2.toLocaleString(undefined,{maximumFractionDigits:2})+" km²" :
    m2.toLocaleString(undefined,{maximumFractionDigits:0})+" m²";
}
function featureClass(source){
  return {nws_fim:"Modeled river inundation area",nws_alert:"Official flood alert/warning area",
    fema_nfhl:"Regulatory flood-hazard zone (not current flooding)",
    nwps_gauge:"River gauge observation point"}[source]||"Flood-map feature";
}
var availableGauges=[];
function nearestGauge(f){
  var c=centerOfFeature(f);
  if(!c||!availableGauges.length){return null;}
  var best=null;
  availableGauges.forEach(function(g){
    var d=haversineKm(c,g.geometry.coordinates);
    if(!best||d<best.distance){best={feature:g,distance:d};}
  });
  return best;
}
function detailsHtml(f){
  var p=f.properties||{},s=p.source||"unknown",g=f.geometry||{},html="";
  html+="<div class='feature-popup'><b>"+esc(LABELS[s]||s)+"</b>";
  html+="<table>"+row("Feature type",featureClass(s));
  html+=row("Name / event",p.name||p.event);
  html+=row("Affected area",p.areaDesc||p.city);
  html+=row("Geometry",g.type);
  if(g.type==="Polygon"||g.type==="MultiPolygon"){
    html+=row("Calculated area",fmtArea(polygonAreaSqM(g)));
    html+=row("Calculated perimeter",perimeterKm(g).toLocaleString(undefined,{maximumFractionDigits:2})+" km");
  }
  html+=row("Severity",p.severity||((p.severity_class!==undefined?p.severity_class+"/4":null)));
  html+=row("Magnitude",p.magnitude);
  html+=row("Flood category",p.floodCategory);
  html+=row("Stage",p.stage!==undefined?p.stage:null);
  html+=row("Flood zone",p.FLD_ZONE);
  html+=row("Gauge/site ID",p.lid||p.ahps_lid);
  html+=row("Certainty",p.certainty);
  html+=row("Urgency",p.urgency);
  html+=row("Effective (CT)",p.effective?formatCentralTime(p.effective):null);
  html+=row("Expires (CT)",(p.ends||p.expires)?formatCentralTime(p.ends||p.expires):null);
  if(s!=="nwps_gauge"){
    var near=nearestGauge(f);
    html+=row("Nearest map gauge",near ?
      (near.feature.properties.name||near.feature.properties.lid||"Gauge")+" — "+
      near.distance.toLocaleString(undefined,{maximumFractionDigits:1})+" km" :
      "No NWPS gauge in current map");
  }
  html+="</table><div class='hint'>Area and distance are calculated from map geometry. "+
    "Land-use classification and USGS readings require additional datasets.</div></div>";
  return html;
}
function hoverHtml(f){
  var p=f.properties||{},g=f.geometry||{},parts=["<b>"+esc(p.name||p.event||LABELS[p.source]||"Map feature")+"</b>"];
  if(g.type==="Polygon"||g.type==="MultiPolygon"){parts.push("Area: "+fmtArea(polygonAreaSqM(g)));}
  if(p.FLD_ZONE){parts.push("FEMA zone: "+esc(p.FLD_ZONE));}
  if(p.stage!==undefined){parts.push("Stage: "+esc(p.stage));}
  if(p.floodCategory){parts.push("Category: "+esc(p.floodCategory));}
  return parts.join("<br>");
}
function onEach(f,layer){
  layer.bindTooltip(hoverHtml(f),{sticky:true,direction:"top",opacity:0.95});
  layer.bindPopup(detailsHtml(f),{maxWidth:390});
  if((f.geometry||{}).type!=="Point"){
    layer.on("mouseover",function(){layer.setStyle({weight:3.5,fillOpacity:0.48});layer.bringToFront();});
    layer.on("mouseout",function(){layer.setStyle(styleFn(f));});
  }
  layer.on("contextmenu",function(e){L.DomEvent.preventDefault(e.originalEvent);layer.openPopup(e.latlng);});
}
fetch('/flood_map.geojson').then(function(r){return r.json();}).then(function(gj){
  var sourceLayers={};
  var sourceOrder=["nws_fim","nws_alert","fema_nfhl","nwps_gauge"];
  var featureLayerMap=new Map();
  availableGauges=(gj.features||[]).filter(function(f){
    return (f.properties||{}).source==="nwps_gauge" && (f.geometry||{}).type==="Point";
  });

  sourceOrder.forEach(function(source){
    sourceLayers[source]=L.geoJSON(gj,{
      filter:function(f){return (f.properties||{}).source===source;},
      style:styleFn,
      pointToLayer:function(f,ll){var c=COLORS[source]||"#1f78b4";
        return L.circleMarker(ll,{pane:"floodGauges",radius:7,color:"#fff",
          fillColor:c,fillOpacity:0.95,weight:2});},
      onEachFeature:function(f,layer){featureLayerMap.set(f,layer);onEach(f,layer);}
    });
  });

  // Show flood-area layers initially. Gauges are available in the checkbox
  // control but remain hidden until the user turns them on.
  sourceLayers.nws_fim.addTo(map);
  sourceLayers.nws_alert.addTo(map);
  sourceLayers.fema_nfhl.addTo(map);

  var overlays={};
  sourceOrder.forEach(function(source){
    var count=sourceLayers[source].getLayers().length;
    overlays[LABELS[source]+" ("+count+")"]=sourceLayers[source];
  });
  function makeFlagCluster(){
    return L.markerClusterGroup({
      showCoverageOnHover:false,
      maxClusterRadius:48,
      spiderfyOnMaxZoom:true,
      disableClusteringAtZoom:8,
      chunkedLoading:true
    });
  }
  var floodFlags=makeFlagCluster();
  var femaFlags=makeFlagCluster();
  var flagCount=0;
  var femaFlagCount=0;
  (gj.features||[]).forEach(function(f){
    var p=f.properties||{},g=f.geometry||{};
    if((p.source!=="nws_fim"&&p.source!=="nws_alert"&&p.source!=="fema_nfhl")||
       (g.type!=="Polygon"&&g.type!=="MultiPolygon")){return;}
    var polygonLayer=featureLayerMap.get(f);
    if(!polygonLayer||!polygonLayer.getBounds){return;}
    var flagPoint=interiorPointOfFeature(f);
    if(!flagPoint){return;}
    var marker=L.marker([flagPoint[1],flagPoint[0]],{
      icon:flagIcon(p.source),pane:"floodFlags",
      title:p.source==="fema_nfhl" ?
        ((p.name||p.city||"FEMA flood zone")+" — static reference, not current flooding") :
        (p.name||p.event||LABELS[p.source]||"Flood location")
    });
    marker.bindTooltip(hoverHtml(f),{direction:"top",offset:[7,-35],opacity:0.96});
    marker.bindPopup(detailsHtml(f),{maxWidth:390});
    marker.on("click",function(){
      var bounds=polygonLayer.getBounds();
      if(bounds&&bounds.isValid()){map.fitBounds(bounds,{padding:[55,55],maxZoom:10});}
      window.setTimeout(function(){marker.openPopup();},220);
    });
    if(p.source==="fema_nfhl"){
      femaFlags.addLayer(marker);
      femaFlagCount++;
    }else{
      floodFlags.addLayer(marker);
      flagCount++;
    }
  });
  floodFlags.addTo(map);
  femaFlags.addTo(map);
  overlays["🚩 Flood location flags ("+flagCount+")"]=floodFlags;
  overlays["⚑ FEMA static-zone flags ("+femaFlagCount+")"]=femaFlags;
  L.control.layers(null,overlays,{collapsed:false,position:"bottomright"}).addTo(map);

  document.getElementById('banner').innerHTML=
    "<b>"+gj.features.length+" mapped features; "+flagCount+" active flood locations; "+
    femaFlagCount+" FEMA reference zones</b> — red/orange flags indicate active NWS data; "+
    "purple flags indicate static FEMA flood-hazard zones, not current flooding.";
}).catch(function(e){document.getElementById('banner').innerHTML="Failed to load map: "+e;});

var refreshStatus=null;
function formatCentralTime(value){
  if(!value){return "Not available";}
  var d=new Date(value);
  return isNaN(d.getTime()) ? value : new Intl.DateTimeFormat("en-US",{
    timeZone:"America/Chicago",year:"numeric",month:"short",day:"2-digit",
    hour:"numeric",minute:"2-digit",second:"2-digit",timeZoneName:"short"
  }).format(d);
}
function formatCountdown(seconds){
  seconds=Math.max(0,Math.floor(seconds));
  var h=String(Math.floor(seconds/3600)).padStart(2,"0");
  var m=String(Math.floor((seconds%3600)/60)).padStart(2,"0");
  var s=String(seconds%60).padStart(2,"0");
  return h+":"+m+":"+s;
}
function renderRefreshStatus(){
  if(!refreshStatus){return;}
  var rows="";
  ["nws_fim","nws_alert","fema_nfhl","nwps_gauge"].forEach(function(source){
    rows+="<tr><td>"+SHORT_LABELS[source]+"</td><td>"+formatCentralTime((refreshStatus.sources||{})[source])+"</td></tr>";
  });
  var next=null;
  if(refreshStatus.next_refresh_at){
    next=new Date(refreshStatus.next_refresh_at);
  }else if(refreshStatus.last_success_at && refreshStatus.refresh_interval_seconds){
    // Compatibility fallback for status files created by older containers.
    var last=new Date(refreshStatus.last_success_at);
    if(!isNaN(last.getTime())){
      next=new Date(last.getTime()+Number(refreshStatus.refresh_interval_seconds)*1000);
    }
  }
  if(next && isNaN(next.getTime())){next=null;}
  var seconds=next ? (next.getTime()-Date.now())/1000 : null;
  var countdown=seconds===null ? "Scheduling..." :
    (seconds<=0 ? "Refreshing now..." : formatCountdown(seconds));
  var state=refreshStatus.status||"waiting";
  var stateClass=state==="waiting"?"waiting":(state==="ok"?"ok":"error");
  var centralNow=formatCentralTime(new Date());
  document.getElementById("refreshBox").innerHTML=
    "<b>Data retrieval status</b><div style='color:#566;margin-top:2px'>"+
    "Times: Central Time (CT; CST/CDT)<br>Current CT: "+centralNow+"</div><table>"+rows+"</table>"+
    "<div style='margin-top:5px'>Next refresh: <span class='countdown'>"+countdown+"</span>"+
    " &nbsp; Status: <span class='"+stateClass+"'>"+esc(state)+"</span></div>";
}
function getRefreshStatus(){
  fetch("/refresh_status?ts="+Date.now(),{cache:"no-store"})
    .then(function(r){if(!r.ok){throw new Error("HTTP "+r.status);}return r.json();})
    .then(function(data){refreshStatus=data;renderRefreshStatus();})
    .catch(function(e){
      document.getElementById("refreshBox").innerHTML="Refresh status unavailable: "+e;
    });
}
getRefreshStatus();
setInterval(renderRefreshStatus,1000);
setInterval(getRefreshStatus,15000);

var legend=L.control({position:'bottomleft'});
legend.onAdd=function(){var d=L.DomUtil.create('div','legend');
  d.innerHTML='<b>Flood map source</b><br>'
   +'<span class="sw poly" style="background:#0b7285"></span>NWS-FIM inundation<br>'
   +'<span class="sw poly" style="background:#e31a1c"></span>NWS flood-warning<br>'
   +'<span class="sw poly" style="background:#6a3d9a"></span>FEMA floodplain<br>'
   +'<span class="sw pt" style="background:#1f78b4"></span>River gauge in flood<br>'
   +'<span style="font-size:17px;vertical-align:middle">🚩</span> Active flood location';
  return d;};
legend.addTo(map);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, bytes):
            payload = body
        elif ctype.startswith("text/"):
            payload = body.encode("utf-8")
        else:
            payload = json.dumps(body, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/map", "/index.html"):
            self._send(200, MAP_HTML.replace("__APP_VERSION__", APP_VERSION),
                       ctype="text/html; charset=utf-8")
        elif path in ("/flood_map.geojson", "/map.geojson"):
            fc = load_map()
            print(f"GET {path} -> {len(fc.get('features', []))} features")
            self._send(200, fc)
        elif path == "/sources":
            s = summarize(load_map())
            print(f"GET {path} -> {s['features_by_source']}")
            self._send(200, s)
        elif path == "/refresh_status":
            self._send(200, load_refresh_status())
        else:
            self._send(404, {"error": "not found",
                             "endpoints": ["/", "/flood_map.geojson",
                                           "/sources", "/refresh_status"]})

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"FloodOps multi-source flood-map API on http://0.0.0.0:{port}")
    print("  GET /                    interactive live map (open in a browser)")
    print("  GET /flood_map.geojson   full combined map (GeoJSON)")
    print("  GET /sources             per-source counts + discrepancy summary")
    print("  GET /refresh_status      retrieval times + next refresh")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
