// Simulated-camera tests for the cropped, guarantee-gated capture page.
//
//     node scripts/capture_sim.js
//
// The capture page cannot be unit-tested against a real phone, and the adaptive
// path is the part most likely to break silently: a sweep that converges on the
// wrong distance, or a run that leaves the torch on, looks fine in code review
// and fails in the field. So this runs the page's own script in a VM against a
// FAKE Android camera — a lens with a true focus distance, ridges that fade as
// it defocuses, a torch that helps, and a crop-zoom that magnifies — and checks
// the search finds the truth.
//
// Six device profiles are covered: iPhone/desktop (no controls at all, must
// fall back to the burst), full Android with a scoring server and real stills,
// a thumb too small at the lens near limit (the zoom stage), a broken
// takePhoto, focus without a torch, and a hand that leaves mid-sweep.
//
// No dependencies: a minimal canvas with real box-average downscaling is
// implemented here, because the ridge signal is measured on downscaled pixels
// and a nearest-neighbour stub would not exercise it.

// Scenario harness: iPhone/desktop regression, a scoring server + real stills,
// the zoom stage, and a broken takePhoto.
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync('fingerphoto/api/static/capture.html', 'utf8');
const js = html.split('<script>')[1].split('</script>')[0];
const sleep = ms => new Promise(r => setTimeout(r, ms));
let failures = 0;
function check(name, ok, detail){
  console.log((ok ? '  PASS  ' : '  FAIL  ') + name + (detail ? '   ' + detail : ''));
  if(!ok) failures++;
}

function build(opt){
  const VW = 1920, VH = 1080;
  const cam = {focus: 0.5, torch: false, ev: 0, zoom: 1, calls: [], stills: 0};
  const trueD = opt.trueD || 0.12, period0 = 23, sigma = 1.1;
  const amplitude = () => {
    let a = 20*Math.exp(-Math.pow(1/cam.focus - 1/trueD, 2)/(2*sigma*sigma));
    if(cam.torch) a *= 1.35;
    return a;
  };
  // skin covers `skinFrac` of the width at zoom 1 and grows with the crop
  const skinFrac = () => Math.min(1, (opt.skinFrac || 1) * cam.zoom);

  class Ctx {
    constructor(c){ this.c = c; }
    drawImage(src, ...a){
      let sx=0, sy=0, sw, sh, dx=0, dy=0, dw, dh;
      const sW = src.width || src.videoWidth, sH = src.height || src.videoHeight;
      if(a.length === 2){ [dx,dy]=a; dw=sW; dh=sH; sw=sW; sh=sH; }
      else if(a.length === 4){ [dx,dy,dw,dh]=a; sw=sW; sh=sH; }
      else { [sx,sy,sw,sh,dx,dy,dw,dh]=a; }
      const sd = src.render ? src.render() : src.data, d = this.c.data;
      for(let y=0; y<dh; y++){
        const y0 = sy+Math.floor(y*sh/dh), y1 = Math.max(y0+1, sy+Math.floor((y+1)*sh/dh));
        for(let x=0; x<dw; x++){
          const x0 = sx+Math.floor(x*sw/dw), x1 = Math.max(x0+1, sx+Math.floor((x+1)*sw/dw));
          let r=0,g=0,b=0,n=0;
          for(let yy=y0; yy<y1 && yy<sH; yy++) for(let xx=x0; xx<x1 && xx<sW; xx++){
            const i=(yy*sW+xx)*4; r+=sd[i]; g+=sd[i+1]; b+=sd[i+2]; n++;
          }
          if(!n) continue;
          const j=((dy+y)*this.c.width+(dx+x))*4;
          if(j<0 || j+3>=d.length) continue;
          d[j]=r/n; d[j+1]=g/n; d[j+2]=b/n; d[j+3]=255;
        }
      }
    }
    getImageData(x,y,w,h){
      const out = new Uint8ClampedArray(w*h*4);
      for(let yy=0; yy<h; yy++) for(let xx=0; xx<w; xx++){
        const si=((y+yy)*this.c.width+(x+xx))*4, di=(yy*w+xx)*4;
        out[di]=this.c.data[si]; out[di+1]=this.c.data[si+1]; out[di+2]=this.c.data[si+2]; out[di+3]=255;
      }
      return {data: out};
    }
  }
  function makeCanvas(){
    const c = {_w:0,_h:0,data:new Uint8ClampedArray(0),_ctx:null};
    Object.defineProperty(c,'width',{get:()=>c._w,set:v=>{c._w=v|0;c.data=new Uint8ClampedArray(c._w*c._h*4);}});
    Object.defineProperty(c,'height',{get:()=>c._h,set:v=>{c._h=v|0;c.data=new Uint8ClampedArray(c._w*c._h*4);}});
    c.getContext = () => (c._ctx || (c._ctx = new Ctx(c)));
    c.toBlob = cb => setTimeout(()=>cb({size:Math.round(c._w*c._h/8), type:'image/jpeg', _c:c}), 0);
    c.toDataURL = () => 'data:,';
    return c;
  }
  const videoEl = {
    videoWidth:VW, videoHeight:VH, style:{}, srcObject:null, play:async()=>{},
    addEventListener(){}, classList:{toggle(){}}, dataset:{},
    render(){
      cam.frames = (cam.frames||0) + 1;
      if(opt.loseAfter && cam.frames > opt.loseAfter){   // the hand leaves
        const d = new Uint8ClampedArray(VW*VH*4);
        for(let i=0; i<d.length; i+=4){ d[i]=70; d[i+1]=90; d[i+2]=110; d[i+3]=255; }
        return d;
      }
      const amp = amplitude(), d = new Uint8ClampedArray(VW*VH*4);
      const period = period0*cam.zoom, sw = skinFrac()*VW, x0 = (VW-sw)/2;
      for(let y=0; y<VH; y++) for(let x=0; x<VW; x++){
        const i=(y*VW+x)*4;
        if(x < x0 || x > x0+sw){                 // background: not skin
          d[i]=70; d[i+1]=90; d[i+2]=110; d[i+3]=255; continue;
        }
        let base = 168 + amp*Math.sin(2*Math.PI*x/period)
                 + 22*Math.sin(2*Math.PI*x/420+1) + (cam.torch?26:0) + cam.ev*18
                 + (((x*7+y*13)%5)-2);
        d[i]=Math.max(0,Math.min(255,base*1.16));
        d[i+1]=Math.max(0,Math.min(255,base*0.86));
        d[i+2]=Math.max(0,Math.min(255,base*0.75));
        d[i+3]=255;
      }
      return d;
    }
  };
  const track = {
    getCapabilities: () => opt.caps(VW, VH),
    getSettings: () => ({width:VW, height:VH, zoom:cam.zoom}),
    applyConstraints: async c => {
      cam.calls.push(JSON.stringify(c));
      // a driver that goes away and never answers — the freeze mode
      if(opt.hangFocus && (c.advanced||[]).some(a => a.focusDistance != null)){
        cam.hangs = (cam.hangs||0) + 1;
        return new Promise(() => {});          // never resolves, never rejects
      }
      if(opt.refuseFocus && (c.advanced||[]).some(a => a.focusDistance != null
                                                    || a.focusMode === 'manual')){
        const e = new Error('Unsupported constraint focusDistance');
        e.name = 'OverconstrainedError';
        throw e;
      }
      for(const a of (c.advanced||[])){
        if(a.focusDistance!=null) cam.focus=a.focusDistance;
        if(a.torch!=null) cam.torch=a.torch;
        if(a.exposureCompensation!=null) cam.ev=a.exposureCompensation;
        if(a.zoom!=null) cam.zoom=a.zoom;
      }
    },
    stop(){},
  };
  const stream = {getVideoTracks:()=>[track], getTracks:()=>[track]};
  const els = {v: videoEl};
  const el = id => els[id] || (els[id] = {id, style:{}, textContent:'', innerHTML:'',
    value: id==='key' ? (opt.key||'') : (id==='name' ? 'Tester' : ''), className:'', dataset:{},
    files:[], disabled:false, classList:{toggle(){},add(){},remove(){},contains:()=>false},
    addEventListener(){}, setAttribute(){}, click(){}, querySelector:()=>({textContent:'',style:{}})});
  // a stand-in for the server's ridge detector: contrast at the known period
  function serverScore(c){
    if(!c) return 0;
    const w=c._w, h=c._h, half=Math.max(1, Math.round(period0*cam.zoom/2));
    let mad=0, n=0, skin=0, tot=0;
    for(let y=0; y<h; y+=3) for(let x=0; x+half<w; x+=2){
      const i=(y*w+x)*4, j=(y*w+x+half)*4;
      const r=c.data[i], g=c.data[i+1], b=c.data[i+2];
      const cr = 0.5*r - 0.419*g - 0.081*b + 128;
      tot++;
      if(cr>135 && cr<185 && r>g && r>b){ skin++; mad += Math.abs(r-c.data[j]); n++; }
    }
    const fill = tot ? skin/tot : 0;
    return n ? Math.min(1,(mad/n)/12) * Math.min(1, fill/0.75) : 0;
  }
  const ctx = {
    console, setTimeout, clearTimeout, setInterval, clearInterval, performance,
    Promise, Math, Date, JSON, Object, Array, Set, Float32Array, Float64Array,
    Uint8ClampedArray, Uint8Array, Error, RegExp, isNaN, parseInt, parseFloat, String, Number,
    requestAnimationFrame: cb => setTimeout(cb, 120),
    localStorage:{getItem:k=>k==='fp_key'?(opt.key||null):null, setItem(){}},
    location:{search:''},
    navigator:{mediaDevices:{getUserMedia: async()=>stream,
                             enumerateDevices: async()=>[{kind:'videoinput',deviceId:'b'}]}},
    document:{getElementById: el, createElement: t => t==='canvas'?makeCanvas():el('t_'+t),
              querySelectorAll: ()=>[], body:{clientWidth:400}},
    FormData: class { constructor(){this.parts=[];} append(k,v){this.parts.push(v);} },
    fetch: async (url, o) => {
      if(/\/capture\/session/.test(url)){
        cam.renews = (cam.renews || 0) + 1;
        return {ok:true, status:200, json:async()=>({status:'ok'})};
      }
      if(!/\/v1\/quality/.test(url)) return {ok:false, status:404, json:async()=>({})};
      if(!opt.key) return {ok:false, status:401, json:async()=>({})};
      if(opt.expireOnce && !cam.expiredOnce){
        cam.expiredOnce = true;
        return {ok:false, status:401, json:async()=>({})};
      }
      const blob = o.body.parts.find(p => p && p._c);
      return {ok:true, status:200, json: async () => ({
        score: serverScore(blob && blob._c), strategy:'distal', min_ridge_quality:0.40})};
    },
    XMLHttpRequest: class { open(){} setRequestHeader(){} send(){} },
    innerHeight:800, addEventListener(){},
  };
  if(opt.imageCapture){
    ctx.ImageCapture = class {
      constructor(){ }
      async getPhotoCapabilities(){
        return {imageWidth:{min:640,max:4000,step:2}, imageHeight:{min:480,max:3000,step:2}};
      }
      async takePhoto(){
        if(opt.stillsBroken) throw new Error('NotSupportedError');
        cam.stills++;
        return {size: 2_400_000, type:'image/jpeg'};
      }
    };
  }
  ctx.window = ctx; ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(js, ctx, {filename:'capture.html'});
  return {ctx, cam};
}

async function legacyAdaptiveTests(){
  // ---- 0. The pieces on their own ---------------------------------------
  console.log('\n== units: the ridge signal, still sizing, name parsing ==');
  {
    const {ctx, cam} = build({caps:(w,h)=>({width:{max:w}, height:{max:h},
      focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
      torch:true})});
    await sleep(400);
    cam.focus = 0.12; const sharp = ctx.ridgeMetric(ctx.cropCanvas());
    cam.focus = 0.50; const blur  = ctx.ridgeMetric(ctx.cropCanvas());
    check('the ridge signal ranks focus far above defocus',
          sharp.metric > blur.metric*3,
          sharp.metric.toFixed(1) + ' in focus vs ' + blur.metric.toFixed(1) + ' out');
    // the frame carries a strong slow shading gradient at every focus setting;
    // a plain sharpness measure would score it, the band-pass must not
    check('the signal ignores slow shading', blur.metric < 2, blur.metric.toFixed(2));
    check('still size capped at 4096, sensor aspect kept', (() => {
      const s = ctx.pickStill({imageWidth:{min:0,max:8000,step:2},
                               imageHeight:{min:0,max:6000,step:2}});
      return s.w === 4096 && s.h === 3072; })());
    check('a candidate name reads back in words',
          ctx.wonLabel('still_t1_d110.jpg') === 'still, torch on, focus 11 cm',
          ctx.wonLabel('still_t1_d110.jpg'));
    check('a name with no distance is not guessed at',
          ctx.wonLabel('still_t1_dna.jpg') === '');
  }

  // ---- 1. iPhone / desktop: no manual focus, no torch --------------------
  console.log('\n== iPhone / desktop (no focus control, no torch) ==');
  {
    const {ctx, cam} = build({caps: (w,h) => ({width:{max:w}, height:{max:h},
                                               focusMode:['continuous']})});
    await sleep(400);
    check('focus reported honestly', ctx.window.__focusCtl === 'continuous',
          ctx.window.__focusCtl);
    check('torch reported unavailable', ctx.window.__torch === 'unavailable');
    check('no zoom label', ctx.window.__zoom === '');
    const shots = await ctx.burst();
    check('burst path still returns 5 frames', shots.length === 5, String(shots.length));
    check('burst frames keep their names', shots[0].name === 'burst_0.jpg');
    check('no camera constraints beyond setup', !cam.calls.some(c => /focusDistance|torch/.test(c)));
  }

  // ---- 2. Android, server scoring, real stills ---------------------------
  console.log('\n== Android Chrome, server scoring, ImageCapture ==');
  {
    const {ctx, cam} = build({key:'k', imageCapture:true,
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
        torch:true, zoom:{min:1,max:8,step:0.1},
        exposureCompensation:{min:-3,max:3,step:0.166}})});
    await sleep(400);
    check('still size capped and aspect kept', ctx.window.__stills === '4000x3000',
          ctx.window.__stills);
    const parts = await ctx.autoCapture();
    const tr = ctx.traceHtml().replace(/<[^>]+>/g,'');
    console.log(tr.split('capture trace')[1].replace(/^[^\n]*\n/,''));
    check('peak chosen by the server score', /peak\s+0\.1[12]m by server ridge score/.test(tr),
          (/peak[^\n]*/.exec(tr)||[])[0]);
    check('server scores present in the trace', /ridge 0\.\d\d/.test(tr));
    check('stills were taken', cam.stills >= 1, cam.stills + ' stills');
    check('parts are stills, not frames',
          parts.filter(p=>/^still_/.test(p.name)).length === 1,
          parts.map(p=>p.name).join(' '));
    check('zoom skipped, the score is already at the ceiling',
          /zoom\s+skipped \(ridge score already at the ceiling/.test(tr),
          (/zoom\s+skipped[^\n]*/.exec(tr)||[])[0]);
  }

  // ---- 3. Small thumb at the near limit: the zoom stage should run -------
  console.log('\n== Android, thumb too small, lens at its near limit ==');
  {
    const {ctx, cam} = build({key:'k', imageCapture:true, trueD:0.10, skinFrac:0.30,
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
        torch:true, zoom:{min:1,max:8,step:0.1}})});
    await sleep(400);
    await ctx.autoCapture();
    const tr = ctx.traceHtml().replace(/<[^>]+>/g,'');
    console.log(tr.split('capture trace')[1].replace(/^[^\n]*\n/,''));
    check('zoom stage ran', /zoom\s+1\.5x/.test(tr));
    check('zoom won and was kept for the shots', /zoom\s+kept 1\.5x/.test(tr),
          (/zoom\s+(kept|no gain)[^\n]*/.exec(tr)||[])[0]);
    check('the shots were taken zoomed in', /zoom\s+1\.5x[^\n]*ridge (0\.9|1\.00)/.test(tr),
          (/zoom\s+1\.5x[^\n]*/.exec(tr)||[])[0]);
    check('camera zoom restored to the user setting after the run',
          Math.abs(cam.zoom - 1) < 1e-9, 'zoom ' + cam.zoom);
  }

  // ---- 4. takePhoto broken: fall back to a recorded clip / burst --------
  console.log('\n== Android with a broken takePhoto ==');
  {
    const {ctx, cam} = build({key:'k', imageCapture:true, stillsBroken:true,
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
        torch:true})});
    await sleep(400);
    const parts = await ctx.autoCapture();
    check('run completed with a real fallback capture', parts.length >= 1, String(parts.length));
    check('did not ship preview-frame fakes', parts.every(p => !/^frame_/.test(p.name)),
          parts.map(p=>p.name).join(' '));
    check('meta says stills are unavailable', /unavailable/.test(ctx.window.__stills),
          ctx.window.__stills);
    check('only one takePhoto attempt was made', cam.stills === 0);
  }

  // ---- 5. Focus control but no torch (some tablets) ---------------------
  console.log('\n== Android, manual focus but no torch ==');
  {
    const {ctx, cam} = build({key:'k', imageCapture:true,
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01}})});
    await sleep(400);
    check('torch reported unavailable', ctx.window.__torch === 'unavailable');
    const parts = await ctx.autoCapture();
    const tr = ctx.traceHtml().replace(/<[^>]+>/g,'');
    check('no light stage in the trace', !/^light/m.test(tr));
    check('one still, torch-off', parts.length === 1
          && parts.every(p => /still_t0_/.test(p.name)), parts.map(p=>p.name).join(' '));
    check('sweep still found the distance', /peak\s+0\.1[12]m/.test(tr),
          (/peak[^\n]*/.exec(tr)||[])[0]);
  }

  // ---- 6. The hand leaves mid-sweep: abort, but restore the camera -------
  console.log('\n== hand removed mid-sweep ==');
  {
    const {ctx, cam} = build({key:'k', loseAfter:14,
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
        torch:true, exposureCompensation:{min:-3,max:3,step:0.166}})});
    await sleep(400);
    let err = null;
    try{ await ctx.autoCapture(); }catch(e){ err = e; }
    check('the run aborted', !!err, err && err.message);
    check('the message tells the user what to do',
          !!err && /keep it inside the outline/.test(err.message));
    check('torch restored to its entry state', cam.torch === true, 'torch ' + cam.torch);
    check('autofocus handed back to the camera',
          cam.calls.slice(-4).some(c => /"focusMode":"continuous"/.test(c)));
  }

  // ---- 7. Manual focus advertised but refused by the driver -------------
  console.log('\n== camera advertises manual focus, then refuses it ==');
  {
    const {ctx, cam} = build({key:'k', refuseFocus:true,
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
        torch:true})});
    await sleep(400);
    let err = null, parts = null;
    try{ parts = await ctx.autoCapture(); }catch(e){ err = e; }
    const tr = ctx.traceHtml().replace(/<[^>]+>/g,'');
    check('the capture still completed', !err && parts && parts.length > 0,
          err ? err.message : (parts||[]).length + ' parts');
    check('the refusal is recorded with its reason',
          /focus\s+REFUSED at all \d+ distances — OverconstrainedError/.test(tr),
          (/focus\s+REFUSED[^\n]*/.exec(tr)||[])[0]);
    check('the reported capabilities are in the trace',
          /caps\s+focusDistance \{"min":0\.1/.test(tr));
    check('it fell back to the camera autofocus',
          /falling back to the camera/.test(tr));
    check('the light search still ran', /^light\s+winner/m.test(tr),
          (/light\s+winner[^\n]*/.exec(tr)||[])[0]);
    check('candidates carry no invented distance',
          parts.every(p => /_dna|burst_|clip\./.test(p.name)), (parts||[]).map(p=>p.name).join(' '));
  }

  // ---- 8. A device reporting focusDistance.min as 0 ---------------------
  console.log('\n== focusDistance.min reported as 0 ==');
  {
    const {ctx, cam} = build({key:'k',
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0, max:3, step:0.12},
        torch:true})});
    await sleep(400);
    await ctx.autoCapture();
    const asked = cam.calls.map(c => JSON.parse(c))
      .flatMap(c => (c.advanced||[]))
      .map(a => a.focusDistance).filter(d => d != null);
    check('some distances were applied', asked.length > 5, asked.length + ' calls');
    check('never asked for zero metres', asked.every(d => d > 0),
          'min asked ' + Math.min.apply(null, asked));
    check('every distance sits on the reported grid',
          asked.every(d => Math.abs(d/0.12 - Math.round(d/0.12)) < 1e-6));
    check('every distance is within the reported range',
          asked.every(d => d >= 0 && d <= 3), 'max asked ' + Math.max.apply(null, asked));
  }

  // ---- 9. applyConstraints never answers: must not freeze the capture ---
  console.log('\n== camera accepts the call and never answers ==');
  {
    const {ctx, cam} = build({key:'k', hangFocus:true,
      caps:(w,h)=>({width:{max:w},height:{max:h},
        focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
        torch:true})});
    await sleep(400);
    const started = Date.now();
    let err = null, parts = null;
    try{ parts = await ctx.autoCapture(); }catch(e){ err = e; }
    const took = (Date.now()-started)/1000;
    const tr = ctx.traceHtml().replace(/<[^>]+>/g,'');
    check('the run finished instead of hanging', !err && parts && parts.length > 0,
          err ? err.message : parts.length + ' parts in ' + took.toFixed(0) + 's');
    check('it gave up on the lens quickly', took < 25, took.toFixed(0) + 's');
    check('it stopped after the first two positions, not all eight',
          cam.hangs <= 2, cam.hangs + ' hanging calls made');
    check('the timeout is named in the trace', /did not answer in \d+ms/.test(tr),
          (/focus\s+REFUSED[^\n]*/.exec(tr)||[])[0]);
    check('it still produced candidates under the camera autofocus',
          parts.every(p => /_dna|burst_|clip\./.test(p.name)), parts.map(p=>p.name).join(' '));
  }

  console.log(failures ? '\n' + failures + ' FAILURES' : '\nall checks passed');
  process.exit(failures ? 1 : 0);
}

(async () => {
  console.log('\n== guarantee-gated cropped capture ==');
  const {ctx, cam} = build({
    key:'k', expireOnce:true,
    caps:(w,h)=>({width:{max:w},height:{max:h},
      focusMode:['continuous','manual'], focusDistance:{min:0.10,max:3,step:0.01},
      torch:true, zoom:{min:1,max:8,step:0.1}})
  });
  ctx.setArmed(false);
  await sleep(400);
  const renewed = await ctx.sessionFetch('/v1/quality', {
    method:'POST', body:new ctx.FormData()
  });
  check('expired browser session renews and retries automatically',
        renewed.status === 200 && cam.renews === 1, `renewals ${cam.renews || 0}`);
  check('new flash control is available', !ctx.document.getElementById('torch').disabled
        && ctx.document.getElementById('torch').textContent === 'Flash: Off');
  await ctx.toggleTorch();
  check('flash button turns the light on', cam.torch
        && ctx.document.getElementById('torch').textContent === 'Flash: On');
  await ctx.toggleTorch();
  check('flash button turns the light off', !cam.torch
        && ctx.document.getElementById('torch').textContent === 'Flash: Off');
  cam.calls = []; // the automatic capture must not add any flash/focus calls
  const full = ctx.grabFrame();
  const fullPixels = full.width * full.height;
  const parts = await ctx.autoCapture();
  const shot = parts[0];
  check('one guide crop is produced', parts.length === 1 && shot.name === 'fingerprint-crop.jpg');
  check('upload contains only the guide region', shot.width * shot.height < fullPixels,
        `${shot.width}x${shot.height} from ${full.width}x${full.height}`);
  check('crop is compressed before upload', shot.blob.size < fullPixels / 8,
        `${(shot.blob.size/1048576).toFixed(2)} MB`);
  check('capture does not sweep focus', !cam.calls.some(c => /focusDistance/.test(c)));
  check('capture does not toggle the torch', !cam.calls.some(c => /torch/.test(c)));
  const tr = ctx.traceHtml().replace(/<[^>]+>/g,' ');
  check('trace reports crop dimensions and upload size', /crop\s+\d+x\d+ inside guide/.test(tr)
        && /upload\s+\d+\.\d MB/.test(tr), tr.replace(/\s+/g,' ').trim());
  await ctx.collectFirstEnrollmentScan();
  check('enrollment stops after scan one and asks for finger removal',
        /Lift your finger/.test(ctx.document.getElementById('status').textContent));
  check('second scan cannot reuse the same placement',
        ctx.document.getElementById('shoot').disabled
        && ctx.document.getElementById('shoot').textContent === 'Waiting for finger to lift');
  console.log(failures ? '\n' + failures + ' FAILURES' : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})();
