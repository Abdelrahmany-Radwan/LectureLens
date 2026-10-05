const $ = (id) => document.getElementById(id);
const E2E = new URLSearchParams(location.search).has("e2e");

const state = {
  chunks: [],
  embeddings: null,
  recorder: null,
  stream: null,
  audioChunks: [],
  recordingStartedAt: null,
  timerHandle: null,
  raf: null,
  audioContext: null,
  analyser: null,
  source: null,
  currentAudioUrl: null,
  currentAudioBlob: null,
  currentLectureId: null,
  worker: null,
  workerSeq: 0,
  workerPending: new Map(),
};

const SAMPLE = `[00:00] Today we're talking about recursion and why a function might call itself.
[00:24] A base case is the stopping condition. It tells the recursive function when it should return instead of calling itself again.
[01:02] Every recursive call creates another frame on the call stack, so the computer keeps track of unfinished calls.
[01:48] If the base case is missing or unreachable, recursion can continue until the program runs out of stack space.
[02:31] The recursive case is the part that reduces the problem and calls the function again with a smaller input.
[03:20] When the base case is reached, the function starts returning and the call stack unwinds in reverse order.
[04:05] A useful way to debug recursion is to trace one small example by hand and write down each function call.
[04:48] Recursion is especially useful for tree traversal because each subtree has the same structure as the original tree.`;

function secondsFromTimestamp(ts){
  const parts = ts.split(":").map(Number);
  if(parts.length===2) return parts[0]*60+parts[1];
  if(parts.length===3) return parts[0]*3600+parts[1]*60+parts[2];
  return 0;
}

function formatTime(sec){
  sec=Math.max(0,Math.floor(sec||0));
  const h=Math.floor(sec/3600), m=Math.floor((sec%3600)/60), s=sec%60;
  return h
    ? `${h.toString().padStart(2,"0")}:${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}`
    : `${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}`;
}

function parseTranscript(text){
  const lines=text.split(/\n+/).map(x=>x.trim()).filter(Boolean);
  const parsed=[];
  for(const line of lines){
    const m=line.match(/^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)$/);
    if(m) parsed.push({time:m[1],seconds:secondsFromTimestamp(m[1]),text:m[2]});
  }
  if(!parsed.length && text.trim()){
    const sentences=text.match(/[^.!?]+[.!?]+|[^.!?]+$/g)||[];
    return sentences.map((s,i)=>({time:formatTime(i*20),seconds:i*20,text:s.trim()}));
  }
  return parsed;
}

function tokenize(text){
  return new Set((text.toLowerCase().match(/[a-z0-9+#.]+/g)||[]).filter(w=>w.length>2));
}

function lexical(a,b){
  const A=tokenize(a), B=tokenize(b);
  if(!A.size||!B.size) return 0;
  let hit=0;
  for(const x of A) if(B.has(x)) hit++;
  return hit/Math.sqrt(A.size*B.size);
}

function cosine(a,b){
  let dot=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  return dot/(Math.sqrt(na)*Math.sqrt(nb)||1);
}

/* ---------- ML worker ---------- */

function ensureWorker(){
  if(state.worker) return state.worker;
  const worker = new Worker("./ml-worker.js",{type:"module"});
  worker.onmessage = (event) => {
    const {id,ok,result,error}=event.data;
    const pending=state.workerPending.get(id);
    if(!pending) return;
    state.workerPending.delete(id);
    ok ? pending.resolve(result) : pending.reject(new Error(error||"Worker task failed."));
  };
  worker.onerror = (event) => {
    for(const pending of state.workerPending.values()){
      pending.reject(new Error(event.message||"ML worker failed."));
    }
    state.workerPending.clear();
  };
  state.worker=worker;
  return worker;
}

function workerTask(type,payload,transfer=[]){
  const worker=ensureWorker();
  const id=++state.workerSeq;
  return new Promise((resolve,reject)=>{
    state.workerPending.set(id,{resolve,reject});
    worker.postMessage({id,type,payload},transfer);
  });
}

async function embedTexts(texts){
  if(E2E){
    $("modelStatus").innerHTML="<i></i> lexical test mode";
    return null;
  }
  $("modelStatus").innerHTML="<i></i> loading MiniLM…";
  try{
    const vectors=await workerTask("embed",{texts});
    $("modelStatus").classList.add("ready");
    $("modelStatus").innerHTML="<i></i> MiniLM ready";
    return vectors;
  }catch(err){
    console.warn(err);
    $("modelStatus").innerHTML="<i></i> lexical fallback";
    return null;
  }
}

/* ---------- IndexedDB lecture library ---------- */

const DB_NAME="lecturelens-db";
const DB_VERSION=1;
const STORE="lectures";

function openDb(){
  return new Promise((resolve,reject)=>{
    const req=indexedDB.open(DB_NAME,DB_VERSION);
    req.onupgradeneeded=()=>{
      const db=req.result;
      if(!db.objectStoreNames.contains(STORE)){
        const store=db.createObjectStore(STORE,{keyPath:"id"});
        store.createIndex("updatedAt","updatedAt");
      }
    };
    req.onsuccess=()=>resolve(req.result);
    req.onerror=()=>reject(req.error);
  });
}

async function dbPut(record){
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,"readwrite");
    tx.objectStore(STORE).put(record);
    tx.oncomplete=()=>resolve();
    tx.onerror=()=>reject(tx.error);
  });
}

async function dbGet(id){
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,"readonly");
    const req=tx.objectStore(STORE).get(id);
    req.onsuccess=()=>resolve(req.result||null);
    req.onerror=()=>reject(req.error);
  });
}

async function dbDelete(id){
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,"readwrite");
    tx.objectStore(STORE).delete(id);
    tx.oncomplete=()=>resolve();
    tx.onerror=()=>reject(tx.error);
  });
}

async function dbAll(){
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,"readonly");
    const req=tx.objectStore(STORE).getAll();
    req.onsuccess=()=>resolve((req.result||[]).sort((a,b)=>b.updatedAt-a.updatedAt));
    req.onerror=()=>reject(req.error);
  });
}

function escapeHtml(value){
  return String(value)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;");
}

async function renderLibrary(){
  const list=$("libraryList");
  const empty=$("libraryEmpty");
  let lectures=[];
  try{ lectures=await dbAll(); }catch(err){ console.warn(err); }
  list.innerHTML="";
  empty.classList.toggle("hidden",lectures.length>0);

  for(const lecture of lectures){
    const item=document.createElement("div");
    item.className="saved-lecture";
    item.innerHTML=`
      <button class="lecture-item ${lecture.id===state.currentLectureId?"active":""}" type="button" data-open="${lecture.id}">
        <span class="lecture-icon">${escapeHtml((lecture.title||"L").slice(0,2).toUpperCase())}</span>
        <span><b>${escapeHtml(lecture.title||"Untitled lecture")}</b><small>${new Date(lecture.updatedAt).toLocaleDateString()} · saved</small></span>
      </button>
      <button class="lecture-delete" type="button" aria-label="Delete ${escapeHtml(lecture.title||"lecture")}" data-delete="${lecture.id}">×</button>
    `;
    list.appendChild(item);
  }

  list.querySelectorAll("[data-open]").forEach(btn=>btn.addEventListener("click",()=>loadLecture(btn.dataset.open)));
  list.querySelectorAll("[data-delete]").forEach(btn=>btn.addEventListener("click",async()=>{
    const id=btn.dataset.delete;
    if(!confirm("Delete this saved lecture from this browser?")) return;
    await dbDelete(id);
    if(state.currentLectureId===id) newLecture();
    await renderLibrary();
  }));
}

async function saveLecture(){
  const transcript=$("transcriptInput").value.trim();
  if(!transcript && !state.currentAudioBlob){
    $("error").textContent="Add audio or a transcript before saving.";
    return;
  }
  const id=state.currentLectureId||crypto.randomUUID();
  const record={
    id,
    title:$("lectureTitle").value.trim()||"Untitled lecture",
    course:$("courseLabel").textContent,
    transcript,
    audioBlob:state.currentAudioBlob||null,
    updatedAt:Date.now(),
  };
  await dbPut(record);
  state.currentLectureId=id;
  $("recordStatus").textContent="Lecture saved in this browser";
  await renderLibrary();
}

async function loadLecture(id){
  const lecture=await dbGet(id);
  if(!lecture) return;
  newLecture(false);
  state.currentLectureId=lecture.id;
  $("lectureTitle").value=lecture.title||"Untitled lecture";
  $("courseLabel").textContent=lecture.course||"LECTURE SESSION";
  $("transcriptInput").value=lecture.transcript||"";

  if(lecture.audioBlob){
    state.currentAudioBlob=lecture.audioBlob;
    state.currentAudioUrl=URL.createObjectURL(lecture.audioBlob);
    $("player").src=state.currentAudioUrl;
    $("transcribeBtn").disabled=false;
    $("recordStatus").textContent="Loaded saved audio";
  }

  if(lecture.transcript) await buildIndex({persist:false});
  await renderLibrary();
}

function newLecture(render=true){
  $("lectureTitle").value="Untitled lecture";
  $("courseLabel").textContent="LECTURE SESSION";
  $("transcriptInput").value="";
  $("timeline").innerHTML="";
  $("timeline").classList.add("hidden");
  if(state.currentAudioUrl) URL.revokeObjectURL(state.currentAudioUrl);
  state.currentAudioUrl=null;
  state.currentAudioBlob=null;
  state.currentLectureId=null;
  $("player").removeAttribute("src");
  $("player").load();
  $("transcribeBtn").disabled=true;
  $("transcribeStatus").className="transcribe-status";
  $("transcribeStatus").querySelector("span:last-child").textContent="Add or record audio, then transcribe it locally with Whisper.";
  state.chunks=[];
  state.embeddings=null;
  $("searchResults").innerHTML='<div class="placeholder-card"><b>New lecture ready.</b><p>Record audio, import audio, or paste a timestamped transcript to begin.</p></div>';
  $("notesOutput").innerHTML='<p class="muted">Build an index and your lecture outline will show up here.</p>';
  $("studyOutput").innerHTML='<p class="muted">Quick-review cards will show up here.</p>';
  if(render) renderLibrary();
}

/* ---------- transcription ---------- */

async function decodeAudioToMono16k(blob){
  const bytes=await blob.arrayBuffer();
  const AudioCtx=window.AudioContext||window.webkitAudioContext;
  const decodeCtx=new AudioCtx();
  let buffer;
  try{
    buffer=await decodeCtx.decodeAudioData(bytes.slice(0));
  }finally{
    await decodeCtx.close();
  }

  const targetRate=16000;
  if(buffer.sampleRate===targetRate){
    const mono=new Float32Array(buffer.length);
    for(let c=0;c<buffer.numberOfChannels;c++){
      const channel=buffer.getChannelData(c);
      for(let i=0;i<mono.length;i++) mono[i]+=channel[i]/buffer.numberOfChannels;
    }
    return mono;
  }

  const length=Math.ceil(buffer.duration*targetRate);
  const offline=new OfflineAudioContext(1,length,targetRate);
  const source=offline.createBufferSource();
  source.buffer=buffer;
  source.connect(offline.destination);
  source.start(0);
  const rendered=await offline.startRendering();
  return new Float32Array(rendered.getChannelData(0));
}

async function transcribeCurrentAudio(){
  const status=$("transcribeStatus");
  const btn=$("transcribeBtn");
  if(!state.currentAudioBlob){
    $("error").textContent="Record or import audio before transcribing.";
    return;
  }
  $("error").textContent="";
  btn.disabled=true;
  const started=performance.now();
  try{
    status.className="transcribe-status loading";
    status.querySelector("span:last-child").textContent="Preparing 16 kHz audio…";
    const audio=await decodeAudioToMono16k(state.currentAudioBlob);

    status.querySelector("span:last-child").textContent="Transcribing in a background worker with Whisper…";
    const buffer=audio.buffer;
    const output=await workerTask("transcribe",{audio:buffer},[buffer]);
    const chunks=(output.chunks||[]).filter(x=>x.text?.trim());
    const transcript=chunks.length
      ? chunks.map(c=>`[${formatTime(c.timestamp?.[0]||0)}] ${c.text.trim()}`).join("\n")
      : `[00:00] ${(output.text||"").trim()}`;

    if(!transcript.trim()||transcript.trim()==="[00:00]") throw new Error("Whisper returned an empty transcript.");

    $("transcriptInput").value=transcript;
    await buildIndex({persist:false});
    const elapsed=((performance.now()-started)/1000).toFixed(1);
    status.className="transcribe-status ready";
    status.querySelector("span:last-child").textContent=`Transcript ready in ${elapsed}s · ${chunks.length||1} timestamped segment${(chunks.length||1)===1?"":"s"}.`;
  }catch(err){
    console.error(err);
    status.className="transcribe-status error";
    status.querySelector("span:last-child").textContent="Transcription failed. You can still paste a transcript manually.";
    $("error").textContent=err?.message||"Transcription failed.";
  }finally{
    btn.disabled=!state.currentAudioBlob;
  }
}

/* ---------- transcript / retrieval ---------- */

function renderTimeline(){
  const box=$("timeline");
  box.innerHTML="";
  state.chunks.forEach((c,i)=>{
    const item=document.createElement("div");
    item.className="timeline-item";
    item.dataset.index=i;
    item.innerHTML=`<time>${c.time}</time><p>${escapeHtml(c.text)}</p>`;
    item.addEventListener("click",()=>jumpTo(c.seconds,i));
    box.appendChild(item);
  });
  box.classList.remove("hidden");
}

function jumpTo(seconds,index){
  const p=$("player");
  if(Number.isFinite(seconds) && p.src) p.currentTime=seconds;
  document.querySelectorAll(".timeline-item").forEach(x=>x.classList.remove("active"));
  const active=document.querySelector(`.timeline-item[data-index="${index}"]`);
  active?.classList.add("active");
  active?.scrollIntoView({behavior:"smooth",block:"center"});
}

function generateNotes(){
  if(!state.chunks.length) return;
  const groups=[];
  const size=Math.max(2,Math.ceil(state.chunks.length/4));
  for(let i=0;i<state.chunks.length;i+=size){
    const group=state.chunks.slice(i,i+size);
    groups.push({
      time:group[0].time,
      title:group[0].text.split(/[.!?]/)[0].slice(0,58),
      text:group.map(x=>x.text).join(" "),
    });
  }

  $("notesOutput").innerHTML=`
    <h4>Lecture chapters</h4>
    ${groups.map((g,i)=>`<div class="note-card"><b>${String(i+1).padStart(2,"0")} · ${g.time}</b><p><strong>${escapeHtml(g.title)}</strong></p><p>${escapeHtml(g.text.slice(0,220))}${g.text.length>220?"…":""}</p></div>`).join("")}
  `;

  const cards=state.chunks.slice(0,3).map(c=>{
    const lead=c.text.split(/[.!?]/)[0];
    return [`What was explained at ${c.time}?`,lead];
  });
  $("studyOutput").innerHTML=`<h4>Quick review</h4>${cards.map(([q,a])=>`<div class="study-card"><b>${escapeHtml(q)}</b><p>${escapeHtml(a)}</p></div>`).join("")}`;
}

async function buildIndex({persist=true}={}){
  $("error").textContent="";
  state.chunks=parseTranscript($("transcriptInput").value);
  if(!state.chunks.length){
    $("error").textContent="Add a transcript first.";
    return;
  }
  renderTimeline();
  generateNotes();
  $("searchBtn").disabled=true;
  try{
    state.embeddings=await embedTexts(state.chunks.map(c=>c.text));
  }finally{
    $("searchBtn").disabled=false;
  }
  if(persist && state.currentLectureId) await saveLecture();
}

async function searchLecture(){
  const q=$("question").value.trim();
  if(!q){$("error").textContent="Ask a question first.";return;}
  if(!state.chunks.length) await buildIndex({persist:false});
  if(!state.chunks.length) return;
  $("error").textContent="";

  let queryEmbedding=null;
  if(state.embeddings){
    const em=await embedTexts([q]);
    queryEmbedding=em?.[0]||null;
  }

  const ranked=state.chunks.map((c,i)=>{
    const lex=lexical(q,c.text);
    const sem=queryEmbedding&&state.embeddings?.[i]?cosine(queryEmbedding,state.embeddings[i]):0;
    return {...c,index:i,score:queryEmbedding?(0.86*sem+0.14*lex):lex};
  }).sort((a,b)=>b.score-a.score).slice(0,3);

  $("searchResults").innerHTML=ranked.map((r,n)=>`
    <div class="result-card">
      <div class="result-head"><span>RESULT ${n+1}</span><span class="score">${Math.round(r.score*100)}% relevance</span></div>
      <p>${escapeHtml(r.text)}</p>
      <button class="source-link" data-index="${r.index}" data-time="${r.seconds}" type="button">Source · ${r.time} ↗</button>
    </div>`).join("");

  document.querySelectorAll(".source-link").forEach(btn=>btn.addEventListener("click",()=>jumpTo(Number(btn.dataset.time),Number(btn.dataset.index))));
}

/* ---------- audio ---------- */

function drawIdleWave(){
  const c=$("waveform"),ctx=c.getContext("2d");
  const w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#b9d5ff";
  ctx.lineWidth=3;
  ctx.beginPath();
  for(let x=0;x<w;x+=8){
    const y=h/2+Math.sin(x*.035)*10+Math.sin(x*.009)*16;
    x?ctx.lineTo(x,y):ctx.moveTo(x,y);
  }
  ctx.stroke();
}

function drawLiveWave(){
  if(!state.analyser) return;
  const c=$("waveform"),ctx=c.getContext("2d"),w=c.width,h=c.height;
  const data=new Uint8Array(state.analyser.fftSize);
  state.analyser.getByteTimeDomainData(data);
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#4d6fff";
  ctx.lineWidth=3;
  ctx.beginPath();
  const slice=w/data.length;
  let x=0;
  for(let i=0;i<data.length;i++){
    const y=(data[i]/128)*h/2;
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    x+=slice;
  }
  ctx.stroke();
  state.raf=requestAnimationFrame(drawLiveWave);
}

async function startRecording(){
  try{
    state.stream=await navigator.mediaDevices.getUserMedia({audio:true});
    state.audioChunks=[];
    state.recorder=new MediaRecorder(state.stream);
    state.recorder.ondataavailable=e=>{if(e.data.size)state.audioChunks.push(e.data);};
    state.recorder.onstop=()=>{
      const blob=new Blob(state.audioChunks,{type:state.recorder.mimeType||"audio/webm"});
      if(state.currentAudioUrl) URL.revokeObjectURL(state.currentAudioUrl);
      state.currentAudioBlob=blob;
      state.currentAudioUrl=URL.createObjectURL(blob);
      $("player").src=state.currentAudioUrl;
      $("transcribeBtn").disabled=false;
      $("transcribeStatus").className="transcribe-status";
      $("transcribeStatus").querySelector("span:last-child").textContent="Recording ready — transcribe it locally with Whisper.";
      state.stream.getTracks().forEach(t=>t.stop());
      cancelAnimationFrame(state.raf);
      drawIdleWave();
    };

    state.audioContext=new AudioContext();
    state.source=state.audioContext.createMediaStreamSource(state.stream);
    state.analyser=state.audioContext.createAnalyser();
    state.analyser.fftSize=1024;
    state.source.connect(state.analyser);
    drawLiveWave();

    state.recorder.start();
    state.recordingStartedAt=Date.now();
    $("recordBtn").classList.add("recording");
    $("recordBtn").innerHTML="<span></span> Stop";
    $("recordStatus").textContent="Recording locally";
    state.timerHandle=setInterval(()=>$("timer").textContent=formatTime((Date.now()-state.recordingStartedAt)/1000),250);
  }catch(err){
    $("error").textContent="Microphone access was not available. You can still import audio or use the sample.";
    console.error(err);
  }
}

function stopRecording(){
  state.recorder?.stop();
  clearInterval(state.timerHandle);
  $("recordBtn").classList.remove("recording");
  $("recordBtn").innerHTML="<span></span> Record";
  $("recordStatus").textContent="Recording saved in this session";
}

/* ---------- events ---------- */

$("recordBtn").addEventListener("click",()=>state.recorder?.state==="recording"?stopRecording():startRecording());

$("audioFile").addEventListener("change",e=>{
  const f=e.target.files?.[0];
  if(!f) return;
  if(state.currentAudioUrl) URL.revokeObjectURL(state.currentAudioUrl);
  state.currentAudioBlob=f;
  state.currentAudioUrl=URL.createObjectURL(f);
  $("player").src=state.currentAudioUrl;
  $("recordStatus").textContent=`Imported: ${f.name}`;
  $("transcribeBtn").disabled=false;
  $("transcribeStatus").className="transcribe-status";
  $("transcribeStatus").querySelector("span:last-child").textContent="Audio ready — transcribe it locally with Whisper.";
});

$("transcribeBtn").addEventListener("click",transcribeCurrentAudio);
$("saveLecture").addEventListener("click",saveLecture);
$("sampleTranscript").addEventListener("click",()=>{$("transcriptInput").value=SAMPLE;});
$("parseTranscript").addEventListener("click",()=>buildIndex());
$("searchBtn").addEventListener("click",searchLecture);

$("loadDemo").addEventListener("click",async()=>{
  newLecture(false);
  $("lectureTitle").value="Recursion & call stack";
  $("courseLabel").textContent="CS 101 · SAMPLE LECTURE";
  $("transcriptInput").value=SAMPLE;
  $("question").value="What is a base case?";
  await buildIndex({persist:false});
});

$("loadDemoTop").addEventListener("click",()=>{
  $("product").scrollIntoView({behavior:"smooth"});
  setTimeout(()=>$("loadDemo").click(),450);
});

$("newLecture").addEventListener("click",()=>newLecture());

document.querySelectorAll(".tab").forEach(btn=>btn.addEventListener("click",()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(x=>x.classList.remove("active"));
  btn.classList.add("active");
  $(`tab-${btn.dataset.tab}`).classList.add("active");
}));

window.addEventListener("scroll",()=>{
  const max=document.documentElement.scrollHeight-innerHeight;
  $("progress").style.width=`${max?scrollY/max*100:0}%`;
  document.querySelectorAll(".reveal").forEach(el=>{
    if(el.getBoundingClientRect().top<innerHeight*.88) el.classList.add("visible");
  });
});

window.addEventListener("beforeunload",()=>{
  if(state.currentAudioUrl) URL.revokeObjectURL(state.currentAudioUrl);
});

drawIdleWave();
renderLibrary();
setTimeout(()=>window.dispatchEvent(new Event("scroll")),50);
