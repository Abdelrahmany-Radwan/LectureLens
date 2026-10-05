import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

const $ = (id) => document.getElementById(id);
const state = {
  chunks: [],
  embeddings: null,
  extractor: null,
  recorder: null,
  stream: null,
  audioChunks: [],
  recordingStartedAt: null,
  timerHandle: null,
  raf: null,
  audioContext: null,
  analyser: null,
  source: null,
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
  return h? `${h.toString().padStart(2,"0")}:${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}`:
    `${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}`;
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
  let hit=0; for(const x of A) if(B.has(x)) hit++;
  return hit/Math.sqrt(A.size*B.size);
}
function cosine(a,b){
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  return dot/(Math.sqrt(na)*Math.sqrt(nb)||1);
}
async function ensureModel(){
  if(state.extractor) return state.extractor;
  $("modelStatus").innerHTML="<i></i> loading MiniLM…";
  try{
    state.extractor=await pipeline("feature-extraction","Xenova/all-MiniLM-L6-v2");
    $("modelStatus").classList.add("ready");
    $("modelStatus").innerHTML="<i></i> MiniLM ready";
    return state.extractor;
  }catch(err){
    console.warn(err);
    $("modelStatus").innerHTML="<i></i> lexical fallback";
    return null;
  }
}
async function embedTexts(texts){
  const model=await ensureModel();
  if(!model) return null;
  const out=[];
  for(const text of texts){
    const result=await model(text,{pooling:"mean",normalize:true});
    out.push(Array.from(result.data));
  }
  return out;
}
function renderTimeline(){
  const box=$("timeline");
  box.innerHTML="";
  state.chunks.forEach((c,i)=>{
    const item=document.createElement("div");
    item.className="timeline-item";
    item.dataset.index=i;
    item.innerHTML=`<time>${c.time}</time><p>${c.text}</p>`;
    item.addEventListener("click",()=>jumpTo(c.seconds,i));
    box.appendChild(item);
  });
  box.classList.remove("hidden");
}
function jumpTo(seconds,index){
  const p=$("player");
  if(Number.isFinite(seconds)) p.currentTime=seconds;
  document.querySelectorAll(".timeline-item").forEach(x=>x.classList.remove("active"));
  const active=document.querySelector(`.timeline-item[data-index="${index}"]`);
  active?.classList.add("active");
  active?.scrollIntoView({behavior:"smooth",block:"center"});
}
function generateNotes(){
  if(!state.chunks.length) return;
  const first=state.chunks.slice(0,6);
  $("notesOutput").innerHTML=`
    <h4>Lecture outline</h4>
    ${first.map((c,i)=>`<div class="note-card"><b>${String(i+1).padStart(2,"0")} · ${c.time}</b><p>${c.text}</p></div>`).join("")}
  `;
  const terms=[
    ["Base case","The stopping condition that prevents recursion from continuing forever."],
    ["Call stack","Tracks unfinished function calls while recursion runs."],
    ["Recursive case","Reduces the problem and calls the function again with a smaller input."]
  ];
  $("studyOutput").innerHTML=`<h4>Quick review</h4>${terms.map(([q,a])=>`<div class="study-card"><b>${q}</b><p>${a}</p></div>`).join("")}`;
}
async function buildIndex(){
  $("error").textContent="";
  state.chunks=parseTranscript($("transcriptInput").value);
  if(!state.chunks.length){$("error").textContent="Add a transcript first.";return;}
  renderTimeline();
  generateNotes();
  localStorage.setItem("lecturelens:lastTranscript",$("transcriptInput").value);
  $("searchBtn").disabled=true;
  try{
    state.embeddings=await embedTexts(state.chunks.map(c=>c.text));
  } finally {
    $("searchBtn").disabled=false;
  }
}
async function searchLecture(){
  const q=$("question").value.trim();
  if(!q){$("error").textContent="Ask a question first.";return;}
  if(!state.chunks.length) await buildIndex();
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
    return {...c,index:i,lex,sem,score:queryEmbedding?(0.86*sem+0.14*lex):lex};
  }).sort((a,b)=>b.score-a.score).slice(0,3);

  $("searchResults").innerHTML=ranked.map((r,n)=>`
    <div class="result-card">
      <div class="result-head"><span>RESULT ${n+1}</span><span class="score">${Math.round(r.score*100)}% relevance</span></div>
      <p>${r.text}</p>
      <button class="source-link" data-index="${r.index}" data-time="${r.seconds}" type="button">Source · ${r.time} ↗</button>
    </div>`).join("");
  document.querySelectorAll(".source-link").forEach(btn=>btn.addEventListener("click",()=>jumpTo(Number(btn.dataset.time),Number(btn.dataset.index))));
}
function drawIdleWave(){
  const c=$("waveform"), ctx=c.getContext("2d");
  const w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle="#b9d5ff";ctx.lineWidth=3;ctx.beginPath();
  for(let x=0;x<w;x+=8){const y=h/2+Math.sin(x*.035)*10+Math.sin(x*.009)*16; x?ctx.lineTo(x,y):ctx.moveTo(x,y);}
  ctx.stroke();
}
function drawLiveWave(){
  if(!state.analyser) return;
  const c=$("waveform"),ctx=c.getContext("2d"),w=c.width,h=c.height;
  const data=new Uint8Array(state.analyser.fftSize);state.analyser.getByteTimeDomainData(data);
  ctx.clearRect(0,0,w,h);ctx.strokeStyle="#4d6fff";ctx.lineWidth=3;ctx.beginPath();
  const slice=w/data.length;let x=0;
  for(let i=0;i<data.length;i++){const y=(data[i]/128)*h/2;i?ctx.lineTo(x,y):ctx.moveTo(x,y);x+=slice;}ctx.stroke();
  state.raf=requestAnimationFrame(drawLiveWave);
}
async function startRecording(){
  try{
    state.stream=await navigator.mediaDevices.getUserMedia({audio:true});
    state.audioChunks=[];
    state.recorder=new MediaRecorder(state.stream);
    state.recorder.ondataavailable=e=>{if(e.data.size)state.audioChunks.push(e.data)};
    state.recorder.onstop=()=>{
      const blob=new Blob(state.audioChunks,{type:state.recorder.mimeType||"audio/webm"});
      $("player").src=URL.createObjectURL(blob);
      state.stream.getTracks().forEach(t=>t.stop());
      cancelAnimationFrame(state.raf);drawIdleWave();
    };
    state.audioContext=new AudioContext();
    state.source=state.audioContext.createMediaStreamSource(state.stream);
    state.analyser=state.audioContext.createAnalyser();state.analyser.fftSize=1024;state.source.connect(state.analyser);drawLiveWave();
    state.recorder.start();
    state.recordingStartedAt=Date.now();
    $("recordBtn").classList.add("recording");$("recordBtn").innerHTML="<span></span> Stop";
    $("recordStatus").textContent="Recording locally";
    state.timerHandle=setInterval(()=>$("timer").textContent=formatTime((Date.now()-state.recordingStartedAt)/1000),250);
  }catch(err){$("error").textContent="Microphone access was not available. You can still import audio or use the sample.";console.error(err);}
}
function stopRecording(){
  state.recorder?.stop();clearInterval(state.timerHandle);
  $("recordBtn").classList.remove("recording");$("recordBtn").innerHTML="<span></span> Record";
  $("recordStatus").textContent="Recording saved in this session";
}
$("recordBtn").addEventListener("click",()=>state.recorder?.state==="recording"?stopRecording():startRecording());
$("audioFile").addEventListener("change",e=>{const f=e.target.files?.[0];if(!f)return;$("player").src=URL.createObjectURL(f);$("recordStatus").textContent=`Imported: ${f.name}`;});
$("sampleTranscript").addEventListener("click",()=>{$("transcriptInput").value=SAMPLE;});
$("parseTranscript").addEventListener("click",buildIndex);
$("searchBtn").addEventListener("click",searchLecture);
$("loadDemo").addEventListener("click",async()=>{$("lectureTitle").value="Recursion & call stack";$("courseLabel").textContent="CS 101 · SAMPLE LECTURE";$("transcriptInput").value=SAMPLE;$("question").value="What is a base case?";await buildIndex();});
$("loadDemoTop").addEventListener("click",()=>{$("product").scrollIntoView({behavior:"smooth"});setTimeout(()=>$("loadDemo").click(),450);});
$("newLecture").addEventListener("click",()=>{$("lectureTitle").value="Untitled lecture";$("transcriptInput").value="";$("timeline").innerHTML="";$("timeline").classList.add("hidden");state.chunks=[];state.embeddings=null;$("searchResults").innerHTML='<div class="placeholder-card"><b>New lecture ready.</b><p>Record audio or paste a timestamped transcript to begin.</p></div>';});
document.querySelectorAll(".tab").forEach(btn=>btn.addEventListener("click",()=>{document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));document.querySelectorAll(".tab-panel").forEach(x=>x.classList.remove("active"));btn.classList.add("active");$(`tab-${btn.dataset.tab}`).classList.add("active");}));
window.addEventListener("scroll",()=>{const max=document.documentElement.scrollHeight-innerHeight;$("progress").style.width=`${max?scrollY/max*100:0}%`;document.querySelectorAll(".reveal").forEach(el=>{if(el.getBoundingClientRect().top<innerHeight*.88)el.classList.add("visible")})});
const saved=localStorage.getItem("lecturelens:lastTranscript");if(saved)$("transcriptInput").value=saved;
drawIdleWave();
setTimeout(()=>window.dispatchEvent(new Event("scroll")),50);
