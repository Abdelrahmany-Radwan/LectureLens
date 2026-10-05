import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

let extractor = null;
let transcriber = null;

async function getExtractor(){
  if(!extractor){
    extractor = await pipeline("feature-extraction","Xenova/all-MiniLM-L6-v2");
  }
  return extractor;
}

async function getTranscriber(){
  if(!transcriber){
    transcriber = await pipeline(
      "automatic-speech-recognition",
      "Xenova/whisper-tiny.en",
      { quantized: true }
    );
  }
  return transcriber;
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;
  try{
    if(type === "embed"){
      const model = await getExtractor();
      const vectors = [];
      for(const text of payload.texts){
        const result = await model(text,{pooling:"mean",normalize:true});
        vectors.push(Array.from(result.data));
      }
      self.postMessage({id,ok:true,result:vectors});
      return;
    }

    if(type === "transcribe"){
      const model = await getTranscriber();
      const audio = new Float32Array(payload.audio);
      const result = await model(audio,{
        return_timestamps:true,
        chunk_length_s:30,
        stride_length_s:5,
      });
      self.postMessage({id,ok:true,result});
      return;
    }

    throw new Error("Unknown worker task.");
  }catch(error){
    self.postMessage({id,ok:false,error:error?.message||String(error)});
  }
};
