(async () => {
  const options=globalThis.NEOPAXVideoExportOptions??{};
  const config = {
    codec: 'avc1.420028', width: options.width??1280, height: options.height??888, bitrate: 2800000,
    framerate: 25, latencyMode: 'realtime', avc: {format: 'avc'}
  };
  const supported = await VideoEncoder.isConfigSupported(config);
  if (!supported.supported) throw Error('H.264 encoder unavailable in this browser');
  const base64 = value => {
    const a = new Uint8Array(value); let s='';
    for (let i=0;i<a.length;i+=8192) s+=String.fromCharCode(...a.subarray(i,i+8192));
    return btoa(s);
  };
  const samples=[]; let description=null, encoderError=null;
  const encoder = new VideoEncoder({
    output(chunk,metadata) {
      if(metadata.decoderConfig?.description) description=base64(metadata.decoderConfig.description);
      const bytes=new Uint8Array(chunk.byteLength);chunk.copyTo(bytes);
      samples.push({type:chunk.type,timestamp:chunk.timestamp,duration:chunk.duration,data:base64(bytes)});
    },
    error(error){encoderError=String(error);}
  });
  encoder.configure(config);
  const canvas=document.createElement('canvas');canvas.width=config.width;canvas.height=config.height;
  const context=canvas.getContext('2d',{alpha:false});
  const model=NEOPAXCost.timeline(4);
  const render=options.style==='arrows'?NEOPAXCost.drawArrows:NEOPAXCost.draw;
  const durations={start:1600,forward:320,objective:1400,restore:900,replay:480,adjoint:480,release:1000,done:3500};
  let timestamp=0;
  for(let i=0;i<model.events.length;i++){
    const svg=render(model,i),url=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'}));
    const image=new Image();image.src=url;await image.decode();
    context.fillStyle='#fffefa';context.fillRect(0,0,canvas.width,canvas.height);
    context.drawImage(image,0,0,canvas.width,canvas.height);URL.revokeObjectURL(url);
    const duration=durations[model.events[i].kind]*1000;
    const frame=new VideoFrame(canvas,{timestamp,duration});
    encoder.encode(frame,{keyFrame:i%12===0});frame.close();timestamp+=duration;
    if(encoder.encodeQueueSize>4)await encoder.flush();
    if(encoderError)throw Error(encoderError);
  }
  await encoder.flush();encoder.close();
  if(encoderError)throw Error(encoderError);
  if(!description||samples.length!==model.events.length)throw Error('Incomplete video export');
  return {width:config.width,height:config.height,codec:config.codec,description,samples,duration:timestamp};
})()
