param([string]$VideoName='neopax_reverse_ad_cost.mp4',[int]$ExpectedHeight=888)
. (Join-Path $PSScriptRoot 'render.ps1') -FunctionsOnly
try {
    $neopaxVerifyExpression = @'
(async () => {
  const video=document.createElement('video');video.muted=true;video.preload='auto';
  video.src=new URL('neopax_reverse_ad_cost.mp4',location.href).href;
  document.body.appendChild(video);
  const wait=event=>new Promise((resolve,reject)=>{
    const t=setTimeout(()=>reject(Error('Timed out waiting for video '+event)),10000);
    video.addEventListener(event,()=>{clearTimeout(t);resolve();},{once:true});
    video.addEventListener('error',()=>{clearTimeout(t);reject(Error('Video decode error '+video.error?.code));},{once:true});
  });
  await wait('loadeddata');
  if(Math.abs(video.duration-27.56)>.001||video.videoWidth!==1280||video.videoHeight!==888)throw Error('Invalid MP4 metadata');
  const checked=[];
  for(const time of [1,11,20,27]){
    const seek=wait('seeked');video.currentTime=time;await seek;
    if(video.readyState<2||video.error)throw Error('Video did not decode at '+time);
    checked.push(video.currentTime);
  }
  const result={duration:video.duration,width:video.videoWidth,height:video.videoHeight,decodedAtSeconds:checked,error:video.error};
  video.remove();return result;
})()
'@
    $neopaxVerifyExpression = $neopaxVerifyExpression.Replace('neopax_reverse_ad_cost.mp4',$VideoName).Replace('video.videoHeight!==888',('video.videoHeight!==' + $ExpectedHeight))
    $neopaxVerification = Invoke-NeopaxJs $neopaxVerifyExpression
    $neopaxVerification | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $PSScriptRoot ($VideoName.Replace('.mp4','_verification.json')))
    Write-Output ($neopaxVerification | ConvertTo-Json -Compress)
    Invoke-NeopaxCdp 'Browser.close' | Out-Null
} finally { $script:neopaxSocket.Dispose() }
