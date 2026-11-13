. (Join-Path $PSScriptRoot 'render.ps1') -FunctionsOnly
$neopaxArrowsDir = Join-Path $PSScriptRoot 'arrows'
New-Item -ItemType Directory -Force -Path $neopaxArrowsDir | Out-Null
try {
    Invoke-NeopaxCdp 'Page.enable' | Out-Null
    Invoke-NeopaxCdp 'Emulation.setDeviceMetricsOverride' @{width=1328;height=2400;deviceScaleFactor=1;mobile=$false} | Out-Null
    $neopaxUrl = ([uri](Join-Path $neopaxRepo 'docs/visualizations/neopax_reverse_ad_cost.html')).AbsoluteUri + '?style=arrows&frame=0'
    Invoke-NeopaxCdp 'Page.navigate' @{url=$neopaxUrl} | Out-Null
    Invoke-NeopaxJs 'new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{if(window.NEOPAXCostUI&&NEOPAXCost.drawArrows){clearInterval(t);resolve(true);}else if(++n>100){clearInterval(t);reject(Error("Arrow renderer not loaded"));}},30);})' | Out-Null
    $neopaxCheck = Invoke-NeopaxJs @'
(() => {
  let frames=0;
  for(const size of [2,3,4,6,12]){
    const model=NEOPAXCost.timeline(size);
    for(let i=0;i<model.events.length;i++){
      const svg=NEOPAXCost.drawArrows(model,i);
      if(/NaN|undefined/.test(svg))throw Error('Invalid minimal frame');
      const state=NEOPAXCost.stateAt(model,i),dom=new DOMParser().parseFromString(svg,'image/svg+xml');
      if(dom.querySelector('parsererror'))throw Error('Invalid SVG XML');
      const labels=[...dom.querySelectorAll('text')].map(t=>t.textContent);
      if(labels.filter(t=>t.startsWith('Segment ')).length!==model.k)throw Error('Missing segment labels');
      if(labels.filter(t=>t.startsWith('Checkpoint ')).length!==model.k-1)throw Error('Missing checkpoint labels');
      const acceptedDots=[...dom.querySelectorAll('circle[data-accepted-step]')];
      if(acceptedDots.length!==state.forward)throw Error('Accepted dots did not follow forward completion');
      if(acceptedDots.some((dot,j)=>Number(dot.getAttribute('data-accepted-step'))!==j+1))throw Error('Accepted dots have incorrect step indices');
      const greenRecords=dom.querySelectorAll('g[data-lane="replay"] path[stroke="#00b44b"]').length;
      if(greenRecords!==state.buffer.length)throw Error('Replay drawing does not match temporary buffer');
      for(const segment of state.completed){
        for(let step=segment*size+1;step<=(segment+1)*size;step++){
          if(dom.querySelector(`g[data-lane="replay"][data-step="${step}"]`))throw Error('Completed replay segment did not disappear');
        }
      }
      frames++;
    }
  }
  NEOPAXCostUI.setStyle('arrows');NEOPAXCostUI.setFrame(0);
  return {framesChecked:frames,style:document.getElementById('appearance').value,transportSolves:0};
})()
'@
    Write-Output ($neopaxCheck | ConvertTo-Json -Compress)
    $neopaxMetadata = Invoke-NeopaxJs 'NEOPAXCostUI.model().events.map((e,i)=>({index:i,kind:e.kind,segment:e.segment??null}))'
    $neopaxMetadata | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $neopaxArrowsDir 'frames.json')
    foreach ($neopaxFrame in $neopaxMetadata) {
        Invoke-NeopaxJs ('NEOPAXCostUI.setFrame(' + $neopaxFrame.index + '); true') | Out-Null
        Save-NeopaxScreenshot (Join-Path $neopaxArrowsDir ('frame_{0:D3}.png' -f [int]$neopaxFrame.index))
    }
    # Poster: middle segment active, last segment already released.
    $neopaxPoster = 30
    Invoke-NeopaxJs ('NEOPAXCostUI.setFrame(' + $neopaxPoster + '); true') | Out-Null
    Save-NeopaxScreenshot (Join-Path $neopaxRepo 'docs/visualizations/neopax_reverse_ad_arrows.png')
    $neopaxSvg = Invoke-NeopaxJs ('NEOPAXCost.drawArrows(NEOPAXCostUI.model(),' + $neopaxPoster + ')')
    [IO.File]::WriteAllText((Join-Path $neopaxRepo 'docs/visualizations/neopax_reverse_ad_arrows.svg'),$neopaxSvg,[Text.UTF8Encoding]::new($false))
    Invoke-NeopaxJs 'globalThis.NEOPAXVideoExportOptions={style:"arrows",width:1280,height:588};true' | Out-Null
    $neopaxVideo = Invoke-NeopaxJs ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'encode_video.js')))
    $neopaxVideo | ConvertTo-Json -Depth 8 -Compress | Set-Content -Encoding UTF8 -Path (Join-Path $neopaxArrowsDir 'h264_samples.json')
    Write-Output ('Exported ' + $neopaxMetadata.Count + ' frames; H.264 duration ' + ($neopaxVideo.duration / 1000000) + ' seconds.')
} finally { $script:neopaxSocket.Dispose() }
