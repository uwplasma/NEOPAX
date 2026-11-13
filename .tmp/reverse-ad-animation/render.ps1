param([switch]$FunctionsOnly)
$ErrorActionPreference = 'Stop'
$neopaxRepo = 'C:/Users/eduar/Github_Codes/NEOPAX'
$neopaxRenderDir = Join-Path $neopaxRepo '.tmp/reverse-ad-animation'
$neopaxPages = Invoke-RestMethod -Uri 'http://127.0.0.1:9237/json/list' -TimeoutSec 5
$neopaxPage = $neopaxPages[0]
$script:neopaxSocket = [System.Net.WebSockets.ClientWebSocket]::new()
$script:neopaxSocket.ConnectAsync([uri]$neopaxPage.webSocketDebuggerUrl,[Threading.CancellationToken]::None).GetAwaiter().GetResult()
$script:neopaxSequence = 0
function Invoke-NeopaxCdp([string]$Method, [hashtable]$Parameters = @{}) {
    $script:neopaxSequence++
    $neopaxRequest = @{id=$script:neopaxSequence; method=$Method; params=$Parameters} | ConvertTo-Json -Depth 30 -Compress
    $neopaxBytes = [Text.Encoding]::UTF8.GetBytes($neopaxRequest)
    $neopaxCancellation = [Threading.CancellationTokenSource]::new(15000)
    try {
        $script:neopaxSocket.SendAsync([ArraySegment[byte]]::new($neopaxBytes),[Net.WebSockets.WebSocketMessageType]::Text,$true,$neopaxCancellation.Token).GetAwaiter().GetResult()
        while ($true) {
            $neopaxStream = [IO.MemoryStream]::new()
            $neopaxBuffer = [byte[]]::new(1048576)
            do {
                $neopaxReceived = $script:neopaxSocket.ReceiveAsync([ArraySegment[byte]]::new($neopaxBuffer),$neopaxCancellation.Token).GetAwaiter().GetResult()
                $neopaxStream.Write($neopaxBuffer,0,$neopaxReceived.Count)
            } until ($neopaxReceived.EndOfMessage)
            $neopaxResponse = [Text.Encoding]::UTF8.GetString($neopaxStream.ToArray()) | ConvertFrom-Json
            $neopaxStream.Dispose()
            if ($neopaxResponse.id -eq $script:neopaxSequence) {
                if ($neopaxResponse.error) { throw ($neopaxResponse.error | ConvertTo-Json -Compress) }
                return $neopaxResponse.result
            }
        }
    } finally { $neopaxCancellation.Dispose() }
}
function Invoke-NeopaxJs([string]$Expression) {
    $neopaxResult = Invoke-NeopaxCdp 'Runtime.evaluate' @{expression=$Expression; returnByValue=$true; awaitPromise=$true}
    if ($neopaxResult.exceptionDetails) { throw ($neopaxResult.exceptionDetails | ConvertTo-Json -Depth 8) }
    return $neopaxResult.result.value
}
function Save-NeopaxScreenshot([string]$Path, [bool]$DiagramOnly = $true) {
    $neopaxCapture = @{format='png'; captureBeyondViewport=$true}
    if ($DiagramOnly) {
        $neopaxCapture.clip = Invoke-NeopaxJs '(() => { const b=document.getElementById("diagram").getBoundingClientRect(); return {x:b.x,y:b.y+scrollY,width:b.width,height:b.height,scale:1}; })()'
    }
    $neopaxPng = Invoke-NeopaxCdp 'Page.captureScreenshot' $neopaxCapture
    [IO.File]::WriteAllBytes($Path,[Convert]::FromBase64String($neopaxPng.data))
}
if ($FunctionsOnly) { return }
try {
    Invoke-NeopaxCdp 'Page.enable' | Out-Null
    Invoke-NeopaxCdp 'Emulation.setDeviceMetricsOverride' @{width=1340;height=1250;deviceScaleFactor=1;mobile=$false} | Out-Null
    $neopaxUrl = ([uri](Join-Path $neopaxRepo 'docs/visualizations/neopax_reverse_ad_cost.html')).AbsoluteUri + '?frame=0'
    Invoke-NeopaxCdp 'Page.navigate' @{url=$neopaxUrl} | Out-Null
    $neopaxReady = Invoke-NeopaxJs 'new Promise((resolve,reject) => {let n=0; const t=setInterval(()=>{if(window.NEOPAXCostUI){clearInterval(t);resolve(true);}else if(++n>100){clearInterval(t);reject(Error("Animation not loaded"));}},30);})'
    $neopaxChecks = Invoke-NeopaxJs 'NEOPAXCost.check()'
    $neopaxChecks | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $neopaxRenderDir 'checks.json')
    Write-Output ($neopaxChecks | ConvertTo-Json -Depth 8 -Compress)
    # Check controls, segment changes, and frame rendering before exporting.
    $neopaxUiCheck = Invoke-NeopaxJs @'
(() => {
  const assert=(v,m)=>{if(!v)throw Error(m);};
  const d=document;
  const segment=d.getElementById('segment');
  let frames=0;
  for(const v of ['2','3','4','6','12']){
    segment.value=v; segment.dispatchEvent(new Event('change'));
    const model=NEOPAXCostUI.model();
    for(let i=0;i<model.events.length;i++){
      NEOPAXCostUI.setFrame(i);
      assert(!/NaN|undefined/.test(d.getElementById('diagram').innerHTML),'Invalid rendered SVG');
      frames++;
    }
  }
  segment.value='4';segment.dispatchEvent(new Event('change'));NEOPAXCostUI.setFrame(0);
  d.getElementById('next').click();assert(+d.getElementById('position').value===1,'Next button');
  d.getElementById('previous').click();assert(+d.getElementById('position').value===0,'Previous button');
  d.getElementById('jump').click();assert(NEOPAXCostUI.model().events[+d.getElementById('position').value].kind==='restore','Jump button');
  const slider=d.getElementById('position');slider.value='19';slider.dispatchEvent(new Event('input'));assert(+slider.value===19,'Scrubber');
  d.getElementById('play').click();assert(NEOPAXCostUI.playing(),'Play button');
  d.getElementById('play').click();assert(!NEOPAXCostUI.playing(),'Pause button');
  return {renderedFrames:frames,controls:'passed',externalResources:performance.getEntriesByType('resource').filter(r=>!r.name.startsWith('file:')).length};
})()
'@
    Write-Output ($neopaxUiCheck | ConvertTo-Json -Compress)
    $neopaxMetadata = Invoke-NeopaxJs 'NEOPAXCostUI.model().events.map((e,i)=>({index:i,kind:e.kind,segment:e.segment??null}))'
    $neopaxMetadata | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $neopaxRenderDir 'frames.json')
    foreach ($neopaxFrame in $neopaxMetadata) {
        Invoke-NeopaxJs ('NEOPAXCostUI.setFrame(' + $neopaxFrame.index + '); true') | Out-Null
        $neopaxFramePath = Join-Path $neopaxRenderDir ('frame_{0:D3}.png' -f [int]$neopaxFrame.index)
        Save-NeopaxScreenshot $neopaxFramePath
    }
    Invoke-NeopaxJs 'NEOPAXCostUI.setFrame(21); true' | Out-Null
    Save-NeopaxScreenshot (Join-Path $neopaxRepo 'docs/visualizations/neopax_reverse_ad_cost.png')
    Save-NeopaxScreenshot (Join-Path $neopaxRenderDir 'desktop.png') $false
    Invoke-NeopaxCdp 'Emulation.setDeviceMetricsOverride' @{width=390;height=844;deviceScaleFactor=1;mobile=$true} | Out-Null
    $neopaxMobile = Invoke-NeopaxJs '({viewport:innerWidth,document:document.documentElement.scrollWidth,svg:!!document.querySelector("#diagram svg")})'
    if ($neopaxMobile.document -gt $neopaxMobile.viewport) { throw 'Horizontal overflow at mobile width' }
    Save-NeopaxScreenshot (Join-Path $neopaxRenderDir 'mobile.png') $false
    Write-Output ('Mobile: ' + ($neopaxMobile | ConvertTo-Json -Compress))
    Write-Output ('Exported ' + $neopaxMetadata.Count + ' diagram frames.')
} finally {
    $script:neopaxSocket.Dispose()
}
