. (Join-Path $PSScriptRoot 'render.ps1') -FunctionsOnly
$neopaxSnapshotDir = Join-Path $neopaxRepo 'docs/visualizations/neopax_reverse_ad_snapshots'
New-Item -ItemType Directory -Force -Path $neopaxSnapshotDir | Out-Null
$neopaxSnapshots = @(
    @{frame=12;stem='01_forward_completed';title='1. Forward run completed: rejected trials, accepted-step dots and checkpoints'},
    @{frame=22;stem='02_segment_3_backward';title='2. Segment 3 replayed; backward sweep completed through segment 3'},
    @{frame=32;stem='03_segment_2_backward';title='3. Segment 2 replayed; backward sweep completed through segments 3 and 2'},
    @{frame=42;stem='04_segment_1_full_backward';title='4. Segment 1 replayed; full backward sweep completed'}
)
try {
    Invoke-NeopaxCdp 'Page.enable' | Out-Null
    $neopaxUrl = ([uri](Join-Path $neopaxRepo 'docs/visualizations/neopax_reverse_ad_cost.html')).AbsoluteUri + '?style=arrows&frame=0'
    Invoke-NeopaxCdp 'Page.navigate' @{url=$neopaxUrl} | Out-Null
    Invoke-NeopaxJs 'new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{if(window.NEOPAXCost&&NEOPAXCost.drawArrows){clearInterval(t);resolve(true);}else if(++n>100){clearInterval(t);reject(Error("Animation renderer not loaded"));}},30);})' | Out-Null
    $neopaxSvgFrames = @()
    $neopaxManifest = @()
    foreach ($neopaxSnapshot in $neopaxSnapshots) {
        $neopaxFrame = [int]$neopaxSnapshot.frame
        $neopaxState = Invoke-NeopaxJs ('(() => {const m=NEOPAXCost.timeline(4),s=NEOPAXCost.stateAt(m,' + $neopaxFrame + ');return {frame:s.index,event:s.event,forwardSteps:s.forward,replayedSteps:s.replay,adjointSteps:s.adjoint,activeSegment:s.segment===null?null:s.segment+1,retainedStageRecords:s.buffer.length,completedSegments:s.completed.map(k=>k+1)};})()')
        if ($neopaxState.forwardSteps -ne 12) { throw 'Snapshot forward run is incomplete' }
        if ($neopaxFrame -eq 12 -and ($neopaxState.replayedSteps -ne 0 -or $neopaxState.adjointSteps -ne 0)) { throw 'First snapshot must precede replay' }
        if ($neopaxFrame -gt 12 -and $neopaxState.replayedSteps -ne $neopaxState.adjointSteps) { throw 'Snapshot must show completed local backward sweep' }
        if ($neopaxFrame -eq 42 -and $neopaxState.adjointSteps -ne 12) { throw 'Last snapshot must show full backward sweep' }
        $neopaxSvg = Invoke-NeopaxJs ('NEOPAXCost.drawArrows(NEOPAXCost.timeline(4),' + $neopaxFrame + ')')
        [IO.File]::WriteAllText((Join-Path $neopaxSnapshotDir ($neopaxSnapshot.stem + '.svg')),$neopaxSvg,[Text.UTF8Encoding]::new($false))
        $neopaxSavedFrame = Join-Path $PSScriptRoot ('arrows/frame_{0:D3}.png' -f $neopaxFrame)
        Copy-Item -LiteralPath $neopaxSavedFrame -Destination (Join-Path $neopaxSnapshotDir ($neopaxSnapshot.stem + '.png'))
        $neopaxSvgFrames += $neopaxSvg
        $neopaxManifest += @{title=$neopaxSnapshot.title;png=($neopaxSnapshot.stem+'.png');svg=($neopaxSnapshot.stem+'.svg');state=$neopaxState}
    }
    # A vector contact sheet gives a compact overview without altering the
    # individual snapshots or the movie's existing white-background style.
    $neopaxContact = '<svg xmlns="http://www.w3.org/2000/svg" width="2560" height="1280" viewBox="0 0 2560 1280"><rect width="2560" height="1280" fill="white"/>'
    $neopaxPages = ''
    for ($neopaxIndex=0;$neopaxIndex -lt 4;$neopaxIndex++) {
        $neopaxCaption = [System.Net.WebUtility]::HtmlEncode($neopaxSnapshots[$neopaxIndex].title)
        $neopaxX = ($neopaxIndex % 2)*1280
        $neopaxY = [Math]::Floor($neopaxIndex/2)*640
        $neopaxUniqueSvg = $neopaxSvgFrames[$neopaxIndex].Replace('arrow-title',('arrow-title-'+$neopaxIndex)).Replace('arrow-description',('arrow-description-'+$neopaxIndex))
        $neopaxContact += '<g transform="translate('+$neopaxX+','+$neopaxY+')"><text x="38" y="35" font-family="Arial,sans-serif" font-size="23" fill="#151515">'+$neopaxCaption+'</text><g transform="translate(0,52)">'+$neopaxUniqueSvg+'</g></g>'
        $neopaxPages += '<section><h1>'+$neopaxCaption+'</h1>'+$neopaxUniqueSvg+'</section>'
    }
    $neopaxContact += '</svg>'
    [IO.File]::WriteAllText((Join-Path $neopaxSnapshotDir 'all_four_snapshots.svg'),$neopaxContact,[Text.UTF8Encoding]::new($false))
    # Print native SVG to PDF so text and arrows remain vector graphics.
    $neopaxPdfHtml = '<!doctype html><html><head><meta charset="utf-8"><style>@page{size:13.333333in 6.666667in;margin:0}*{box-sizing:border-box}html,body{margin:0;background:white}section{width:1280px;height:640px;break-after:page;overflow:hidden}section:last-child{break-after:auto}h1{height:52px;margin:0;padding:12px 38px 0;font:23px Arial,sans-serif;font-weight:400;color:#151515}svg{display:block}</style></head><body>'+$neopaxPages+'</body></html>'
    $neopaxDocument = Invoke-NeopaxCdp 'Page.getFrameTree'
    Invoke-NeopaxCdp 'Page.setDocumentContent' @{frameId=$neopaxDocument.frameTree.frame.id;html=$neopaxPdfHtml} | Out-Null
    $neopaxPdf = Invoke-NeopaxCdp 'Page.printToPDF' @{printBackground=$true;preferCSSPageSize=$true;displayHeaderFooter=$false;marginTop=0;marginBottom=0;marginLeft=0;marginRight=0}
    [IO.File]::WriteAllBytes((Join-Path $neopaxSnapshotDir 'all_four_snapshots.pdf'),[Convert]::FromBase64String($neopaxPdf.data))
    # Render the 2x2 contact sheet using the same headless browser.
    $neopaxContactHtml = '<!doctype html><html><head><style>html,body{margin:0;width:2560px;height:1280px;background:white;overflow:hidden}svg{display:block}</style></head><body>'+$neopaxContact+'</body></html>'
    Invoke-NeopaxCdp 'Emulation.setDeviceMetricsOverride' @{width=2560;height=1280;deviceScaleFactor=1;mobile=$false} | Out-Null
    Invoke-NeopaxCdp 'Page.setDocumentContent' @{frameId=$neopaxDocument.frameTree.frame.id;html=$neopaxContactHtml} | Out-Null
    $neopaxPng = Invoke-NeopaxCdp 'Page.captureScreenshot' @{format='png';captureBeyondViewport=$true;clip=@{x=0;y=0;width=2560;height=1280;scale=1}}
    [IO.File]::WriteAllBytes((Join-Path $neopaxSnapshotDir 'all_four_snapshots.png'),[Convert]::FromBase64String($neopaxPng.data))
    $neopaxManifest | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 (Join-Path $neopaxSnapshotDir 'snapshots.json')
    Write-Output ($neopaxManifest | ConvertTo-Json -Depth 8 -Compress)
    Invoke-NeopaxCdp 'Browser.close' | Out-Null
} finally { $script:neopaxSocket.Dispose() }
