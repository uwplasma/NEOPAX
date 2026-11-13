. (Join-Path $PSScriptRoot 'render.ps1') -FunctionsOnly
try {
    $neopaxSupported = Invoke-NeopaxJs 'VideoEncoder.isConfigSupported({codec:"avc1.420028",width:1280,height:888,bitrate:2800000,framerate:25,avc:{format:"avc"}})'
    Write-Output ('H.264 support: ' + ($neopaxSupported | ConvertTo-Json -Depth 5 -Compress))
    $neopaxVideo = Invoke-NeopaxJs ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'encode_video.js')))
    $neopaxVideo | ConvertTo-Json -Depth 8 -Compress | Set-Content -Encoding UTF8 -Path (Join-Path $PSScriptRoot 'h264_samples.json')
    Write-Output ('Encoded ' + $neopaxVideo.samples.Count + ' frames, duration ' + ($neopaxVideo.duration / 1000000) + ' seconds.')
} finally { $script:neopaxSocket.Dispose() }
