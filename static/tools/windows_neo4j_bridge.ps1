# WSL -> Windows localhost HTTP bridge. Credentials arrive on stdin, never argv.
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
try {
    $inputText = [Console]::In.ReadToEnd()
    $request = $inputText | ConvertFrom-Json
    $pair = '{0}:{1}' -f $request.username, $request.password
    $auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pair))
    $body = @{ statements = $request.statements } | ConvertTo-Json -Depth 100 -Compress
    $headers = @{ Authorization = "Basic $auth" }
    $result = Invoke-RestMethod -Uri $request.url -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 30
    $result | ConvertTo-Json -Depth 100 -Compress
} catch {
    $status = $null
    if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
    $codes = @()
    try {
        $errorBody = $_.ErrorDetails.Message | ConvertFrom-Json
        $codes = @($errorBody.errors | ForEach-Object { $_.code })
    } catch {}
    @{ bridge_error = 'Neo4jHttpError'; status = $status; codes = $codes; message = 'Windows Neo4j HTTP request failed; check endpoint and credentials.' } | ConvertTo-Json -Compress
    exit 1
}
