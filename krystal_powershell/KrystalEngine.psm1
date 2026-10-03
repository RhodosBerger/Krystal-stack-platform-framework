<#
.SYNOPSIS
    Krystal-Stack PowerShell 5.1 / 7+ Engine & Hardware Orchestrator.
.DESCRIPTION
    Provides high-performance queue scheduling using .NET RunspacePool,
    Win32 VT-100 console streaming, and bidirectional communication with
    the Krystal Localhost Hub and Krystal-Lang Topological VM.
#>

function Enable-KrystalVt100 {
    <#
    .SYNOPSIS
        Enables Win32 VT-100 Virtual Terminal Processing on the standard output handle.
    #>
    [CmdletBinding()]
    param()

    $Definition = @"
    using System;
    using System.Runtime.InteropServices;

    public class WinConsole {
        public const int STD_OUTPUT_HANDLE = -11;
        public const uint ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern IntPtr GetStdHandle(int nStdHandle);

        [DllImport("kernel32.dll")]
        public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);

        [DllImport("kernel32.dll")]
        public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
    }
"@
    try {
        if (-not ([System.Management.Automation.PSTypeName]'WinConsole').Type) {
            Add-Type -TypeDefinition $Definition -ErrorAction SilentlyContinue
        }
        $handle = [WinConsole]::GetStdHandle([WinConsole]::STD_OUTPUT_HANDLE)
        $mode = 0
        if ([WinConsole]::GetConsoleMode($handle, [ref]$mode)) {
            $mode = $mode -bor [WinConsole]::ENABLE_VIRTUAL_TERMINAL_PROCESSING
            [WinConsole]::SetConsoleMode($handle, $mode) | Out-Null
        }
    } catch {
        # Fallback if Add-Type is restricted
    }
}

function Invoke-KrystalRunspaceQueue {
    <#
    .SYNOPSIS
        Demonstrates parallel queue scheduling across CPU cores using .NET RunspacePool.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$false)]
        [int]$WorkerCount = 4,

        [Parameter(Mandatory=$false)]
        [int]$TaskCount = 16
    )

    Write-Host "[KRYSTAL-PS] Initializing .NET RunspacePool with $WorkerCount parallel worker threads..." -ForegroundColor Cyan

    $Pool = [runspacefactory]::CreateRunspacePool(1, $WorkerCount)
    $Pool.Open()

    $Queue = [System.Collections.Concurrent.ConcurrentQueue[object]]::new()
    for ($i = 0; $i -lt $TaskCount; $i++) {
        $Queue.Enqueue(@{ TaskId = $i; Data = "Payload_Stage_$i"; Complexity = ($i * 0.15) })
    }

    $Jobs = [System.Collections.Generic.List[object]]::new()
    $ScriptBlock = {
        param($Q, $WorkerId)
        $Processed = 0
        $Item = $null
        while ($Q.TryDequeue([ref]$Item)) {
            $val = [System.Math]::Sin($Item.TaskId * 0.5) * [System.Math]::Cos($Item.Complexity)
            $Processed++
            [System.Threading.Thread]::Sleep(10)
        }
        return @{ WorkerId = $WorkerId; Processed = $Processed; Result = "SUCCESS" }
    }

    for ($w = 0; $w -lt $WorkerCount; $w++) {
        $PSInstance = [powershell]::Create()
        $PSInstance.RunspacePool = $Pool
        $PSInstance.AddScript($ScriptBlock).AddArgument($Queue).AddArgument($w) | Out-Null
        $AsyncHandle = $PSInstance.BeginInvoke()
        $Jobs.Add(@{ Instance = $PSInstance; Handle = $AsyncHandle })
    }

    $TotalProcessed = 0
    foreach ($j in $Jobs) {
        $Result = $j.Instance.EndInvoke($j.Handle)
        $TotalProcessed += $Result.Processed
        $j.Instance.Dispose()
    }

    $Pool.Close()
    $Pool.Dispose()

    Write-Host "[KRYSTAL-PS] All $TotalProcessed parallel queue tasks completed successfully." -ForegroundColor Green
    return @{ TotalTasks = $TaskCount; Processed = $TotalProcessed; Status = "OPTIMAL" }
}

function Get-KrystalHubStatus {
    <#
    .SYNOPSIS
        Queries the running Localhost Mission Control Hub API.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$false)]
        [string]$Uri = "http://127.0.0.1:8080/api/status"
    )

    try {
        $response = Invoke-RestMethod -Uri $Uri -Method Get -TimeoutSec 3 -ErrorAction Stop
        return $response
    } catch {
        Write-Warning "Localhost hub is unreachable at $Uri. Ensure server.py is running."
        return $null
    }
}

function Invoke-KrystalPromptCompile {
    <#
    .SYNOPSIS
        Compiles a natural language prompt via the Krystal-Stack Open-World Compiler.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,

        [Parameter(Mandatory=$false)]
        [string]$Uri = "http://127.0.0.1:8080/api/openworld/compile"
    )

    $body = @{ prompt = $Prompt } | ConvertTo-Json -Compress
    try {
        $response = Invoke-RestMethod -Uri $Uri -Method Post -Body $body -ContentType "application/json" -TimeoutSec 10 -ErrorAction Stop
        return $response
    } catch {
        Write-Warning "Compile endpoint error: $_"
        return $null
    }
}

Export-ModuleMember -Function Enable-KrystalVt100, Invoke-KrystalRunspaceQueue, Get-KrystalHubStatus, Invoke-KrystalPromptCompile
