# demolab installer (Windows / PowerShell).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://demolab.eoinmurray.info/install.ps1 | iex"
#
# Installs the toolchain (uv, typst, go-task), makes you a fresh, owned copy of demolab,
# scaffolds the folder structure, and prints how to continue. It does NOT install or launch a
# coding agent -- bring your own and paste the prompt it prints.
#
# NOTE: needs verification on a real Windows box. winget isn't on every machine, and a
# just-installed tool may not be on PATH until you reopen the terminal.
$ErrorActionPreference = "Stop"

# DEMOLAB_REPO lets a fork/mirror (or the test suite) install from elsewhere; defaults to upstream.
$Repo   = if ($env:DEMOLAB_REPO) { $env:DEMOLAB_REPO } else { "https://github.com/eoinmurray/demolab" }
$Dir    = if ($args.Count -ge 1) { $args[0] } else { "demolab" }
$Prompt = "Read AGENTS.md and follow the Getting started runbook to set up my lab."

function Have($cmd) { $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue) }
function Ok($name, $cmd) { if (-not $cmd) { $cmd = $name }; Write-Host ("  [ok]   {0,-8} {1}" -f $name, (Get-Command $cmd -ErrorAction SilentlyContinue).Source) -ForegroundColor Green }
function Adding($name)  { Write-Host ("  [..]   {0,-8} installing..." -f $name) -ForegroundColor Yellow }

if (Test-Path $Dir) { throw "A '$Dir' directory already exists here. Pass another name after the command." }

# --- dependencies: report every one; auto-install what we can ---
Write-Host ""
Write-Host "Checking dependencies" -ForegroundColor White

if (-not (Have "git")) { throw "git is required -- install it first, then re-run." }
Ok "git"

if (Have "uv")    { Ok "uv" }    else { Adding "uv";      Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression }
if (Have "task")  { Ok "go-task" "task" } else { Adding "go-task"; winget install Task.Task   --accept-source-agreements --accept-package-agreements }
if (Have "typst") { Ok "typst" } else { Adding "typst";   winget install Typst.Typst --accept-source-agreements --accept-package-agreements }

# --- your own copy ---
Write-Host "Fetching demolab into ./$Dir ..."
git clone --depth 1 $Repo $Dir
Set-Location $Dir
Remove-Item -Recurse -Force .git
# Drop upstream-only deploy workflow.
Remove-Item -Recurse -Force .github/workflows/landing.yml -ErrorAction SilentlyContinue
git init -q; git add -A; git commit -q -m "Start my lab from demolab"

# --- lay down the bare structure ---
Write-Host "Scaffolding the folder structure..."
task scaffold

Write-Host ""
Write-Host "--------------------------------------------------------------"
Write-Host "demolab is ready in ./$Dir"
Write-Host ""
Write-Host "Next, either:"
Write-Host "  - Open your coding agent in ./$Dir and paste:"
Write-Host "      $Prompt"
Write-Host "  - Or explore by hand:"
Write-Host "      cd $Dir; task add-demo-content; task dev"
Write-Host "--------------------------------------------------------------"
Write-Host ""
Write-Host "(If 'task' or 'typst' isn't found, reopen your terminal so PATH refreshes.)"
