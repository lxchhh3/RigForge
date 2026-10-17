@echo off
REM RigForge dev launcher. Brings up:
REM   - BE:  FastAPI on http://127.0.0.1:8000  (Anthropic LLM client, claude-sonnet-4-6)
REM   - FE:  Vite dev server on http://localhost:5173
REM Each runs in its own window so logs stay separate and Ctrl+C kills
REM only the service in that window.

setlocal
set "ROOT=%~dp0"

set "OUTPUT_DIR=%ROOT%Output"

echo === RigForge launch ===
echo   ROOT   = %ROOT%
echo   OUTPUT = %OUTPUT_DIR%
echo   BE     : python -m rigforge.cli serve --llm-anthropic --cache-dir cache --output-dir Output
echo   FE     : npm run dev  (in frontend\)
echo.

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Fail fast if the Anthropic API key file is missing — BE reads it at
REM startup. (When using --llm-ollama instead, check for data\ollama.json.)
if not exist "%ROOT%API_KEY_MAKE_VERY_SURE_NO_LEAKAGE.md" (
  echo [warn] API_KEY_MAKE_VERY_SURE_NO_LEAKAGE.md missing — BE will crash on
  echo        first LLM call with --llm-anthropic. Put your Anthropic API key
  echo        in that file in ROOT (it is gitignored via *API_KEY*).
  echo.
)

REM start /D sets the spawned cmd's cwd directly — more reliable than
REM piping `cd /d` into cmd /k, which has fragile quoting with && and
REM trailing backslashes. The `%ROOT%.` trick strips the trailing slash
REM from %~dp0 so we don't pass `"D:\path\"` (the `\"` can read as escape).
REM cmd /k keeps the window open after the server exits so you can read
REM the final log; Ctrl+C in the window stops just that service.
start "RigForge BE" /D "%ROOT%." cmd /k python -m rigforge.cli serve --llm-anthropic --cache-dir cache --output-dir Output

REM Tiny delay so the BE banner doesn't race the FE startup output.
timeout /t 2 /nobreak >nul

REM Vite picks 5173 if free, otherwise 5174/5175. The CORS allowlist on the
REM BE already includes all three.
start "RigForge FE" /D "%ROOT%frontend" cmd /k npm run dev

echo.
echo Two windows opened. Close them or hit Ctrl+C in each to stop.
endlocal
