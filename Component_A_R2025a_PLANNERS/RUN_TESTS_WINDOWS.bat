@echo off
setlocal
cd /d "%~dp0"
echo Component A - planner, PID, world and clearance checks
echo.
where py >nul 2>&1
if not errorlevel 1 (
  py -3 -m unittest discover -s tests -v
  goto finished
)
where python >nul 2>&1
if not errorlevel 1 (
  python -m unittest discover -s tests -v
  goto finished
)
echo Python was not found. Install Python 3.10 or newer to run these checks.
echo The Webots world can still be opened separately.
:finished
echo.
echo These checks do not launch Webots or replace a full simulator run.
pause
