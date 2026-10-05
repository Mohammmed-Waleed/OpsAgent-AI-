@echo off
rem Launch OpsAgent from anywhere: opsagent.bat run "task" | resume | approve | runs | trace | tools | world
setlocal
set "HERE=%~dp0"
pushd "%HERE%"
"%HERE%.venv\Scripts\python.exe" -m opsagent %*
popd
