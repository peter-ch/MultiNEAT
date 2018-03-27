
REM show test env before latering it
set

REM Make sure to use proper python from conda
set PATH=%PREFIX%:C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\

%CONDA_PYTHON_EXE% "%SRC_DIR%/examples/TestTraits.py"
REM %CONDA_PYTHON_EXE% "%SRC_DIR%/examples/NoveltySearch.py"
%CONDA_PYTHON_EXE% "%SRC_DIR%/examples/TestNEAT_xor.py"
%CONDA_PYTHON_EXE% "%SRC_DIR%/examples/TestHyperNEAT_xor.py"
