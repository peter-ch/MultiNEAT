
echo "PATH = %PATH%"

REM Make sure to use proper python from conda
set PATH=%PREFIX%\bin:C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\

python "%SRC_DIR%/examples/TestTraits.py"
REM python "%SRC_DIR%/examples/NoveltySearch.py"
python "%SRC_DIR%/examples/TestNEAT_xor.py"
python "%SRC_DIR%/examples/TestHyperNEAT_xor.py"
