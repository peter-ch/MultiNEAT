@ECHO ON

setx MN_BUILD boost

python %SRC_DIR%/setup.py build_ext
python %SRC_DIR%/setup.py install
