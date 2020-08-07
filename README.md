### Installation
Use the following commands to build and install  pyReCoDe:
1. python setup.py clean --all  
2. python setup.py build
3. pip3 install -r requirements.txt
4. python setup.py install --force  

To test the installation is successful, run a minimal read/write test using the following commands:
1. cd tests (changing directory to the 'tests' folder is necessary)
2. python minimal_read_write_test.py

### Additional Dependencies
The default compression scheme in pyReCoDe is deflate (zlib). 
When using other compression schemes the following additional dependencies need to be installed as required:  
* blosc
* zstandard
* snappy
* lz4
* lzma
* bz2

 