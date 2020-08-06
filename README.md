### Installation
Use the following commands to build and install  pyReCoDe
1. python setup.py clean --all  
2. python setup.py build   
3. python setup.py -r requirements.txt install --force  

### Additional Dependencies
The default compression scheme in pyReCoDe is deflate (zlib). 
When using other compression schemes the are following additional dependencies need to be installed as required:  
* blosc
* zstandard
* snappy
* lz4
* lzma
* bz2

 