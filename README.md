# pyReCoDe
Pythonic implementation of the ReCoDe data reduction-compression scheme as described in:   
Datta, A., Ng, K.F., Balakrishnan, D. et al. A data reduction and compression description for high throughput time-resolved electron microscopy. Nat Commun 12, 664 (2021). https://doi.org/10.1038/s41467-020-20694-z

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

 
