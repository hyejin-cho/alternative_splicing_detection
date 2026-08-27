#coding: UTF-8
#!/usr/bin/env python
##hycho@coh.org

import sys
import re
import os
import gzip
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s %(message)s', filename='log_file.txt')

class File(object):
    '''File Base Class'''
    
    def __init__(self, inFile):
        self.file = inFile
        self.baseName = os.path.basename(inFile)
        self.type = self._findType()
        self.sName = self._shortName()
    
    def _findType(self):
        self._fileExists()
        if self.baseName.endswith(('.gz', '.z', '.gzip', '.GZ', '.Z', '.GZIP')):
            return 'gz'
        else:
            return 'txt'

    def _shortName(self):
        lable = -1
        if self.type == 'gz':
            lable -= 1
        return '.'.join(self.baseName.split('.')[:lable]) + '.'

    def _openFile(self):
        if self.type == 'txt':
            return open(self.file)
        else:
            return gzip.open(self.file, 'rt')
                    
    def _closeFile(self, handle):
        handle.close()

    def _fileExists(self):
        if not os.path.isfile(self.file):
            logging.error('Error, ' + self.file + ' is not exists.')
            sys.exit(1)
    
def readRecord(self):
    pass

