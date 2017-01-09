#coding=utf8

from __future__ import print_function

class NotAFileException(Exception):
    """Exception that shows the given path is not a file"""
    def __init__(self, path):
        super(NotAFileException, self).__init__()
        self.path = path


    def __str__(self):
        return 'Not a file: ' + self.path
