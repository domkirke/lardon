import os, sys

def checkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)