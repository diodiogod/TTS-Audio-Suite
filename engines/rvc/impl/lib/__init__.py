from functools import lru_cache
import os
import shelve
from contextlib import contextmanager
import sys
import weakref

@lru_cache
def get_cwd():
    CWD = os.getcwd()
    if CWD not in sys.path:
        sys.path.append(CWD)
    return CWD

class ObjectNamespace(dict):
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            if hasattr(v,"items"): self[k]=ObjectNamespace(**v)
            else: self[k]=v

    def __missing__(self, name: str): return ObjectNamespace()
    def get(self, name: str, default_value=None): return self.__getitem__(name) if name in self.keys() else default_value
    def __getattr__(self, name: str): return self.__getitem__(name) if name in self.keys() else ObjectNamespace()
    def __getitem__(self, name: str):
        value = super().__getitem__(name) # get the value from the parent class
        if isinstance(value, weakref.ref): # check if the value is a weak reference
            value = value() # call the weak reference object to get the referent
        return value # return the referent or the original value
    def __setattr__(self, name: str, value): return self.__setitem__(name, value)
    def __delattr__(self, name: str): return self.__delitem__(name) if name in self.keys() else None
    def __delitem__(self, name: str): return super().__delitem__(name) if name in self.keys() else None
    def __setstate__(self, state):
        for key in state: self[key]=state[key]
    def __getstate__(self): return dict(**self)

class PersistedDict:
    # initialize the class with an optional filename and dict arguments
    def __init__(self, filename=None, **data):
        # store the filename as an attribute
        self.filename = filename

        for key, value in data.items():
            # recursively convert the values to NestedDict
            self.__setattr__(key, value)

    # define a context manager to open and close the shelve file
    @contextmanager
    def open_shelf(self):
        # if filename is given, open the shelve file
        shelf = shelve.open(self.filename) if self.filename else {}

        # yield the shelf as the resource
        yield shelf
        if hasattr(shelf,"close"):
            # close the shelf when exiting the context
            shelf.close()

    # define a method to get the attribute value given a key
    def __getattr__(self, key: str):
        is_private = key.startswith("_") and key.endswith("_")
        
        # if the key is filename, set it as an attribute
        if key == "filename" or is_private:
            if key in self.__dict__: return self.__dict__[key]
            else: return None

        # use the context manager to open the shelve file
        with self.open_shelf() as shelf:
            # if the key exists in the shelve file, return the value
            # return getattr(shelf, key, None)
            if key in shelf:
                return shelf[key]
            # else, return None
            else:
                return None

    # define a method to set the attribute value given a key
    def __setattr__(self, key, value):
        # if the key is filename, set it as an attribute
        if key == "filename":
            self.__dict__[key] = value
        # else, use the context manager to open the shelve file
        else:
            with self.open_shelf() as shelf:
                # store the value in the shelve file
                print(f"{key}={value}")
                shelf[key] = value

    # define a method to represent the class as a dict
    def __repr__(self):
        # initialize an empty dict
        result = {}
        # use the context manager to open the shelve file
        with self.open_shelf() as shelf:
            # loop through the keys in the shelve file
            for key in shelf.keys():
                # add the key and value to the result
                result[key] = shelf[key]
        # return the result
        return str(result)
    
    def __setitem__(self, key, value): self.__setattr__(key, value)
    def __getitem__(self, key): self.__getattr__(key)
    def __lt__(self,_): return False
    def __eq__(self,other):
        if hasattr(other,"filename"): return self.filename==other.filename
        else: return False
    def __call__(self,*args,**kwargs):
        print(f"{args=}, {kwargs=}")
        return str(self)

BASE_DIR = get_cwd()
BASE_MODELS_DIR = os.path.join(BASE_DIR,"models")
SONG_DIR = os.path.join(BASE_DIR,"songs")
BASE_CACHE_DIR = os.path.join(BASE_DIR,".cache")
DATASETS_DIR = os.path.join(BASE_DIR,"datasets")
LOG_DIR = os.path.join(BASE_DIR,"logs")
OUTPUT_DIR = os.path.join(BASE_DIR,"output")