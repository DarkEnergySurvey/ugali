#!/usr/bin/env python
"""
Python utilities.
"""
import six
import numpy as np

def iterable(obj):
    """return true if *obj* is iterable"""
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def isstring(obj):
    """Python 2/3 compatible string check"""
    return isinstance(obj, six.string_types)

def rec_append_fields(rec, names, arrs, dtypes=None):
    """
    Return a new record array with field names populated with data
    from arrays in *arrs*.  If appending a single field, then *names*,
    *arrs* and *dtypes* do not have to be lists. They can just be the
    values themselves.
    """
    if (not isstring(names) and iterable(names) and len(names) and isstring(names[0])):
        if len(names) != len(arrs):
            raise ValueError("number of arrays do not match number of names")
    else:  # we have only 1 name and 1 array
        names = [names]
        arrs = [arrs]
    arrs = list(map(np.asarray, arrs))
    if dtypes is None:
        dtypes = [a.dtype for a in arrs]
    elif not iterable(dtypes):
        dtypes = [dtypes]
    if len(arrs) != len(dtypes):
        if len(dtypes) == 1:
            dtypes = dtypes * len(arrs)
        else:
            raise ValueError("dtypes must be None, a single dtype or a list")
    old_dtypes = rec.dtype.descr
    if six.PY2:
        old_dtypes = [(name.encode('utf-8'), dt) for name, dt in old_dtypes]
    newdtype = np.dtype(old_dtypes + list(zip(names, dtypes)))
    newrec = np.recarray(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    for name, arr in zip(names, arrs):
        newrec[name] = arr
    return newrec
