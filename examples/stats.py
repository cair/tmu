import numpy as np
import sys

m = np.loadtxt(sys.argv[1])
start = int(sys.argv[2])
#length = int(sys.argv[3])

print("MIN", m[np.logical_and(m[:,1] >= start, m[:,1] < start+25),:].min(axis=0))
print("MEAN", m[np.logical_and(m[:,1] >= start, m[:,1] < start+25),:].mean(axis=0))
print("MEDIAN", np.median(m[np.logical_and(m[:,1] >= start, m[:,1] < start+25),:], axis=0))
print("STD", m[np.logical_and(m[:,1] >= start, m[:,1] < start+25),:].std(axis=0))
print("95%", np.percentile(m[np.logical_and(m[:,1] >= start, m[:,1] < start+25),:], 95, axis=0))
print("MAX", m[np.logical_and(m[:,1] >= start, m[:,1] < start+25),:].max(axis=0))

ids = m[:,0]
data = m[:,1:]
_ndx = np.argsort(ids)
_id, _pos  = np.unique(ids[_ndx], return_index=True)
g_max = np.maximum.reduceat(data[_ndx], _pos)
print("MIN-MAX:", g_max.min(axis=0))
