"""Provide vocabulary"""
from os import remove, listdir

NPATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner'
npaths = [NPATH_PREFIX + '/ncomp/', NPATH_PREFIX + '/nsimp/',
          NPATH_PREFIX + '/ncomp_map/', NPATH_PREFIX + '/nsimp_map/',
          NPATH_PREFIX + '/meta2/', NPATH_PREFIX + '/ppdb2/', NPATH_PREFIX + '/ppdb_rule2/']
for npath in npaths:
    files = listdir(npath)
    for file in files:
        if file.startswith('shard'):
            f = open(npath + file)
            content = ''
            lines = f.readlines()
            if len(lines) == 1 or len(lines) == 0:
                for line in f.readlines():
                    content += line
                if len(content) == 0:
                    remove(npath + file)
                    print('removed' + (npath + file))
        else:
            print('Ok with file %s.' % file)