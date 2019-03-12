#!/usr/bin/env python
import hadoopy
import os

hdfs_path = ''


def read_local_dir(local_path):
    for fn in os.listdir(local_path):
        path = os.path.join(local_path, fn)
        if os.path.isfile(path):
            yield path


def main():
    local_path = './BigData/dummy_data'
    for file in  read_local_dir(local_path):
        hadoopy.put(file, 'data')
        print "The file %s has been put into hdfs" % (file,)

if __name__ == '__main__':
    main()
