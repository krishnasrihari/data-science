import hadoopy

def mapper(key, value):
    for w in value.split():
        yield w, 1

def reducer(key, values):
    counter = 0
    for count in values:
        counter += int(count)
    yield key, counter

if __name__ == "__main__":
    hadoopy.run(mapper, reducer, doc=__doc__)
