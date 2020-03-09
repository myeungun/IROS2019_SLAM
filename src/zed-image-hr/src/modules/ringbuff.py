from collections import deque
import numpy as np
from time import time

# RingBuffer class, implementation of Circular Queue in python numpy array
# from https://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/
# Modified by Kiyoon Kim (yoonkr33@gmail.com)

def ringbuff_deque_test():
    ringlen = 2
    ringbuff = deque(np.zeros((15,ringlen,ringlen), dtype='f'), ringlen)
    for i in range(40):
        ringbuff.extend(i*np.ones((2,2,3), dtype='f')) # write
        print np.array(ringbuff)
class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length, dtype='float64'):
        self.data = np.zeros(length, dtype=dtype)
        self.head = 0
        self.tail = 0
        self.nb_data = 0

    def empty(self):
        self.head = 0
        self.tail = 0
        self.nb_data = 0

    def is_empty(self):
        return self.nb_data == 0

    def is_full(self):
        return self.nb_data >= self.data.size

    def push(self, x):
        "adds array x to ring buffer"
        if x.size == 0:
            raise ValueError("RingBuffer error: 'push' got zero-sized array.")
        if self.nb_data+x.size > self.data.size:
            raise Exception("RingBuffer error: 'push' not enough capacity.\nInput size: %d, number of data in buffer: %d, and capacity: %d" % (x.size, self.nb_data, self.data.size))
        x_index = (self.tail + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.tail = x_index[-1] + 1
        self.nb_data += x.size

    def get(self, length=-1):
        """Returns the first-in-first-out data in the ring buffer.
        Note that this does not 'pop' out the data"""
        if length < 0:
            length = self.nb_data
        elif length == 0:
            return None
        elif length > self.nb_data:
            raise Exception("RingBuffer error: 'get' not enough data.\nget size: %d, number of data in buffer: %d, and capacity: %d" % (length, self.nb_data, self.data.size))

        idx = (self.head + np.arange(length)) %self.data.size
        return self.data[idx]

    def pop(self, length=-1):
        "pop out the data"
        if length < 0:
            length = self.nb_data
        elif length > self.nb_data:
            raise Exception("RingBuffer error: 'pop' not enough data.\npop size: %d, number of data in buffer: %d, and capacity: %d" % (length, self.nb_data, self.data.size))
        # elif type(length) not in (int, long):
        #     raise ValueError("RingBuffer error: 'pop' length %s given. Should be an integer." % type(length).__name__)

        self.head = (self.head + length) % self.data.size
        self.nb_data -= length


def ringbuff_numpy_test():
    ringlen = 1100
    ringbuff = RingBuffer(ringlen)
    for i in range(1000):
        ringbuff.push(np.ones(200, dtype='f')) # write
        tmp = ringbuff.get(200) #read
        tmp = ringbuff.pop(200) #read
if __name__ == '__main__':
    #ringbuff_deque_test()
    start = time()
    ringbuff_numpy_test()
    print time() - start
    ringlen = 10
    ringbuff = RingBuffer(ringlen)
    ringbuff.push(np.ones(2, dtype='float64'))
    ringbuff.push(np.zeros(2, dtype='float64'))
    ringbuff.push(2*np.ones(2, dtype='float64'))
    print ringbuff.get(3)
    ringbuff.pop(3)
    print ringbuff.get(2)
    ringbuff.pop(2)

