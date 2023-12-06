# multi_threaded.py
import time
from threading import Thread

COUNT = 250000000

def countdown(n):
    while n>0:
        n -= 1

n_threads = 10
threads = []
for i in range(n_threads):

    t1 = Thread(target=countdown, args=(COUNT//n_threads,))
    threads.append((t1))
# t2 = Thread(target=countdown, args=(COUNT//n_threads,))

start = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
# t1.join()
# t2.join()
end = time.time()

print('Time taken in seconds -', end - start)
