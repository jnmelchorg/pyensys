from libc.stdlib cimport malloc, free

N = 16

cdef int *row = <int*>malloc(N*sizeof(int))

for i in range(N):
    row[i] = i

for i in range(N):
    print(row[i])

row_arr = [row[i] for i in range(N)]
print(row_arr)

free(row)
