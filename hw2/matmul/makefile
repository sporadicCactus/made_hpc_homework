CC = gcc
CFLAGS = -I. -lblas -lm
CFLAGS_OPT = 

main: main.c matutils.o matmul_naive.o matmul_blas.o matmul_strassen.o
	$(CC) -o main main.c matutils.o matmul_naive.o matmul_blas.o matmul_strassen.o $(CFLAGS) $(CFLAGS_OPT)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) $(CFLAGS_OPT)

clean:
	rm *.o
