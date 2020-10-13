CC=gcc

files=$(ls . | grep -P ".c$")

for file in $files
do
    $CC -fopenmp -o "${file%.*}" $file -lm
done
