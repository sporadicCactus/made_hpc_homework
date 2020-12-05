all: linear_filter median_filter histogram

%: src/%.cpp obj/%.o
	g++ -o $@ -I inc/ $^ -lopencv_core -lopencv_imgcodecs -lcuda -lcudart

obj/%.o: src/%.cu
	nvcc $< -c -o $@

clean:
	rm -f obj/*.o
