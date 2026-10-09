corefiles := generator.o
BIN_DIR := ./bin

all: $(BIN_DIR)/libgenerator.so generator.o

$(BIN_DIR)/libgenerator.so: generator.c
	mkdir -p $(BIN_DIR)
	gcc -g -shared -fPIC $^ -o $@

generator.o: generator.c
	gcc -c generator.c

clean:
	rm -f *.o
	rm -f *.so
	rm -rf $(BIN_DIR)
