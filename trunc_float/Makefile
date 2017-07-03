# Makefile for truncated float class

CMP=g++
FLAGS=-std=gnu++11
SHELL=bash
CLASS=trunc_float
EXEC=test_float

$(EXEC): $(CLASS).o main.cpp
	@echo Linking $@...
	@$(CMP) $(FLAGS) $^ -o $@

$(CLASS).o: $(CLASS).cpp $(CLASS).h
	@echo Compiling $@...
	@$(CMP) $(FLAGS) -c $< -o $@

clean:
	@echo Cleaning up...
	@rm *.o
	@rm $(EXEC)
