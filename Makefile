raytracer: main.cpp
	g++ main.cpp -lGL -lglut -lGLEW -o raytracer

run: raytracer
	./raytracer

