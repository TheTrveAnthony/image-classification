
# include openCV
cv = `pkg-config --cflags --libs opencv`

classify : classify.cpp
		g++ $^ $(cv) -g -o $@ 