g++ server.cpp -o server.exe $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale libswresample opencv4)
./server.exe