# T1
./src/sh.sh

# T2
python websocket.py

# T3: assim que o log do servidor mostrar "foto salva"
ls -l /dev/shm/frames_cam
python leitor.py