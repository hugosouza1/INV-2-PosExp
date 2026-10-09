# rodar server.exe na pasta src
./src/server

# rodar "servidor de leitura dos frames"
python3 -m pipeline.live.ao_vivo --model model_v3_e100s2.pt 

# mandar websocket
python3 websocket.py 