bom dia
### compilar e rodar
para compilar e executar é so rodar:

```
./sh.sh
```
antes de permissão pro arquivo 

```
chmod +x sh.sh
```
</br>
</br>

## tratamento do mp4
para mandar pela rota, tem que ser somente o compimido, não o conteiner(mp4). o formato do comprimido é h264. 
```
ffmpeg -i videoTeste2.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 videoTeste2.h264
```

## Envio
mande um video por websocket:
```
websocat --binary ws://localhost:18080/ws/video < videoTeste2.h264
```

# Dependencias:
biblioteca de manipulação de video:

``` 
 sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev pkg-config 
``` 

Crow (rotas): 
```
https://github.com/crowcpp/crow
```

# acho que é so isso
