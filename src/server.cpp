#include "crow.h"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <memory>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/error.h>
    #include <libswscale/swscale.h>
}

static constexpr AVCodecID VIDEO_CODEC_ID = AV_CODEC_ID_H264;
static constexpr int DETECTAR_A_CADA_N_FRAMES = 3;
static const char* PASTA_CAPTURAS = "capturas";

static const char* SSD_PROTOTXT = "MobileNetSSD_deploy.prototxt";
static const char* SSD_MODELO = "MobileNetSSD_deploy.caffemodel";
static const char* HAAR_ROSTO = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
static constexpr int SSD_ENTRADA = 300;
static constexpr int SSD_CLASSE_PESSOA = 15;
static constexpr float PESSOA_CONF_MIN = 0.5f;



#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cstring>

struct alignas(64) ShmHeader {
    uint32_t magic, width, height, channels;
    std::atomic<uint64_t> seq;
    std::atomic<uint64_t> frame_id;
    uint64_t ts_ms;
};
static_assert(sizeof(ShmHeader) == 64);

class FrameShm {
    private:
        std::string nome_;
        int w_, h_, fd_ = -1;
        size_t size_ = 0;
        uint8_t* base_ = nullptr;
        ShmHeader* hdr_ = nullptr;


    public:
        FrameShm(const char* nome, int w, int h) : nome_(nome), w_(w), h_(h) {
            size_ = sizeof(ShmHeader) + size_t(w) * h * 3;
            fd_ = shm_open(nome, O_CREAT | O_RDWR, 0666);
            if (fd_ < 0) return;
            if (ftruncate(fd_, size_) < 0) return;
            void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (p == MAP_FAILED) return;
            base_ = static_cast<uint8_t*>(p);
            hdr_ = reinterpret_cast<ShmHeader*>(base_);
            hdr_->magic = 0x46524D31;  // "FRM1"
            hdr_->width = w; hdr_->height = h; hdr_->channels = 3;
            hdr_->seq.store(0); hdr_->frame_id.store(0);
        }
        ~FrameShm() {
            if (base_) munmap(base_, size_);
            if (fd_ >= 0) { close(fd_); shm_unlink(nome_.c_str()); }
        }

        bool ok() const { return base_ != nullptr; }
        int w() const { return w_; }
        int h() const { return h_; }

        void escrever(const cv::Mat& bgr, uint64_t ts_ms) {
            // bgr precisa ser CV_8UC3 contínuo e do tamanho w_ x h_
            uint64_t s = hdr_->seq.load(std::memory_order_relaxed);
            hdr_->seq.store(s + 1, std::memory_order_relaxed);   // ímpar
            std::atomic_thread_fence(std::memory_order_release);

            std::memcpy(base_ + sizeof(ShmHeader), bgr.data, size_t(w_) * h_ * 3);
            hdr_->ts_ms = ts_ms;
            hdr_->frame_id.fetch_add(1, std::memory_order_relaxed);

            hdr_->seq.store(s + 2, std::memory_order_release);   // par
        }
};



// ---------------------------------------------------------------------
// Estado do decoder (uma única conexão ativa)
// ---------------------------------------------------------------------

struct DecoderState {
    AVCodecContext* codec_ctx = nullptr;
    AVCodecParserContext* parser = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    uint64_t frames_recebidos = 0;

    cv::dnn::Net ssd;
    cv::CascadeClassifier rosto;
    SwsContext* sws_bgr = nullptr;
    cv::Mat bgr;

    std::unique_ptr<FrameShm> shm;

    ~DecoderState() {
        av_frame_free(&frame);
        av_packet_free(&pkt);
        if (parser) av_parser_close(parser);
        avcodec_free_context(&codec_ctx);
        sws_freeContext(sws_bgr);
    }
};

static std::unique_ptr<DecoderState> g_decoder;
static crow::websocket::connection* g_conn = nullptr;

static std::unique_ptr<DecoderState> criar_decoder() {
    const AVCodec* codec = avcodec_find_decoder(VIDEO_CODEC_ID);
    if (!codec) {
        CROW_LOG_ERROR << "Codec de vídeo não encontrado";
        return nullptr;
    }

    auto st = std::make_unique<DecoderState>();

    st->codec_ctx = avcodec_alloc_context3(codec);
    if (!st->codec_ctx || avcodec_open2(st->codec_ctx, codec, nullptr) < 0) {
        CROW_LOG_ERROR << "Falha ao abrir o decoder";
        return nullptr;
    }

    st->parser = av_parser_init(VIDEO_CODEC_ID);
    if (!st->parser) {
        CROW_LOG_ERROR << "Falha ao iniciar o parser";
        return nullptr;
    }

    st->pkt = av_packet_alloc();
    st->frame = av_frame_alloc();

    st->ssd = cv::dnn::readNetFromCaffe(SSD_PROTOTXT, SSD_MODELO);
    if (st->ssd.empty()) {
        CROW_LOG_ERROR << "Falha ao carregar MobileNet-SSD";
        return nullptr;
    }

    if (!st->rosto.load(HAAR_ROSTO)) {
        CROW_LOG_ERROR << "Falha ao carregar cascade de rosto: " << HAAR_ROSTO;
        return nullptr;
    }
    return st;
}

// ---------------------------------------------------------------------
// Detecção: pessoa (SSD) -> rosto (Haar) dentro da caixa da pessoa
// ---------------------------------------------------------------------

static void detectar_e_salvar(DecoderState& st, const AVFrame* frame) {
    const int w = frame->width, h = frame->height;
    const cv::Rect limites(0, 0, w, h);

    // AVFrame (YUV) -> cv::Mat BGR
    st.sws_bgr = sws_getCachedContext(
        st.sws_bgr, w, h, (AVPixelFormat)frame->format,
        w, h, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!st.sws_bgr) return;

    st.bgr.create(h, w, CV_8UC3);
    uint8_t* dst[1] = { st.bgr.data };
    int dst_linesize[1] = { static_cast<int>(st.bgr.step) };
    sws_scale(st.sws_bgr, frame->data, frame->linesize, 0, h, dst, dst_linesize);

    cv::Mat anotado = st.bgr.clone();

    // Etapa 1: pessoas
    cv::Mat blob = cv::dnn::blobFromImage(
        st.bgr, 0.007843, cv::Size(SSD_ENTRADA, SSD_ENTRADA),
        cv::Scalar(127.5, 127.5, 127.5), false, false);
    st.ssd.setInput(blob);
    cv::Mat saida = st.ssd.forward();             // [1, 1, N, 7]
    cv::Mat det(saida.size[2], saida.size[3], CV_32F, saida.ptr<float>());

    bool achou = false;
    for (int i = 0; i < det.rows; ++i) {
        const float* d = det.ptr<float>(i);       // _, classe, conf, x1, y1, x2, y2 (normalizados)
        if (static_cast<int>(d[1]) != SSD_CLASSE_PESSOA || d[2] < PESSOA_CONF_MIN) continue;

        cv::Rect pessoa(static_cast<int>(d[3] * w), static_cast<int>(d[4] * h),
                        static_cast<int>((d[5] - d[3]) * w), static_cast<int>((d[6] - d[4]) * h));
        pessoa &= limites;
        if (pessoa.area() <= 0) continue;

        // Etapa 2: rosto, só na parte de cima do recorte da pessoa
        cv::Rect topo(pessoa.x, pessoa.y, pessoa.width, static_cast<int>(pessoa.height * 0.6));
        cv::Mat cinza;
        cv::cvtColor(st.bgr(topo), cinza, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(cinza, cinza);

        std::vector<cv::Rect> rostos;
        st.rosto.detectMultiScale(cinza, rostos, 1.1, 5, 0, cv::Size(30, 30));
        if (rostos.empty()) continue;

        achou = true;
        // cv::rectangle(st.bgr, pessoa, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(anotado, pessoa, cv::Scalar(0, 255, 0), 2);
        for (const cv::Rect& r : rostos) {
            // cv::rectangle(st.bgr, r + topo.tl(), cv::Scalar(255, 0, 0), 2);
            cv::rectangle(anotado, r + topo.tl(), cv::Scalar(255, 0, 0), 2);
        }
    }

    if (!achou) return;

    if (!st.shm || st.shm->w() != w || st.shm->h() != h) {
        st.shm.reset();  // libera o antigo antes de recriar com o mesmo nome
        st.shm = std::make_unique<FrameShm>("/frames_cam", w, h);
        if (!st.shm->ok()) { CROW_LOG_ERROR << "Falha ao criar shm"; st.shm.reset(); return; }
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    st.shm->escrever(st.bgr, ms); // escrita em memoria
    
    std::string caminho = std::string(PASTA_CAPTURAS) + "/pessoa_" +
    std::to_string(ms) + "_f" + std::to_string(st.frames_recebidos) + ".jpg";
    
    
    if (cv::imwrite(caminho, anotado)) {
        CROW_LOG_INFO << "Pessoa com rosto detectada, foto salva: " << caminho;
    } else {
        CROW_LOG_ERROR << "Falha ao salvar " << caminho;
    }
    
    
}

// ---------------------------------------------------------------------
// Consumo do vídeo
// ---------------------------------------------------------------------
// Uma mensagem pode conter várias frames concatenadas (ou uma frame partida
// em duas mensagens), então o parser acha os limites de cada access unit
// antes de mandar packet por packet ao decoder.

static void consumir_video(DecoderState& st, const std::string& data) {
    const uint8_t* buf = reinterpret_cast<const uint8_t*>(data.data());
    int buf_size = static_cast<int>(data.size());

    while (buf_size > 0) {
        uint8_t* parsed_data = nullptr;
        int parsed_size = 0;

        int consumido = av_parser_parse2(
            st.parser, st.codec_ctx,
            &parsed_data, &parsed_size,
            buf, buf_size,
            AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);

        buf += consumido;
        buf_size -= consumido;

        if (parsed_size > 0) {
            st.pkt->data = parsed_data;
            st.pkt->size = parsed_size;

            int ret = avcodec_send_packet(st.codec_ctx, st.pkt);
            if (ret < 0) {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errbuf, sizeof(errbuf));
                CROW_LOG_WARNING << "Falha ao enviar packet ao decoder: " << errbuf;
            } else {
                while (avcodec_receive_frame(st.codec_ctx, st.frame) == 0) {
                    st.frames_recebidos++;
                    if (st.frames_recebidos % DETECTAR_A_CADA_N_FRAMES == 0) {
                        detectar_e_salvar(st, st.frame);
                    }
                }
            }
        }

        if (consumido == 0) break; // evita loop infinito se o parser não avançar
    }
}

// ---------------------------------------------------------------------
// App
// ---------------------------------------------------------------------

int main() {
    std::filesystem::create_directories(PASTA_CAPTURAS);

    crow::SimpleApp app;

    CROW_WEBSOCKET_ROUTE(app, "/ws/video")
        .onopen([](crow::websocket::connection& conn) {
            if (g_conn) {
                conn.close("já existe uma conexão de vídeo ativa");
                return;
            }
            g_decoder = criar_decoder();
            if (!g_decoder) {
                conn.close("falha ao iniciar decoder");
                return;
            }
            g_conn = &conn;
            CROW_LOG_INFO << "Conexão de vídeo aberta";
        })
        .onclose([](crow::websocket::connection& conn, const std::string& reason, uint16_t) {
            if (&conn != g_conn) return; // conexão recusada
            CROW_LOG_INFO << "Conexão de vídeo encerrada (" << reason
                          << "), frames recebidos: " << g_decoder->frames_recebidos;
            g_decoder.reset();
            g_conn = nullptr;
        })
        .onerror([](crow::websocket::connection&, const std::string& msg) {
            CROW_LOG_ERROR << "Erro no WebSocket de vídeo: " << msg;
        })
        .onmessage([](crow::websocket::connection& conn, const std::string& data, bool is_binary) {
            if (!is_binary || &conn != g_conn) return;
            consumir_video(*g_decoder, data);
        });

    app.port(18080).run();
    return 0;
}