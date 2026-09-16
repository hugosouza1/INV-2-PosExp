#include <crow.h>

using namespace crow;

int main() {
    SimpleApp app;

    CROW_ROUTE(app, "/")
    ([] {
        return "Servidor Crow funcionando!";
    });

    CROW_ROUTE(app, "/hello")
    ([] {
        return "Hello World!";
    });

    CROW_ROUTE(app, "/teste")
    ([] {
        return "Rota de teste funcionando!";
    });

    app.bindaddr("0.0.0.0")
       .port(18080)
       .multithreaded()
       .run();
}