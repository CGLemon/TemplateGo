#ifndef GTP_H_INCLUDE
#define GTP_H_INCLUDE

#include "Engine.h"
#include "Utils.h"

#include <memory>
#include <string>

class GTP {
public:
    GTP();
    GTP(const GTP&) = delete;
    GTP& operator=(const GTP&) = delete;

private:
    void init();
    void loop();
    void execute(Utils::CommandParser &parser);

    Engine *m_gtp_engine{nullptr};

};

#endif
