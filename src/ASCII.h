#ifndef ASCII_H_INCLUDE
#define ASCII_H_INCLUDE

#include "Engine.h"
#include "Utils.h"

#include <string>

class ASCII {
public:
    ASCII();
    ASCII(const ASCII&) = delete;
    ASCII& operator=(const ASCII&) = delete;

private:
    void init();
    void loop();

    Engine *m_ascii_engine{nullptr};

    std::string execute(Utils::CommandParser &parser);
};

#endif
