#include "Winograd_helper.h"

std::vector<float> winograd_transform_f(const std::vector<float> &f,
                                        const int outputs, const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    constexpr auto G = std::array<float, 3 * WINOGRAD_ALPHA>{
        1.0f,         0.0f,         0.0f,        -2.0f / 3.0f, -SQ2 / 3.0f,
        -1.0f / 3.0f, -2.0f / 3.0f, SQ2 / 3.0f,  -1.0f / 3.0f, 1.0f / 6.0f,
        SQ2 / 6.0f,   1.0f / 3.0f,  1.0f / 6.0f, -SQ2 / 6.0f,  1.0f / 3.0f,
        0.0f,         0.0f,         1.0f};

    auto temp = std::array<float, 3 * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    auto buffer = std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA>{};

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs / buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < 3; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += G[i * 3 + k] * f[o * channels * 9 + c * 9 + k * 3 + j];
                        }
                        temp[i * 3 + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                    auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += temp[xi * 3 + k] * G[nu * 3 + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels + c * outputs + o] = buffer[buffersize * i + entry];
                }
            }
        }
    }

    U.shrink_to_fit();
    return U;
}
