#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WORKERSAMPLER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WORKERSAMPLER_HPP

#include <cstdint>

namespace AgcDriver {

void StartWorkerSampler(std::uint32_t queue = 0);
void StartCommitterSampler();
void StartFinisherSampler();
void StopWorkerSampler();
bool ReadableStackRegion(std::uint64_t address, std::uint64_t& low, std::uint64_t& high);

class SampledHitch {
public:
    explicit SampledHitch(const char* label, std::uint32_t value = 0, bool armed = true) noexcept;
    ~SampledHitch();
    SampledHitch(const SampledHitch&) = delete;
    SampledHitch& operator=(const SampledHitch&) = delete;

private:
    const char* label;
    std::uint32_t value;
    std::int64_t began;
};

}

#endif
