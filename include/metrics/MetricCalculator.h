#pragma once
#include "utils/timer.h"
#include "Sequence.h"

class MetricCalculator {
    public:
        size_t calculateLatency(const Sequence& seq);

        size_t calculateITL(const Sequence& seq);

        size_t calculateTPOT(const Sequence& seq);

        size_t calculateTTFT(const Sequence& seq);
};