#include "metrics/MetricCalculator.h"

size_t MetricCalculator::calculateLatency(const Sequence& seq) {
    if (seq.first_token_time == 0 || seq.last_token_time == 0) {
        return 0;
    }
    return ns_to_ms(seq.last_token_time - seq.first_token_time);
}

size_t MetricCalculator::calculateITL(const Sequence& seq) {
    if (seq.itl_count == 0) {
        return 0;
    }
    return ns_to_ms(seq.itl_sum / seq.itl_count);
}

size_t MetricCalculator::calculateTPOT(const Sequence& seq) {
    if (seq.generated_token_count <= 1) {
        return 0;
    }
    size_t total_time_ns = seq.last_token_time - seq.first_token_time;
    return ns_to_ms(total_time_ns / (seq.generated_token_count - 1));
}

size_t MetricCalculator::calculateTTFT(const Sequence& seq) {
    if (seq.first_token_time == 0 || seq.submitted_time == 0) {
        return 0;
    }
    return ns_to_ms(seq.first_token_time - seq.submitted_time);
}