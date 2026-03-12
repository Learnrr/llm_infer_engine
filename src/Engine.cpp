#include "Engine.h"

void Engine::init(){
    cache_manager = make_unique<KVCacheManager>();
    workspace = make_unique<Workspace>();
    model = make_unique<Model>(*workspace);
    scheduler = make_unique<Scheduler>(cache_manager, model);

    runner_thread = std::thread(&Engine::run, this);
}

void Engine::run() {
    scheduler->schedule();
}   

void Engine::create_sequence(size_t seq_id, vector<size_t> token_ids) {
    scheduler->addSequence(seq_id, token_ids);
}

void Engine::get_sequence_output(size_t seq_id, vector<size_t>& output_token_ids) {
    // Implement logic to retrieve the output token IDs for the given sequence ID
    Sequence* seq;
    scheduler->getFinishedSequenceById(seq_id, seq);
    if (seq) {
        output_token_ids = seq->output_token_ids;
    } else {
        output_token_ids = {};
    }
}

void check_sequence_state(size_t seq_id, SequenceState& state) {
    Sequence* seq;
    scheduler->getSequenceById(seq_id, seq);
    if (seq) {
        state = seq->state;
    } else {
        //
    }
}