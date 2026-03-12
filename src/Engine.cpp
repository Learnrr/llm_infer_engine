#include "Engine.h"

void Engine::init(){
    cache_manager = make_unique<KVCacheManager>();
    workspace = make_unique<Workspace>();
    model = make_unique<Model>(*workspace);
    scheduler = make_unique<Scheduler>(cache_manager, model);
}

void Engine::run() {
    runner_thread = std::thread(&Scheduler::schedule, scheduler.get());
}   

void Engine::create_sequence(size_t seq_id, vector<size_t> token_ids) {
    scheduler->addSequence(seq_id, token_ids);
}

void Engine::get_sequence_output(size_t seq_id, vector<size_t>& output_token_ids) {

    Sequence* seq;
    scheduler->getFinishedSequenceById(seq_id, seq);
    std::unique_lock<std::mutex> lock(seq->mtx);
    seq->cv.wait(lock, [&seq]{ return seq->state == SequenceState::FINISHED; });
    
    output_token_ids = seq->output_token_ids;
}

void Engine::check_sequence_state(size_t seq_id, SequenceState& state) {
    Sequence* seq;
    scheduler->getSequenceById(seq_id, seq);
    if (seq) {
        state = seq->state;
    } else {
        //
    }
}