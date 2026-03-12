#include "Scheduler.h"

void Scheduler::schedule() {
    
    while(True){

        launchSequence();

        if(!prefilling_queue.empty() || !decoding_queue.empty()){
            Batch decode_batch = buildDecodeBatch();
            
            model.decode_forward(decode_batch);

        }
        

        if(!waiting_queue.empty()){
            Batch prefill_batch = buildPrefillBatch();
            model.prefill_forward(prefill_batch);

            movePrefilledToDecoding(prefill_batch);
        }

        handleFinishedSequence();
    }
}

void Scheduler::movePrefilledToDecoding(const Batch& prefill_batch) {
    for(auto it = prefill_batch.sequences.begin(); it != prefill_batch.sequences.end();){
        if(it->SequenceState == SequenceState::PREFILLED){
            it->SequenceState = SequenceState::DECODING;
            decoding_queue.push_back(*it);
            it = prefilling_queue.erase(it);
        } else {
            ++it;
        }
    }
}

Batch Scheduler::buildDecodeBatch() {
    // Implement the logic to build a batch of sequences for processing
    Batch batch;
    batch.batch_size = 0;
    
    for(auto& seq : decoding_queue){
        if(seq.SequenceState!= SequenceState::FINISHED){
            
            batch.token_ids.insert(batch.token_ids.end(), seq.token_ids.begin(), seq.token_ids.end());

            size_t seq_len = seq.token_ids.size();
            for(size_t i = 0; i < seq_len; ++i){
                batch.token_positions.push_back(i);
                batch.sequences.push_back(&seq);
            }
            batch.num_tokens += seq.token_ids.size();
            batch.batch_size++;

            if(seq.sequence_len % BLOCK_SIZE == 0){
                CacheBlock* block = cache_manager->allocate_cache_block();
                if(block){
                    seq.blocks.push_back(block);
                } else {
                    // Handle cache allocation failure (e.g., log an error, skip the sequence, etc.)
                }
            }
            
            if(batch.batch_size >= MAX_DECODE_BATCH_SIZE){
                break;
            }
        }
    }
    while(batch.batch_size < MAX_DECODE_BATCH_SIZE && !prefilling_queue.empty()){
        Sequence seq = prefilling_queue.front();
        if(seq.SequenceState == SequenceState::PREFILLED){
            prefilling_queue.erase(prefilling_queue.begin());
            seq.SequenceState = SequenceState::DECODING;
            decoding_queue.push_back(seq);
            
            batch.token_ids.insert(batch.token_ids.end(), seq.token_ids.begin(), seq.token_ids.end());

            size_t seq_len = seq.token_ids.size();
            for(size_t i = 0; i < seq_len; ++i){
                batch.token_positions.push_back(i);
                batch.sequences.push_back(&seq);
            }
            batch.num_tokens += seq.token_ids.size();
            batch.batch_size++;
        }
    }
    return batch;
}

Batch Scheduler::buildPrefillBatch() {
    Batch batch;
    batch.batch_size = 0;
    
    for(auto& seq : waiting_queue){
        if(seq.SequenceState == SequenceState::WAITING){
            waiting_queue.erase(waiting_queue.begin());
            seq.SequenceState = SequenceState::PREFILLING;
            prefilling_queue.push_back(seq);
            
            batch.token_ids.insert(batch.token_ids.end(), seq.token_ids.begin(), seq.token_ids.end());
            size_t seq_len = seq.token_ids.size();
            for(size_t i = 0; i < seq_len; ++i){
                batch.token_positions.push_back(i);
                batch.sequences.push_back(&seq);
            }
            batch.num_tokens += seq.token_ids.size();
            batch.batch_size++;
            
            //allocate cache blocks for the sequence prefill
            size_t num_blocks = (seq.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for(size_t i = 0; i < num_blocks; ++i){
                CacheBlock* block = cache_manager->allocate_cache_block();
                if(block){
                    seq.blocks.push_back(block);
                } else {
                    // Handle cache allocation failure (e.g., log an error, skip the sequence, etc.)
                }
            }

            if(batch.batch_size >= MAX_PREFILL_BATCH_SIZE){
                break;
            }
        }
    }
    return batch;
}

void Scheduler::addSequence(size_t seq_id, vector<size_t> token_ids) {
    Sequence new_seq(seq_id);
    new_seq.token_ids = token_ids;
    waiting_queue.push_back(new_seq);
}


void Scheduler::launchSequence(){
    while(!waiting_queue.empty()){
        Sequence seq = waiting_queue.front();
        waiting_queue.erase(waiting_queue.begin());
        seq.SequenceState = SequenceState::PREFILLING;
        prefilling_queue.push_back(seq);
    }
}

void Scheduler::handleFinishedSequence(){
    for(auto it = decoding_queue.begin(); it != decoding_queue.end();){
        if(it->SequenceState == SequenceState::FINISHED){
            // Handle the finished sequence (e.g., remove from decoding queue, update cache, etc.)
            it = decoding_queue.erase(it);
            finished_queue.push_back(*it);
        } else {
            ++it;
        }
    }
}

void Scheduler::getSequenceById(size_t seq_id, Sequence* seq) {           
    for (auto& sequence : waiting_queue) {
        if (sequence.seq_id == seq_id) {
            seq = &sequence;
            return;
        }
    }
    for (auto& sequence : prefilling_queue) {
        if (sequence.seq_id == seq_id) {
            seq = &sequence;
            return;
        }
    }
    for (auto& sequence : decoding_queue) {
        if (sequence.seq_id == seq_id) {
            seq = &sequence;
            return;
        }
    }
    for (auto& sequence : finished_queue) {
        if (sequence.seq_id == seq_id) {
            seq = &sequence;
            return;
        }
    }     
    seq = nullptr; // Sequence not found
}

void Scheduler::getFinishedSequenceById(size_t seq_id, Sequence* seq) {
    for (auto& sequence : finished_queue) {
        if (sequence.seq_id == seq_id) {
            seq = &sequence;
            return;
        }
    }
    seq = nullptr; // Sequence not found
}