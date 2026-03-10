#pragma once

#include"Sequence.h"
#include"define.h"
#include "KVCacheManager.h"
#include "model.h"
#include "Batch.h"
#include <vector>

class Scheduler{
    public:
        Scheduler(KVCacheManager* cache_manager, Model* model) {
            init(cache_manager, model);
        }
        void init(KVCacheManager* cache_manager, Model* model) {
            this->cache_manager = cache_manager;
            this->model = model;
        }

        void schedule() {
            
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
        
        void movePrefilledToDecoding(const Batch& prefill_batch) {
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

        Batch buildDecodeBatch() {
            // Implement the logic to build a batch of sequences for processing
            Batch batch;
            batch.batch_size = 0;
            
            for(auto& seq : decoding_queue){
                if(seq.SequenceState!= SequenceState::FINISHED){
                    batch.sequences.push_back(&seq);
                    batch.token_ids.insert(batch.token_ids.end(), seq.token_ids.begin(), seq.token_ids.end());
                    batch.num_tokens += seq.token_ids.size();
                    batch.batch_size++;
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
                    batch.sequences.push_back(&decoding_queue.back());
                    batch.token_ids.insert(batch.token_ids.end(), seq.token_ids.begin(), seq.token_ids.end());
                    batch.num_tokens += seq.token_ids.size();
                    batch.batch_size++;
                }
            }
            return batch;
        }

        Batch buildPrefillBatch() {
            Batch batch;
            batch.batch_size = 0;
            
            for(auto& seq : waiting_queue){
                if(seq.SequenceState == SequenceState::WAITING){
                    waiting_queue.erase(waiting_queue.begin());
                    seq.SequenceState = SequenceState::PREFILLING;
                    prefilling_queue.push_back(seq);
                    batch.sequences.push_back(&seq);
                    batch.token_ids.insert(batch.token_ids.end(), seq.token_ids.begin(), seq.token_ids.end());
                    batch.num_tokens += seq.token_ids.size();
                    batch.batch_size++;
                    if(batch.batch_size >= MAX_PREFILL_BATCH_SIZE){
                        break;
                    }
                }
            }
            return batch;
        }

        void addSequence(size_t seq_id, vector<size_t> token_ids) {
            Sequence new_seq(seq_id);
            new_seq.token_ids = token_ids;
            waiting_queue.push_back(new_seq);
        }


        void launchSequence(){
            while(!waiting_queue.empty()){
                Sequence seq = waiting_queue.front();
                waiting_queue.erase(waiting_queue.begin());
                seq.SequenceState = SequenceState::PREFILLING;
                prefilling_queue.push_back(seq);
            }
        }

        void handleFinishedSequence(){
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

    private:
        vector<Sequence> waiting_queue;
        vector<Sequence> decoding_queue;
        vector<Sequence> prefilling_queue;
        vector<Sequence> finished_queue;

        KVCacheManager* cache_manager;
        Model* model;
        
}