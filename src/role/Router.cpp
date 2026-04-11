#include "role/Router.h"
#include "utils/logger.h"
#include "channel/ChannelMessage.h"
#include "channel/ChannelManager.h"

void Router::set_channels() {
    ChannelManager* manager = ChannelManager::get_instance();
    auto get_or_null = [manager](const std::string& name) -> Channel* {
        Channel* channel = nullptr;
        ErrorCode err = manager->get_channel(name, channel);
        if (err != ErrorCode::SUCCESS) {
            return nullptr;
        }
        return channel;
    };

    to_prefiller_channel = get_or_null("router_to_prefill_scheduler");
    to_decoder_channel = get_or_null("router_to_decode_scheduler");
    from_prefiller_channel = get_or_null("prefill_scheduler_to_router");
    from_decoder_channel = get_or_null("decode_scheduler_to_router");
}

void Router::run() {
    LOG_INFO("Router started");
    route();
    LOG_INFO("Router stopped");
}

void Router::route() {
    while(!stop_requested.load()){
        size_t prefill_seq_id = 0;
        bool has_prefill = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if(!prefill_ready_queue.empty()){
                prefill_seq_id = prefill_ready_queue.front();
                prefill_ready_queue.pop_front();
                has_prefill = true;
            }
        }
        if (has_prefill) {
            add_to_prefiller(prefill_seq_id);
        }

        size_t decode_seq_id = 0;
        bool has_decode = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if(!decode_ready_queue.empty()){
                decode_seq_id = decode_ready_queue.front();
                decode_ready_queue.pop_front();
                has_decode = true;
            }
        }
        if (has_decode) {
            add_to_decoder(decode_seq_id);
        }

        from_prefiller_handler();

        from_decoder_handler();

    }

}

ErrorCode Router::add_sequence(
    size_t seq_id,
    std::vector<size_t> token_ids,
    const SequenceConfig& sequence_config
) {
    auto seq = std::make_shared<Sequence>(seq_id, sequence_config);
    seq->token_ids = std::move(token_ids);
    seq->seq_len = seq->token_ids.size();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        sequence_store[seq_id] = seq;
        route_states[seq_id] = RouteType::PREFILL;
        prefill_ready_queue.push_back(seq_id);
    }
    return ErrorCode::SUCCESS;
}

void Router::add_to_prefiller(size_t seq_id) {
    if (to_prefiller_channel == nullptr) {
        LOG_ERROR("to_prefiller_channel is null in add_to_prefiller");
        return;
    }
    RouteMessage msg;
    msg.seq_id = seq_id;
    msg.route_type = RouteType::PREFILL;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        RouteMeta meta;
        meta.route_type = RouteType::PREFILL;
        prefill_inflight[seq_id] = meta;
        route_states[seq_id] = RouteType::PREFILL;
        auto seq_it = sequence_store.find(seq_id);
        if (seq_it != sequence_store.end() && seq_it->second) {
            msg.sequence_config = seq_it->second->seq_config;
            msg.token_ids = seq_it->second->token_ids;
        }
    }
    to_prefiller_channel->send(msg);
}

void Router::add_to_decoder(size_t seq_id) {
    if (to_decoder_channel == nullptr) {
        LOG_ERROR("to_decoder_channel is null in add_to_decoder");
        return;
    }
    RouteMessage msg;
    msg.seq_id = seq_id;
    msg.route_type = RouteType::DECODE;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        RouteMeta meta;
        meta.route_type = RouteType::DECODE;
        decode_inflight[seq_id] = meta;
        route_states[seq_id] = RouteType::DECODE;
        auto seq_it = sequence_store.find(seq_id);
        if (seq_it != sequence_store.end() && seq_it->second) {
            msg.sequence_config = seq_it->second->seq_config;
            msg.token_ids = seq_it->second->token_ids;
        }
    }
    to_decoder_channel->send(msg);
}

void Router::from_decoder_handler() {
    if (from_decoder_channel == nullptr) {
        LOG_ERROR("from_decoder_channel is null in from_decoder_handler");
        return;
    }
    //receive the sequence from decode channel
    RouteMessage msg;
    if (!from_decoder_channel->try_receive(msg)) {
        return;
    }
    size_t seq_id = msg.seq_id;
    std::lock_guard<std::mutex> lock(queue_mutex);
    auto it = decode_inflight.find(seq_id);
    if (it != decode_inflight.end()) {
        if (msg.route_type != RouteType::DECODE) {
            LOG_ERROR("Router expected DECODE return from decoder for seq " + std::to_string(seq_id));
            return;
        }
        it->second.route_type = RouteType::FINISHED;
        route_states[seq_id] = RouteType::FINISHED;

        auto seq_it = sequence_store.find(seq_id);
        if (seq_it != sequence_store.end() && seq_it->second) {
            if (!msg.token_ids.empty()) {
                seq_it->second->token_ids = msg.token_ids;
                seq_it->second->seq_len = msg.token_ids.size();
            }
        }
        decode_inflight.erase(it);
        route_cv.notify_all();
    }
}

void Router::from_prefiller_handler() {
    if (from_prefiller_channel == nullptr) {
        LOG_ERROR("from_prefiller_channel is null in from_prefiller_handler");
        return;
    }
    //receive the sequence from prefill channel
    RouteMessage msg;
    if (!from_prefiller_channel->try_receive(msg)) {
        return;
    }
    size_t seq_id = msg.seq_id;
    std::lock_guard<std::mutex> lock(queue_mutex);
    auto it = prefill_inflight.find(seq_id);
    if (it != prefill_inflight.end()) {
        if (msg.route_type != RouteType::PREFILL) {
            LOG_ERROR("Router expected PREFILL return from prefiller for seq " + std::to_string(seq_id));
            return;
        }
        it->second.route_type = RouteType::PREFILLED;
        route_states[seq_id] = RouteType::PREFILLED;

        //move the sequence to decode ready queue
        auto seq_it = sequence_store.find(seq_id);
        if (seq_it != sequence_store.end() && seq_it->second) {
            if (!msg.token_ids.empty()) {
                seq_it->second->token_ids = msg.token_ids;
                seq_it->second->seq_len = msg.token_ids.size();
            }
            seq_it->second->seq_config = msg.sequence_config;
        }
        prefill_inflight.erase(it);
        decode_ready_queue.push_back(seq_id);
    }
}

ErrorCode Router::wait_until_finished(size_t seq_id) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    auto exists = sequence_store.find(seq_id);
    if (exists == sequence_store.end() || !exists->second) {
        return ErrorCode::SEQUENCE_NOT_FOUND;
    }

    route_cv.wait(lock, [this, seq_id]() {
        auto it = route_states.find(seq_id);
        return it != route_states.end() && it->second == RouteType::FINISHED;
    });

    return ErrorCode::SUCCESS;
}

ErrorCode Router::getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    auto it = sequence_store.find(seq_id);
    if (it == sequence_store.end() || !it->second) {
        seq = nullptr;
        return ErrorCode::SEQUENCE_NOT_FOUND;
    }
    seq = it->second;
    return ErrorCode::SUCCESS;
}

ErrorCode Router::removeFinishedSequenceById(size_t seq_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    auto it = sequence_store.find(seq_id);
    if (it == sequence_store.end()) {
        return ErrorCode::SEQUENCE_NOT_FOUND;
    }
    sequence_store.erase(it);
    prefill_inflight.erase(seq_id);
    decode_inflight.erase(seq_id);
    route_states.erase(seq_id);
    return ErrorCode::SUCCESS;
}