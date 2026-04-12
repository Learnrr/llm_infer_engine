Apr. 4, 2026

1. add engine role
In initialization, the engine will choose a role, worker or scheduler.
Scheduler only coordinate the workers to do forward. Worker only does
model forward under the coordination of scheduler. Different role has
different setups. scheduler will not have kv manager, workspace, etc
while worker will have most of the parts.

2. add prerequisite for Pipeline Parallellism 
when workers doing the model forward, different worker is responsible
for different layers of a model. For example, 2 workers will split the
model forward, 0-13 for worker0, 14-28 for worker1 if a model has 28
hidden layers. The embedding will be the first worker's job, and the
postprocessing will be the last worker's job.

3. To enable this and remain compatibility of other forward methods or
techniques, a model executor is added. It's between the role level and
the model level. Currently there are only pipeline and singleCard
executor. Depending on the config, the engine initially will select
different executor, and pass the model (different model also can be selected) into it.

4. To enable communication between workers and scheduler, Channels are
created when engine does the initialization. They are created
automatically depending on the worker id in config. One Scheduler and
multiple workers are connected by channels to form a circle. To
communicate, A channel protocol is defined to enable different
situations, e.g. prefill, decode, release resource and so on. Usually
the scheduler send a message to the first worker to let it do the job,
then the worker pass the message down after it finishes, and the last
worker will send response back to scheduler.

Apr. 5, 2026  

1. support Pipeline Parallelism
add Pipeline executor to support pipeline parallelism without modifying
too much other parts. Currently there are singleCardexecutor which is
for simple single GPU inference and PipelineExecutor which is for
multiple GPU inference with pipeline parallelism.

2. split executor into scheduler side and worker side
scheduler side executor is for scheduling, protocol sending and
receiveing, while worker side executor is for model forward
Different exectuor is for different mechanism. Currently scheduler side executor there are
coordinator (for normal) and pipelineCoordinator (for pipeline)
and for worker side executor there are singlecard for normal, and Pipeline for pipeline.

3. pipeline parallel inference
every time the scheduler can submit a prefill/decode batch without
blocking. The maximum number of in-flight batches can be configured in
config file. But it doesnot support multiple batch computation when do
the inference. SO the point is, previously after the scheduler submit a
prefill/decode batch, there is only one batch in the pipeline, even
there are multiple GPUs, which means only one is working while others
idle. Now all are working but only one batch on one GPU. So different
batch sizes may have different GPU utilization.

Apr. 7, 2026 

1. add prefix caching
the system is able to save and retrieve prefix cache when doing the
prefill, both in single GPU (normal) and multiple GPUs (pipeline) mode.

2. prefix caching mechanism
to enable this, a prefix cache manager is created in scheduler/worker
and then be passed into executor. Every worker caches its own cache
blocks. Every time after worker executes a prefill, it will cache
the cache blocks locally.

3. A protocol message type called 'Prefix Probe' is added, when a prefill
batch is about to be submitted, a prefix probe will be submitted around
the worker cycle to see what is the common hit tokens. Then the
scheduler will block itself until get the results and  add the common tokens
to the batch and submit the prefill batch. Each worker will skip the
prefill of the common tokens. If for a sequence in a batch all the tokens
are hit commonly, the sequence prefill will be skipped by the workers.

Apr. 11, 2026

1. add disaggregated prefill and decode
spliting execution of prefill and decode on different GPU. prefill and
decode is assigned to different worker, prefiller and decoder. Each is
with a scheduler, prefill scheduler and decode scheduler. The scheduler
only build prefill or decode batch according to the role.

2. router to receive and dispatch requests
A new role 'router' is added as a high-level coordinator of
prefill/decode scheduler. It receives requests and routes them first to
prefill scheduler. After prefill finishes, it routes the requests to
decode scheduler. After decode finishes, it return the result tokens to
the user.

3. Communication between prefiller and decoder
new channels between prefiller and decoder are added. They are for
control level communications like seq meta, pulling kv request or
kvcache cuda handle. Besides thee control messages, KVCache is
transfered from prefiller to decoder when the decoder first decode a sequence.