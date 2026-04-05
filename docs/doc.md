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
