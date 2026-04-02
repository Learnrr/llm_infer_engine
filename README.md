# llm_infer_engine
supported model: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
## features
1. moduled layers to build up model structure  
2. paged attention  
3. continuous batching  
4. openai API support  
## quick start  
**1. clone to local**  
`git clone ...`  
**2. download weights**  
`cd llm_infer_engine/weights`  
`git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`  
**3. compile with make**  
`cd llm_infer_engine`  
`make clean`  
`make all`  
**4. start serving**  
`python3 -m uvicorn python.fastapi:app --host 0.0.0.0 --port 8000 --workers 1`   
**5. curl through api**  
```
curl -s http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{
    "model":"qwen",
    "messages":[
      {"role":"user","content":"what is the weather like today."}
    ],
    "max_tokens":128,
    "temperature":0.7
  }'
```
**6. response like:**   
{"id":"chatcompletion-f4be52f405204aeda349c02f93a1b781","object":"chat.completion","created":1775108575,"model":"qwen","choices":[{"index":0,"message":{"role":"assistant","content":"I'm sorry, but I don't have real-time data access to provide the current weather. You can check a reliable weather website or app, such as the Weather Channel, AccuWeather, or your local news station for the most accurate and up-to-date weather information for your location.","name":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":59, "total_tokens":74}}   

**7. performance (improving)**   
[2026-04-02 05:42:55] [INFO] /llm_infer_engine/src/Engine.cpp:124 - Sequence 1 metrics: Latency=8819ms, ITL=152ms, TPOT=152ms, TTFT=975ms     
INFO:     127.0.0.1:43134 - "POST /v1/chat/completions HTTP/1.1" 200 OK  

**8. benchmark (improving)**   
SIZE FOR KVCACHE: 2147483648  
```
python3 benchmark/benchmark_concurrency.py --base-url http://127.0.0.1:8000 --prompt "Write a short poem."  --requests 50  --concurrency 8 --top-p 1.0 --top-k 50 --max-tokens 128
```
**max_decode_batch_size = 8, max_prefill_batch_size = 8**   

when max_decode_batch_size = 8, max_prefill_batch_size = 8  and -concurrency 8, decode batch size in runtime switches between 7 and 1.   
```
=== Benchmark Report ===  
Total requests:      50  
Success:             50  
Failed:              0  
Wall time (s):       370.571  
Throughput req/s:    0.13  
Success req/s:       0.13  

Latency (all, ms):  
  min:               24801.81  
  mean:              54692.31  
  p50:               54889.65  
  p95:               56917.34  
  p99:               80427.87  
  max:               88810.40  
```

**max_decode_batch_size = 4, max_prefill_batch_size = 4**  

when max_decode_batch_size = 4, max_prefill_batch_size = 4  and -concurrency 8, decode batch size in runtime always 4, except for last batches.  
```
=== Benchmark Report ===
Total requests:      50
Success:             50
Failed:              0
Wall time (s):       398.035
Throughput req/s:    0.13
Success req/s:       0.13

Latency (all, ms):
  min:               28421.06
  mean:              60712.41
  p50:               54245.03
  p95:               78064.94
  p99:               90371.84
  max:               90410.61
```

**max_decode_batch_size = 1, max_prefill_batch_size = 1**  
```
=== Benchmark Report ===
Total requests:      50
Success:             50
Failed:              0
Wall time (s):       988.131
Throughput req/s:    0.05
Success req/s:       0.05

Latency (all, ms):
  min:               100957.19
  mean:              150269.34
  p50:               157829.82
  p95:               177685.11
  p99:               187645.75
  max:               196983.15
```





