# llm_infer_engine
supported model: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
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

**7. performance (needs improvement)**   
[2026-04-02 05:42:55] [INFO] /llm_infer_engine/src/Engine.cpp:124 - Sequence 1 metrics: Latency=8819ms, ITL=152ms, TPOT=152ms, TTFT=975ms     
INFO:     127.0.0.1:43134 - "POST /v1/chat/completions HTTP/1.1" 200 OK  
