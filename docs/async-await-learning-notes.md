# Learning Notes — Async, Await, Sync and FastAPI

A Q&A reference from the async upgrade discussion session.

---

## Celery vs Python async/await — are they the same thing?

No. They solve completely different problems and are not substitutes for each other.

**Python async/await (asyncio)** — concurrency *within* a single process.
When a route handler is `async def`, FastAPI's event loop can pause it while it waits for I/O (a DB query, an API call) and go handle another incoming HTTP request in the meantime — all in one thread, no blocking.

**Celery** — background task processing in *separate* worker processes.
This offloads long-running work completely outside your web server process. The FastAPI process hands off the task and returns immediately. A separate Celery worker process picks it up and runs it independently.

**Simple analogy — a coffee shop:**
- `async def` = one barista who takes your order, starts your espresso, then takes the next customer's order while the machine runs. Fast, non-blocking. Works great for short tasks.
- Celery = a barista takes your order for a custom cake and hands a ticket to the *bakery in the back*. The bakery (Celery worker) handles the long job. The barista is free immediately.

Parsing a 50-page PDF is the cake. You would never `await` that inline — you offload it to Celery.

**In production you need both:**
- `async def` for the HTTP layer (handling many concurrent requests efficiently)
- Celery for background tasks (document ingestion, email sending, large file processing)

---

## What does `await` expect?

`await` expects a **coroutine** — something that knows how to pause and resume.

Specifically, `await` works on any object that implements the `__await__` protocol. In practice that means three things:

**1. A coroutine — result of calling an `async def` function**
```python
async def fetch_data():
    return 42

result = await fetch_data()   # fetch_data() returns a coroutine object
```

**2. A Task or Future — returned by `asyncio.create_task()`, `asyncio.to_thread()`, etc.**
```python
result = await asyncio.to_thread(some_sync_fn)  # to_thread() returns a coroutine
```

**3. Any object with `__await__` defined — like asyncpg connections, aiofiles, etc.**
```python
async with aiofiles.open("file.txt") as f:
    content = await f.read()   # f.read() returns a coroutine
```

What `await` **cannot** accept:
```python
await some_regular_function()              # ❌ plain def returns a value, not a coroutine
await tempfile.NamedTemporaryFile(...)     # ❌ not a coroutine
await 42                                   # ❌ an int is not awaitable
```

One-line mental model: **`await` needs something that can be suspended mid-execution and resumed later.**

---

## Why can't we use `async with tempfile.NamedTemporaryFile(...)` ?

Two reasons:

**Reason 1 — `tempfile.NamedTemporaryFile` doesn't support `async with`.**
`async with` only works on objects that implement `__aenter__` and `__aexit__` (the async context manager protocol). `tempfile.NamedTemporaryFile` is a plain sync standard library class — it only has `__enter__` and `__exit__`. Python will throw a `TypeError`.

**Reason 2 — `asyncio.to_thread()` takes a callable, not a code block.**
`asyncio.to_thread(fn)` means "run this function in a thread pool." It needs a function object. A `with` block is not a callable — you cannot pass a code block directly to it. The wrapper function `_save_to_temp` packages that block so it can be handed off to the thread pool.

---

## Why do we need `asyncio.to_thread` — can't we just `await _save_to_temp()`?

No. `await` and `asyncio.to_thread` do two completely separate jobs.

`asyncio.to_thread` does one job — **moves the function to a thread pool** (this is what makes it non-blocking).

`await` does a different job — **pauses the current coroutine until the result is ready** (this is just how you call it).

You need both. Neither works without the other here.

```python
# ❌ Can't do this — _save_to_temp() returns a string, not a coroutine
await _save_to_temp()

# ❌ Can't do this either — async def alone does NOT move anything to a thread.
# The blocking file I/O inside still runs on the event loop thread.
async def _save_to_temp():
    with tempfile.NamedTemporaryFile(...) as tmp:   # still blocking
        shutil.copyfileobj(...)                     # still blocking

await _save_to_temp()   # event loop still frozen

# ✅ Correct — two separate things happening:
await asyncio.to_thread(_save_to_temp)
#     ↑ moves to thread    ↑ waits for result
```

**Simple analogy:**
- `asyncio.to_thread` = sending the photocopying to a copy room so someone else does it (frees you up)
- `await` = you waiting at your desk until they bring it back

If you just `await` without `to_thread`, you walked to the photocopier yourself and stood there blocking everyone else in the hallway.

---

## Does `async def` alone make code non-blocking?

No. `async def` alone does absolutely nothing for non-blocking behaviour. What matters is what is **inside** the function.

```python
# This LOOKS async but STILL BLOCKS — no real await inside, just sync code
async def _save_to_temp():
    with tempfile.NamedTemporaryFile(...) as tmp:   # blocking sync
        shutil.copyfileobj(file.file, tmp)          # blocking sync
        return tmp.name

tmp_path = await _save_to_temp()   # event loop still frozen
```

When the event loop runs this coroutine, it hits the blocking code and freezes. There is nothing inside it that yields control back. `async def` just gave it a coroutine wrapper — the contents are still sync.

Contrast with a genuinely non-blocking function:
```python
# Non-blocking ONLY because inside it awaits real async I/O
async def get_user(user_id):
    row = await db.fetch("SELECT ...")   # ← THIS is what makes it non-blocking
    return row
```

If `get_user` had sync blocking DB calls with no `await` inside, it would block too, regardless of `async def`.

---

## Does every `await` need `asyncio.to_thread`?

No. `asyncio.to_thread` is only needed for **sync blocking code** that has no async version.

**Situation 1 — Calling a natively async library (no `to_thread` needed)**
```python
# aiohttp, asyncpg, AsyncOpenAI, timescale client.Async etc. are natively async
embedding = await self.async_openai_client.embeddings.create(...)   # ✅ no to_thread
rows = await async_conn.fetchall()                                    # ✅ no to_thread
results = await self.async_vec_client.search(...)                     # ✅ no to_thread
```
These libraries already return coroutines. Just await them directly.

**Situation 2 — Calling another `async def` function you wrote (no `to_thread` needed)**
```python
async def get_user(user_id):
    row = await db.fetch(...)
    return row

user = await get_user(123)   # ✅ no to_thread — it's already async def with real awaits inside
```

**Situation 3 — Wrapping sync blocking code that has no async version (`to_thread` needed)**
```python
# tempfile, shutil, requests — standard library sync with no async equivalent
tmp_path = await asyncio.to_thread(_save_to_temp)   # ✅ to_thread needed here
```

**The rule:**
```
natively async?              →  just await it directly
sync code block?             →  asyncio.to_thread first, then await
pure computation (no I/O)?   →  no await at all, just call it normally
```

---

## Can we do `await return IngestResponse(...)` ?

No. `await` is not about returning — it is specifically about waiting for I/O to finish.
`IngestResponse(...)` is just building a Pydantic object in memory. No network, no disk, no waiting. There is nothing to pause for.

```python
await return IngestResponse(...)   # ❌ SyntaxError and conceptually wrong

return IngestResponse(...)         # ✅ just return it, no awaiting needed
```

Think of `await` as "pause here until the slow thing outside this process finishes." Creating an object in memory is never slow in that sense.

---

## Can a function be `async def` without any `await` inside?

Yes, technically completely fine. Python won't complain.

```python
async def greet():
    return "hello"   # no await inside, perfectly valid
```

But it is pointless. All it does is make `greet()` return a coroutine object instead of a string, so the caller is forced to `await greet()` unnecessarily.

The only real-world case where this makes sense is implementing an interface or base class that requires an async signature, but that specific implementation happens to not need any I/O:

```python
class BaseHandler:
    async def handle(self):   # interface requires async
        raise NotImplementedError

class SimpleHandler(BaseHandler):
    async def handle(self):   # must match signature
        return "done"         # no await needed here, but async is required by contract
```

Outside of that — if there is no `await` inside, there is no reason for `async def`.

---

## Does FastAPI require async, or is async about the I/O operations?

FastAPI does **not** force async. You can use `def` or `async def` — FastAPI handles both.
The driving reason for async is always the I/O operations, not FastAPI itself.

But FastAPI has an important connection — it is built on an **ASGI event loop** (via uvicorn/Starlette).

```
async def route  →  runs directly ON the event loop
                     no thread overhead
                     but blocking I/O inside freezes everything

def route        →  FastAPI automatically moves it to a thread pool
                     safe for blocking I/O
                     but has thread creation overhead per request
```

**The I/O is the reason.** OpenAI calls, DB queries, file reads — these are slow external operations. Async lets the event loop handle other requests while waiting for them.

**FastAPI is the enabler.** Because it runs on an event loop, `async def` routes take full advantage of that. In a sync framework like Flask, `async def` would give you nothing useful.

---

## `async def` route vs `def` route — detailed explanation

**The setup: FastAPI runs on a single-threaded event loop.**
uvicorn starts one event loop. That loop is one thread. All incoming HTTP requests come to that one loop.

**`async def` route — runs directly on the event loop:**
```python
@router.post("/query")
async def query_documents(request: QueryRequest):
    results = await vector_store.search(...)             # pauses, loop handles other requests
    response = await Synthesizer.generate_response(...)  # pauses again
    return QueryResponse(...)
```
FastAPI calls this coroutine directly on the event loop thread. No new thread is created.
When it hits `await`, it pauses and the event loop picks up another waiting request.
When the I/O finishes, it resumes. One thread, many concurrent requests, no thread overhead.

But if you do blocking I/O inside without `await`:
```python
async def query_documents(request):
    results = requests.get("https://api.openai.com/...")  # blocking — no await
    # event loop is FROZEN here
    # every other request is stuck waiting
    # your server is effectively dead during this call
```

**`def` route — FastAPI moves it to a thread pool:**
```python
@router.get("/health")
def health_check():
    return {"status": "ok"}
```
FastAPI internally does this:
```python
await asyncio.get_event_loop().run_in_executor(thread_pool, your_sync_function)
```
It borrows a thread from a pool, runs your function there, and the event loop stays free.
Safe for blocking I/O — the blocking happens in a separate thread, not on the event loop.

The cost: thread pool has limits (default ~40 threads in Python). Under very high concurrency, threads can be exhausted. Also each `def` route invocation needs a thread — more overhead than a pure async call.

**Decision rule:**
```
route does I/O (DB, network, file)  →  async def  +  await the I/O parts
route does pure computation only    →  def  (no async needed at all)
```

---

## Why can't we do `await celery_app.AsyncResult(job_id)` despite the name?

Despite the name `AsyncResult`, it is **not** an async function. The name is Celery's own terminology for "a result object that will be resolved in the future" — it has nothing to do with Python's `async/await`.

```python
task = await celery_app.AsyncResult(job_id)   # ❌ TypeError: object is not awaitable
```

`AsyncResult` is just a plain Python object. When you access `.status` or `.result` on it, those properties make a synchronous Redis call under the hood. That Redis call is the actual blocking I/O — which is why the whole thing is wrapped in `asyncio.to_thread`.

```python
def _fetch_task_info():
    task = celery_app.AsyncResult(job_id)   # just creates an object
    return task.status, task.result          # THIS hits Redis — blocking sync

status, result = await asyncio.to_thread(_fetch_task_info)   # ✅
```

---

## Summary — How to decide where to put `await`

`await` goes before anything that involves **waiting outside your process** — network, disk, database. Not randomly, not everywhere.

```
Already a natively async library call?   →  await it directly
Another async def function you wrote?    →  await it directly (if it has real awaits inside)
Sync blocking code with no async version →  asyncio.to_thread(fn), then await that
Pure in-memory computation (no I/O)?     →  no await, just call normally
```

`async def` on a function means only "this function is allowed to contain `await`."
It does nothing on its own. The actual non-blocking behaviour comes from what you `await` inside it.
