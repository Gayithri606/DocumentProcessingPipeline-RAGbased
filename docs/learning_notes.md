# Learning Notes — Document Processing Pipeline (RAG)

A Q&A reference from code exploration sessions.

---

## vector_store.py

### What does `__init__` do?

Each line explained:

```python
self.settings = get_settings()
```
Loads all app configuration (API keys, model names, DB URL) from `config/settings.py`.

```python
self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
```
Creates a connection to OpenAI's API using the API key from settings.

```python
self.embedding_model = self.settings.openai.embedding_model
```
Saves the embedding model name (e.g. `text-embedding-3-small`) as a shortcut.

```python
self.vector_settings = self.settings.vector_store
```
Saves vector-store-specific settings (table name, dimensions, partition interval) as a shortcut.

```python
self.vec_client = client.Sync(
    self.settings.database.service_url,
    self.vector_settings.table_name,
    self.vector_settings.embedding_dimensions,
    time_partition_interval=self.vector_settings.time_partition_interval,
)
```
Creates a synchronous connection to the Timescale Vector database with all the stored configuration.

---

### What is a synchronous connection?

Synchronous means your code **waits** for each operation to finish before moving to the next line. Like a phone call — you wait for the answer before continuing.

The opposite is **asynchronous** — fire off a request and continue doing other things (like sending a text message instead of a phone call).

This pipeline processes documents step-by-step (embed → store → search), so synchronous is the simpler, correct approach.

---

### What does `self.vec_client.create_tables()` do, and where does it get its info?

Tells Timescale Vector to create the necessary database tables. All the info (which DB, which table, vector size, time partitioning) was already passed when `vec_client` was built in `__init__` — so this line just uses that stored config to issue the `CREATE TABLE` command.

---

### Do we need Alembic here?

No. Alembic manages **schema migrations** — versioned changes to table structure over time. In this project the table structure is fixed and `create_tables()` handles it in one call. Alembic would only be needed if there were evolving schema changes to track (e.g. adding columns to relational tables).

---

### What is DiskAnnIndex?

**DiskANN** (Disk-based Approximate Nearest Neighbor) is an indexing algorithm that speeds up vector similarity search.

- **Without index:** scans ALL vectors one by one — slow at scale.
- **With DiskANN:** builds a graph structure over vectors so searches jump directly to relevant ones — much faster.
- **Approximate:** trades a tiny bit of accuracy for a lot of speed — negligible difference in practice.
- **Disk-based:** stored on disk, not RAM — works well with large datasets.

Call `create_index()` after loading a good chunk of data into the table.

---

### What is the purpose of `vector_store.py`?

It is the **interface between the app and the vector database**. All vector operations live here — creating tables, generating embeddings, inserting data, searching, and deleting. It abstracts all database complexity so other parts of the app just call simple methods like `.search()` or `.upsert()`.

---

### Why do we do `list(df.to_records(index=False))` in `upsert()`?

```python
records = df.to_records(index=False)
self.vec_client.upsert(list(records))
```

`vec_client.upsert()` expects a **plain Python list of tuples**. `df.to_records()` returns a NumPy recarray — it looks like tuples but is a NumPy-specific object the client doesn't understand.

- `records` = NumPy recarray ❌
- `(records)` = still a NumPy recarray (parentheses do nothing) ❌
- `list(records)` = plain Python list of tuples ✅

Example:
```python
# DataFrame:
# name    | age
# "Alice" | 25
# "Bob"   | 30

list(df.to_records(index=False))
# → [("Alice", 25), ("Bob", 30)]  ← plain Python list of tuples
```

Note: `(records)` is NOT a tuple. To make a tuple you need a comma: `(records,)` — but that would be a tuple *containing* the entire recarray as one element, which is wrong.

---

### What does `Union[List[Tuple[Any, ...]], pd.DataFrame]` mean?

The `search()` function can return **either** a list of tuples OR a DataFrame, controlled by the `return_dataframe` parameter:

```python
store.search("shipping", return_dataframe=True)   # → DataFrame
store.search("shipping", return_dataframe=False)  # → List of tuples
```

`Union[A, B]` means "either A or B". We can't know at code-writing time which one comes back because `return_dataframe` is a runtime value.

---

### Explaining search() parameters

```python
metadata_filter: Union[dict, List[dict]] = None
```
Accepts either a single dict or a list of dicts (or None by default).

```python
predicates: Optional[client.Predicates] = None
```
Accepts either a Predicates object or None.

```python
time_range: Optional[Tuple[datetime, datetime]] = None
```
Accepts either a tuple of two datetimes or None.

---

### Why `Optional[client.Predicates] = None` and not just `client.Predicates = None`?

`client.Predicates = None` is **contradictory** — it says "I expect a Predicates object" but defaults to `None`, which is not a Predicates object. Type checkers like mypy flag this as an error.

`Optional[client.Predicates] = None` is **honest** — it explicitly says "I accept a Predicates object OR None." No contradiction, type checker is happy.

`Optional[X]` is just shorthand for `Union[X, None]`.

---

### Why `Union[dict, List[dict]]` and not `Optional[Union[dict, List[dict]]]`?

They are functionally equivalent:
```python
Optional[Union[dict, List[dict]]]  ==  Union[dict, List[dict], None]
```
The `= None` default already implies None is acceptable, so wrapping with `Optional` is redundant. It's a style choice, not a functional difference.

---

## settings.py

### What is the purpose of `settings.py`?

Stores all **configuration data** — API keys, model names, database URLs, temperature, max tokens, retries. Just data, no logic.

---

### What does `lambda` do in settings.py?

```python
api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
```

`default_factory` expects a callable (a function). `lambda: os.getenv(...)` is a tiny anonymous function that returns the env variable value when called.

Why not `default=os.getenv("OPENAI_API_KEY")`?
- `default=` is evaluated **once at import time** — may miss env vars loaded later.
- `default_factory=lambda:` is evaluated **each time a Settings object is created** — always picks up the latest value.

---

## llm_factory.py

### What is the purpose of `llm_factory.py`?

It handles **client creation and usage** — takes settings and builds a working LLM client, wraps it with `instructor` for structured output, and provides `create_completion()` so the rest of the app can call any LLM the same way.

| File | Role |
|---|---|
| `settings.py` | Stores configuration (the "what") |
| `llm_factory.py` | Creates clients and makes LLM calls (the "how") |

Analogy: `settings.py` = recipe card with ingredients. `llm_factory.py` = the chef who reads the card and cooks.

---

### What is `instructor`?

A Python library that makes LLMs return **structured, typed outputs** (Pydantic models) instead of raw text strings.

```python
class Patient(BaseModel):
    symptom: str
    temperature: float

result = client.chat.completions.create(
    response_model=Patient,
    messages=[{"role": "user", "content": "Patient has fever of 101°F"}]
)
result.temperature  # → 101.0  (actual float, not text)
```

---

### Why use `lambda` in `client_initializers` dict?

```python
client_initializers = {
    "openai": lambda s: instructor.from_openai(OpenAI(api_key=s.api_key)),
}
```

Without `lambda`, the code runs **immediately** when the dict is defined — but `s` (settings) doesn't exist yet, causing a `NameError`.

With `lambda`, the function is **stored** and only runs on line 31 when called with `self.settings`:
```python
initializer(self.settings)  # NOW s = self.settings
```

Analogy:
- Without lambda = a result (cake already baked)
- With lambda = a recipe (bake it later when you have ingredients)

---

### How do you know `s` in the lambda means settings?

Trace through the code:
1. `lambda s:` — `s` is just a placeholder name
2. Line 31: `initializer(self.settings)` — `self.settings` is passed as the argument, so `s = self.settings`
3. `self.settings = getattr(get_settings(), provider)` — this is the provider-specific settings object (e.g. `OpenAISettings`) which has `.api_key`

The name `s` means nothing on its own — it could be named anything. The **argument position** on line 31 is what connects it to settings.

---

### What does `self.settings = getattr(get_settings(), provider)` mean?

`get_settings()` returns the full Settings object with all sections (openai, database, vector_store).

`getattr(object, name)` is like doing `object.name` but dynamically using a variable:

```python
# These two are identical when provider = "openai":
getattr(get_settings(), "openai")  # dynamic ✅
get_settings().openai              # hardcoded ❌ can't use a variable here
```

So `self.settings` becomes just the `OpenAISettings` object for that provider.

**Java equivalent:**
```java
Field field = settings.getClass().getDeclaredField(provider);
this.settings = field.get(settings);
```

Why not hardcode? Because `provider` is a runtime variable — you don't know at code-writing time if it'll be `"openai"`, `"anthropic"`, or `"llama"`.

---

### If provider is `"anthropic"` or `"llama"`, does `get_settings()` support them?

**No — this is a bug / incomplete implementation.**

`settings.py` only defines:
```python
class Settings(BaseModel):
    openai: OpenAISettings        # ✅ exists
    database: DatabaseSettings    # ✅ exists
    vector_store: VectorStoreSettings  # ✅ exists
    # anthropic and llama are MISSING ❌
```

So `getattr(get_settings(), "anthropic")` would throw `AttributeError` at runtime.

To fix it, you'd need to add:
```python
class AnthropicSettings(LLMSettings):
    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    default_model: str = Field(default="claude-sonnet-4-6")

class LlamaSettings(LLMSettings):
    api_key: str = Field(default_factory=lambda: os.getenv("LLAMA_API_KEY"))
    base_url: str = Field(default_factory=lambda: os.getenv("LLAMA_BASE_URL"))
    default_model: str = Field(default="llama3")

class Settings(BaseModel):
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)  # ← add
    llama: LlamaSettings = Field(default_factory=LlamaSettings)              # ← add
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
```

Java would catch this at compile time — Python only fails at runtime.

---

### Does `vec_client.search` use cosine similarity by default?

Yes, but more precisely it uses **cosine distance**:

```python
cosine distance = 1 - cosine similarity

distance = 0.0  →  identical vectors, perfect match
distance = 1.0  →  completely unrelated
```

This is why lower distance = better match.

**Why cosine for text?** It measures the angle between vectors, not their magnitude — so it focuses on meaning rather than length. "shipping options" and "what are your shipping options?" would score as very similar.

**Other pgvector distance metrics:**
| Metric | Operator | Best for |
|---|---|---|
| Cosine distance | `<=>` | Text embeddings ✅ |
| L2 (Euclidean) | `<->` | Image embeddings |
| Inner product | `<#>` | Normalized vectors |

---

### Does `vec_client.search` accept any number of search args?

Yes. A `search_args` dict is built dynamically and unpacked with `**`:

```python
search_args = {"limit": limit}  # always included

if metadata_filter:
    search_args["filter"] = metadata_filter      # added only if provided
if predicates:
    search_args["predicates"] = predicates        # added only if provided
if time_range:
    search_args["uuid_time_filter"] = ...         # added only if provided

results = self.vec_client.search(query_embedding, **search_args)
```

Only relevant parameters are included — avoids sending unnecessary `None` values to the database client.

---

### What does the metadata expansion line do in `_create_dataframe_from_results`?

```python
df = pd.concat(
    [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
)
```

It **flattens the `metadata` column** — expands the nested dict into separate columns.

**Before:**
```
id  | metadata                              | content       | distance
"1" | {"category": "Shipping", "source": "faq"} | "We offer..." | 0.187
```

**After:**
```
id  | content       | distance | category   | source
"1" | "We offer..." | 0.187    | "Shipping" | "faq"
```

- `df.drop(["metadata"], axis=1)` — removes the metadata column
- `df["metadata"].apply(pd.Series)` — expands the dict into separate columns
- `pd.concat([...], axis=1)` — joins them side by side

Why? So you can filter easily: `df[df["category"] == "Shipping"]` instead of awkward lambda expressions.

---

### What does `sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1` mean?

Ensures **exactly one** of the three delete parameters is provided.

**Step by step:**
```python
# Example: delete(ids=["123"], metadata_filter=None, delete_all=False)
(ids, metadata_filter, delete_all) = (["123"], None, False)

bool(["123"])  # → True   (non-empty list)
bool(None)     # → False
bool(False)    # → False

sum([True, False, False])  # → 1
1 != 1  # → False → no error ✅
```

**Java equivalent:**
```java
int count = 0;
if (ids != null && !ids.isEmpty()) count++;
if (metadataFilter != null) count++;
if (deleteAll) count++;

if (count != 1) {
    throw new IllegalArgumentException("Provide exactly one of: ids, metadata_filter, or delete_all");
}
```

**All scenarios:**
| ids | metadata_filter | delete_all | sum | valid? |
|---|---|---|---|---|
| `["123"]` | `None` | `False` | 1 | ✅ |
| `None` | `{"cat":"X"}` | `False` | 1 | ✅ |
| `None` | `None` | `True` | 1 | ✅ |
| `None` | `None` | `False` | 0 | ❌ nothing provided |
| `["123"]` | `{"cat":"X"}` | `False` | 2 | ❌ two provided |

---

## LLM Prompting

### When and why to use the `assistant` role in messages?

| Role | Who speaks | Purpose |
|---|---|---|
| `system` | Developer | Sets behavior, rules, persona |
| `user` | The human | Asks questions or gives instructions |
| `assistant` | The LLM | Responses from the model |

The `assistant` role is used to **inject fake prior conversation history** — telling the model "this is what you already said earlier."

**Use case 1 — Continuing a conversation:**
```python
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},  # fake prior response
    {"role": "user", "content": "What is its population?"}  # model knows "its" = Paris
]
```

**Use case 2 — Few-shot examples (teach by example):**
```python
messages = [
    {"role": "system", "content": "You extract structured data."},
    {"role": "user", "content": "John is 25 years old"},
    {"role": "assistant", "content": '{"name": "John", "age": 25}'},  # showing expected format
    {"role": "user", "content": "Sarah is 30 years old"},             # model follows the pattern
]
```

**Use case 3 — Prefilling the response (Anthropic):**
```python
{"role": "assistant", "content": "The answer is:"}  # model continues from here
```
Forces the model to complete your sentence rather than starting fresh.
