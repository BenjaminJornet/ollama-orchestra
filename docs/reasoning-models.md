# Reasoning Models

Some Ollama reasoning models can return an empty visible response even though the server did work.

## Symptom

You may see a response shaped like this:

```json
{
  "message": {"content": ""},
  "done_reason": "length"
}
```

The model spent the available generation budget in hidden reasoning before producing user-visible content.

## Fix

Ollama supports disabling model-side thinking with `think: false`, but it must be placed at the top level of the request body.

Correct:

```json
{
  "model": "your-model",
  "messages": [{"role": "user", "content": "Summarize this log"}],
  "think": false,
  "options": {"num_predict": 256}
}
```

Incorrect:

```json
{
  "model": "your-model",
  "messages": [{"role": "user", "content": "Summarize this log"}],
  "options": {"think": false, "num_predict": 256}
}
```

## Helper usage

```python
from ollama_orchestra import chat

result = await chat(
    "http://localhost:11434",
    "your-model",
    [{"role": "user", "content": "Summarize this log"}],
    think=False,
    num_predict=256,
)
```

`chat()` also strips leftover `<think>`, `<reasoning>`, `<thought>`, and simple Markdown fences from returned content by default.
