OPENAI_PRICE = {
    "gpt-4o-mini": {
        "input": 0.15, # per million
        "output": 0.6, # per million
    },
    "gpt-4o": {
        "input": 2.5, # per million
        "output": 10, # per million
    },
}

def reset_llm_cost():
    global llm_cost, llm_cost_records
    llm_cost = 0.0
    llm_cost_records = []

def get_llm_cost():
    return llm_cost

def get_llm_cost_ledger():
    return list(llm_cost_records)


def _ensure_state():
    global llm_cost, llm_cost_records
    if "llm_cost" not in globals():
        reset_llm_cost()


def _usage_value(usage, key, default=0):
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(key, default)
    return getattr(usage, key, default)


def _response_value(response, key, default=None):
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _openai_cost(model, prompt_tokens, completion_tokens):
    if "gpt-4o-mini" in model:
        price = OPENAI_PRICE["gpt-4o-mini"]
    elif "gpt-4o" in model:
        price = OPENAI_PRICE["gpt-4o"]
    else:
        return 0.0
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def add_llm_cost(model, prompt_tokens=0, completion_tokens=0, provider="openai", metadata=None):
    _ensure_state()
    global llm_cost, llm_cost_records
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    cost = _openai_cost(model, prompt_tokens, completion_tokens) if provider == "openai" else 0.0
    llm_cost += cost
    llm_cost_records.append({
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
        **(metadata or {}),
    })
    return cost


def add_openai_cost_from_response(openai_response):
    usage = _response_value(openai_response, "usage")
    model = _response_value(openai_response, "model", "")
    prompt_tokens = _usage_value(usage, "prompt_tokens", 0)
    completion_tokens = (
        _usage_value(usage, "completion_tokens", None)
        or _usage_value(usage, "generated_tokens", None)
        or _usage_value(usage, "output_tokens", 0)
    )
    return add_llm_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider="openai",
    )


def add_openai_cost_from_tokens_count(model, prompt_tokens, generated_tokens=None, completion_tokens=None):
    return add_llm_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens if completion_tokens is not None else generated_tokens,
        provider="openai",
    )


reset_llm_cost()
