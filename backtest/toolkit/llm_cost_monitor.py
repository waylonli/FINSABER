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
    global llm_cost
    llm_cost = 0

def get_llm_cost():
    return llm_cost

def add_openai_cost_from_response(openai_response):
    # first check if the global variable exists
    try:
        global llm_cost
    except Exception as e:
        raise ValueError("llm_cost is not defined. Please run reset_llm_cost() first.")

    usage = openai_response.get("usage", None)
    prompt_token = usage.get("prompt_tokens", 0) if usage else 0
    generated_token = usage.get("generated_tokens", 0) if usage else 0
    if "gpt-4o-mini" in openai_response.get("model", ""):
        cost = (prompt_token * OPENAI_PRICE["gpt-4o-mini"]["input"] + generated_token * OPENAI_PRICE["gpt-4o-mini"][
            "output"]) / 1000000
    elif "gpt-4o" in openai_response.get("model", ""):
        cost = (prompt_token * OPENAI_PRICE["gpt-4o"]["input"] + generated_token * OPENAI_PRICE["gpt-4o"][
            "output"]) / 1000000
    else:
        cost = 0

    llm_cost += cost
    return cost
