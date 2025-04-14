from typing import Dict

QUERY_TYPES = {}

def register_query(name):
    def decorator(query):
        QUERY_TYPES[name] = query
        return query

    return decorator

@register_query("plain")
def plain(params: Dict):
    query_text = params["query_text"]
    return query_text

@register_query("short_term")
def short_term(params: Dict):
    query_text = f"""{params['query_text']}. SHORT-TERM impact of the stock."""
    return query_text


@register_query("medium_term")
def medium_term(params: Dict):
    query_text = f"""{params['query_text']}. MEDIUM-TERM impact of the stock."""
    return query_text


@register_query("long_term")
def long_term(params: Dict):
    query_text = f"""{params['query_text']}. LONG-TERM impact of the stock."""
    return query_text

def extract_query_type(query_text: str):
    if "SHORT-TERM" in query_text or "short-term" in query_text or "short term" in query_text or "short_term" in query_text:
        return "short_term"
    elif "MEDIUM-TERM" in query_text or "medium-term" in query_text or "medium term" in query_text or "medium_term" in query_text:
        return "medium_term"
    elif "LONG-TERM" in query_text or "long-term" in query_text or "long term" in query_text or "long_term" in query_text:
        return "long_term"
    else:
        return "plain"

if __name__ == "__main__":
    print(QUERY_TYPES)
    params={}
    params['query_text']="""hello"""
    print(QUERY_TYPES["long_term"](params))
   