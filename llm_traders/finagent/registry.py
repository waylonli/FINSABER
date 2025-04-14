from mmengine.registry import Registry

DATASET = Registry('data', locations=['llm_traders.finagent.data'])
PROMPT = Registry('prompt', locations=['llm_traders.finagent.prompt'])
AGENT = Registry('agent', locations=['llm_traders.finagent.agent'])
PROVIDER = Registry('provider', locations=['llm_traders.finagent.provider'])
DOWNLOADER = Registry('downloader', locations=['llm_traders.finagent.downloader'])
PROCESSOR = Registry('processor', locations=['llm_traders.finagent.processor'])
ENVIRONMENT = Registry('environment', locations=['llm_traders.finagent.environment'])
MEMORY = Registry('memory', locations=['llm_traders.finagent.memory'])
PLOTS = Registry('plot', locations=['llm_traders.finagent.plots'])