import requests
from llm_traders.finagent.utils.get_proxy import get_china_free_proxy, get_us_free_proxy, Kuaidaili
class Downloader():
    def __init__(self,
                 *args,
                 use_proxy: str = None,
                 max_retry: int = 5,
                 proxy_pages: int = 5,
                 tunnel: str = None,
                 username: str = None,
                 password: str = None,
                 **kwargs):

        self.use_proxy = use_proxy

        if self.use_proxy:
            self.country = use_proxy
        else:
            self.country = None

        self.max_retry = max_retry if max_retry else 1
        self.proxy_pages = proxy_pages if proxy_pages else 5
        self.tunnel = tunnel
        self.username = username
        self.password = password

        if self.use_proxy:
            if "kuaidaili" in self.country:
                assert self.tunnel and self.username and self.password
                self.proxy_list = Kuaidaili(self.tunnel, self.username, self.password)
            else:
                self.proxy_id = 0
                self.proxy_list = self._update_proxy()
        else:
            self.proxy_list = []

    def _get_proxy(self):
        if self.use_proxy:
            if "kuaidaili" in self.country:
                proxy = self.proxy_list.get_kuaidaili_tunnel_proxy()
                return proxy
            elif len(self.proxy_list) > 0:
                proxy = self.proxy_list[self.proxy_id]
                self.proxy_id += 1
                if self.proxy_id == len(self.proxy_list):
                    self.proxy_id = 0
                return proxy
        else:
            return None

    def _update_proxy(self):
        if "china" in self.country or "China" in self.country:
            return get_china_free_proxy(self.proxy_pages)
        else:
            return get_us_free_proxy(self.proxy_pages)

    def _request_get(self, url, headers=None, verify=None, params=None):
        if headers is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0"
            }
        max_retry = self.max_retry
        proxies = self._get_proxy()
        for _ in range(max_retry):
            try:
                response = requests.get(url=url, proxies=proxies, headers=headers, verify=verify, params=params)
                if response.status_code == 200:
                    break
            except:
                response = None

        if response is not None and response.status_code != 200:
            response = None

        return response

    def _request_post(self, url, headers, json):
        max_retry = self.max_retry
        proxies = self._get_proxy()
        for _ in range(max_retry):
            try:
                response = requests.post(url=url, headers=headers, json=json, proxies=proxies)
                if response.status_code == 200:
                    break
            except:
                response = None

        if response is not None and response.status_code != 200:
            response = None

        return response
