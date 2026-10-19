# -*- coding: utf-8 -*-
"""
V3.7 远程 GPU 算力模块：支持 FRP (8765->38765) 穿透诊断、控制台实时日志、WebSocket 二进制流与 HTTP Keep-Alive 双协议复用。
"""

import os
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from .tracker import UniversalUNetTracker

# ==============================================================================
# 3.5 远程算力服务端与客户端接口 (V3.7: 专为 frpc 8765->38765 穿透优化 + 控制台全链路日志 + 向后兼容 V3.7~V3.7 服务端)
# ==============================================================================
import base64
import hashlib
import http.client
import socket
import ssl
import struct
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

APP_VERSION = "V3.7"
APP_VERSION_SEMVER = "3.7.0"


class FastIPv4ThreadingHTTPServer(ThreadingHTTPServer):
    """原生 IPv4 (AF_INET, 0.0.0.0 / 127.0.0.1) 多线程服务器，确保本机 frpc -> 127.0.0.1:8765 100% 零延迟直通"""
    address_family = socket.AF_INET
    daemon_threads = True
    allow_reuse_address = True


class FastIPv6ThreadingHTTPServer(ThreadingHTTPServer):
    """独立 IPv6 (AF_INET6, [::]) 多线程服务器 (IPV6_V6ONLY=1)，与 IPv4 互不抢占端口"""
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        except Exception:
            pass
        super().server_bind()


def resolve_host_via_public_udp_dns(domain: str, dns_ip: str = "223.5.5.5", timeout: float = 1.8) -> List[str]:
    """
    当本地开启了 Clash/v2rayN TUN Fake-IP (198.18.x.x) 或系统 DNS 污染时，
    直接通过 UDP 53 端口向公共 DNS (223.5.5.5 阿里DNS / 119.29.29.29 腾讯DNS) 查询 A 记录真实公网 IP。
    """
    try:
        tid = os.urandom(2)
        flags = b"\x01\x00"  # Standard query, recursion desired
        counts = b"\x00\x01\x00\x00\x00\x00\x00\x00"
        qname = b"".join(bytes([len(p)]) + p.encode("ascii") for p in domain.strip(".").split(".")) + b"\x00"
        qtype_class = b"\x00\x01\x00\x01"  # TYPE A (1), CLASS IN (1)
        pkt = tid + flags + counts + qname + qtype_class

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(pkt, (dns_ip, 53))
        resp, _ = sock.recvfrom(1024)
        sock.close()
        if len(resp) < 12 or resp[:2] != tid:
            return []
        ancount = struct.unpack("!H", resp[6:8])[0]
        idx = 12
        while idx < len(resp) and resp[idx] != 0:
            if (resp[idx] & 0xC0) == 0xC0:
                idx += 2
                break
            idx += 1 + resp[idx]
        else:
            idx += 1
        idx += 4  # skip QTYPE & QCLASS
        ips: List[str] = []
        for _ in range(ancount):
            if idx >= len(resp):
                break
            if (resp[idx] & 0xC0) == 0xC0:
                idx += 2
            else:
                while idx < len(resp) and resp[idx] != 0:
                    idx += 1 + resp[idx]
                idx += 1
            if idx + 10 > len(resp):
                break
            rtype, _, _, rdlen = struct.unpack("!HHIH", resp[idx:idx + 10])
            idx += 10
            if rtype == 1 and rdlen == 4 and idx + 4 <= len(resp):
                ips.append(".".join(str(b) for b in resp[idx:idx + 4]))
            idx += rdlen
        return ips
    except Exception:
        return []


class RemoteTrackerClient:
    """
    远程 GPU 算力客户端 V3.7 (与 UniversalUNetTracker 接口 100% 兼容，专为 frpc 8765->38765 内网穿透优化):
    - 针对 frpc/frps 穿透链路的五大核心强化:
      1) 全链路控制台实时日志 (`_log`): 将域名解析、Clash Fake-IP 检测、TCP 握手耗时、WebSocket 握手报文、frps/frpc 响应码及错误堆栈实时打印到控制台与 GUI 日志窗。
      2) 100% 兼容旧版 (V3.7~V3.7) 与新版 (V3.7~V3.7) 服务端: `ping_health` 测速采用全版本通用的 `{"cmd": "reset"}` 握手校验 + `GET /health` 双保险，彻底根治向旧版服务端发送 `{"cmd": "ping"}` 触发 `KeyError: 'image_base64'` 断开的问题。
      3) FRP 工作连接 (`WorkConn`) 冷启动自适应重试: 首次访问 `frps:38765` 时若 `frpc` 正在建立反向工作连接，自动退避重试最多 3 次，且绝不向明文 `ws://` 端口误发二进制 TLS 握手导致服务端阻塞。
      4) Clash / v2rayN Fake-IP (`198.18.x.x`) 自动识别与绕过: 自动向 `223.5.5.5` / `119.29.29.29` 查询真实公网 IP 直连。
      5) 支持 RFC 6455 `Ping(0x09)/Pong(0x0A)` 自动应答与 HTTP Keep-Alive `/track` 自动降级。
    """

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8765/ws",
        fps: float = 30.0,
        perf_mode: str = "ultra",
        jpeg_quality: int = 92,
        timeout: float = 8.0,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.server_url = (server_url or "ws://127.0.0.1:8765/ws").strip()
        self.fps = float(fps)
        self.perf_mode = perf_mode
        self.jpeg_quality = int(jpeg_quality)
        self.timeout = float(timeout)
        self.log_callback = log_callback

        self.ref_w = 104.0
        self.ref_h = 104.0
        self.shape_desc = "REMOTE_WAITING"
        self.canonical_contour: Optional[np.ndarray] = None
        self.last_unet_prob_s: Optional[np.ndarray] = None
        self.remote_device_name = "Remote GPU (未连接)"
        self.last_rtt_ms = 0.0
        self.connected = False
        self.last_error = ""

        self.ema_alpha = 0.50
        self.gate_radius_half = 44.0
        self.max_step_px = 21.0

        self._lock = threading.Lock()
        self._ws_sock: Optional[socket.socket] = None
        self._ws_rx_buf: bytearray = bytearray()
        self._http_conn: Optional[http.client.HTTPConnection] = None
        self._force_http_fallback: bool = False
        self._ssl_ctx = ssl._create_unverified_context()
        self._parse_url(self.server_url)

    def _log(self, msg: str) -> None:
        """实时打印连接诊断信息到控制台 (stdout) 并同步推送至 GUI 日志面板"""
        ts = time.strftime("%H:%M:%S")
        ms = int((time.time() % 1.0) * 1000)
        line = f"[{ts}.{ms:03d}] [RemoteClient] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            try:
                print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
            except Exception:
                pass
        if self.log_callback is not None:
            try:
                self.log_callback(line)
            except Exception:
                pass

    @staticmethod
    def _clean_url_string(url: str) -> str:
        """清理从微信/QQ/浏览器复制 URL 时可能夹带的零宽字符、全角冒号、首尾空格与引号"""
        if not url:
            return "ws://127.0.0.1:8765/ws"
        for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\r", "\n", "\t", "\"", "'"):
            url = url.replace(ch, "")
        url = url.strip().replace("：", ":")
        return url

    def _parse_url(self, url: str) -> None:
        raw_in = url
        url = self._clean_url_string(url)
        self.server_url = url
        if "://" not in url:
            url = "ws://" + url
        parsed = urlparse(url)
        self.scheme = (parsed.scheme or "ws").lower()
        self.host = parsed.hostname or "127.0.0.1"
        try:
            self.port = int(parsed.port) if parsed.port else (443 if self.scheme in ("https", "wss") else 8765)
        except Exception:
            self.port = 8765
        self.path = parsed.path if parsed.path else ("/ws" if self.scheme in ("ws", "wss") else "/track")
        if parsed.query:
            self.path = f"{self.path}?{parsed.query}"
        self._force_http_fallback = False
        if raw_in != url:
            self._log(f"URL 已自动规范化: '{raw_in}' -> '{self.scheme}://{self.host}:{self.port}{self.path}'")

    def set_server_url(self, url: str) -> None:
        with self._lock:
            self.close()
            self._parse_url(url)

    def close(self) -> None:
        self.connected = False
        self._ws_rx_buf.clear()
        if self._ws_sock is not None:
            try:
                self._ws_sock.close()
            except Exception:
                pass
            self._ws_sock = None
        if self._http_conn is not None:
            try:
                self._http_conn.close()
            except Exception:
                pass
            self._http_conn = None

    def _resolve_candidate_Targets(self) -> List[Tuple[int, Tuple]]:
        """
        解析目标主机候选地址列表 (优先 IPv4 AF_INET 以完美匹配 frps/frpc，同时检测 Clash Fake-IP 198.18.x.x 并查询真实 IP)
        """
        candidates: List[Tuple[int, Tuple]] = []
        v4_list: List[Tuple[int, Tuple]] = []
        v6_list: List[Tuple[int, Tuple]] = []
        fake_ip_detected = False

        try:
            infos = socket.getaddrinfo(self.host, self.port, socket.AF_UNSPEC, socket.SOCK_STREAM)
            seen = set()
            for family, _, _, _, sockaddr in infos:
                ip_str = sockaddr[0]
                if (family, ip_str) in seen:
                    continue
                seen.add((family, ip_str))
                if ip_str.startswith("198.18.") or ip_str.startswith("198.19."):
                    fake_ip_detected = True
                if family == socket.AF_INET:
                    v4_list.append((family, sockaddr))
                elif family == socket.AF_INET6:
                    v6_list.append((family, sockaddr))
        except socket.gaierror as e:
            self._log(f"⚠️ 系统 DNS 解析 '{self.host}' 失败: {e}")

        # 若域名非纯 IP，且检测到 Clash Fake-IP 或系统 DNS 未返回结果，主动查询 223.5.5.5 / 119.29.29.29
        is_literal_ip = False
        try:
            socket.inet_aton(self.host)
            is_literal_ip = True
        except Exception:
            is_literal_ip = (":" in self.host)

        if not is_literal_ip:
            public_ips: List[str] = []
            for dns_srv in ("223.5.5.5", "119.29.29.29"):
                res_ips = resolve_host_via_public_udp_dns(self.host, dns_ip=dns_srv)
                for rip in res_ips:
                    if rip not in public_ips and not rip.startswith("198.18."):
                        public_ips.append(rip)
                if public_ips:
                    break
            if fake_ip_detected and public_ips:
                self._log(
                    f"⚠️ 检测到本地代理 Fake-IP ({[x[1][0] for x in v4_list]})! "
                    f"已通过公共 DNS 获取 '{self.host}' 真实公网 IP: {public_ips} (优先直连真实 IP)"
                )
                real_v4 = [(socket.AF_INET, (rip, self.port)) for rip in public_ips]
                v4_list = real_v4 + v4_list
            elif public_ips:
                existing_ips = {x[1][0] for x in v4_list}
                for rip in public_ips:
                    if rip not in existing_ips:
                        v4_list.append((socket.AF_INET, (rip, self.port)))

        # FRP (frps) 绝大多数监听在 IPv4 上，因此优先尝试 IPv4，其次 IPv6
        candidates = v4_list + v6_list
        if not candidates:
            raise ConnectionError(f"无法解析域名 '{self.host}'，请检查网络连接或直接填写公网 IP")
        return candidates

    def _connect_raw_socket(self, timeout: Optional[float] = None, use_tls: Optional[bool] = None, verbose: bool = True) -> socket.socket:
        """建立底层 TCP 连接 (优先 IPv4，打印每个候选 IP 的连接耗时或具体的操作系统错误码)"""
        tm = float(timeout if timeout is not None else self.timeout)
        tls_flag = (self.scheme in ("wss", "https")) if use_tls is None else bool(use_tls)
        candidates = self._resolve_candidate_Targets()
        if verbose:
            ip_summary = ", ".join(f"{sa[0]}:{sa[1]}" for _, sa in candidates)
            self._log(f"DNS 解析 '{self.host}:{self.port}' -> 候选地址: [{ip_summary}]")

        last_err = None
        for family, sockaddr in candidates:
            s = None
            t_c0 = time.perf_counter()
            try:
                s = socket.socket(family, socket.SOCK_STREAM)
                s.settimeout(tm)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except Exception:
                    pass
                s.connect(sockaddr)
                tcp_ms = (time.perf_counter() - t_c0) * 1000.0
                if verbose:
                    self._log(f"✅ TCP 握手成功 -> {sockaddr[0]}:{sockaddr[1]} (耗时 {tcp_ms:.1f}ms)")
                if tls_flag:
                    s = self._ssl_ctx.wrap_socket(s, server_hostname=self.host)
                    if verbose:
                        self._log(f"✅ TLS/SSL 握手成功 ({self.host})")
                return s
            except Exception as e:
                tcp_ms = (time.perf_counter() - t_c0) * 1000.0
                last_err = e
                if verbose:
                    self._log(f"❌ TCP 连接 {sockaddr[0]}:{sockaddr[1]} 失败 ({tcp_ms:.1f}ms): {type(e).__name__}: {e}")
                if s is not None:
                    try:
                        s.close()
                    except Exception:
                        pass
        raise ConnectionError(f"TCP 无法连接到 {self.host}:{self.port} -> {last_err}")

    def ping_health(self) -> Tuple[bool, str, float]:
        """
        全方位测试远程算力连通性 (专为 frpc 8765->38765 穿透及全版本服务端兼容设计):
        - 打印完整的控制台诊断日志；
        - 先尝试 HTTP GET /health (兼顾 V3.7~V3.7 服务端获取显卡名称)；
        - 再尝试 WebSocket `/ws` 握手 + 发送全版本兼容的 `{"cmd": "reset"}` 指令 (支持 frpc WorkConn 自动退避重试)。
        """
        with self._lock:
            self.close()
            self._log("=" * 68)
            self._log(f"🚀 开始远程算力测速连接: URL='{self.server_url}'")
            self._log(f"   解析参数: scheme='{self.scheme}', host='{self.host}', port={self.port}, path='{self.path}'")
            for env_k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
                if os.environ.get(env_k):
                    self._log(f"   [环境提示] 检测到系统环境变量 {env_k}={os.environ.get(env_k)} (本客户端采用原生 Socket 直连，不受 HTTP_PROXY 干扰)")

            use_tls = self.scheme in ("wss", "https")
            http_dev_name: Optional[str] = None
            http_rtt_ms: float = 0.0

            # 步骤 1: 先通过原始 Socket 发送 HTTP GET /health 探测 (测试 frpc 8765->38765 链路并读取远端显卡名称)
            for attempt in range(1, 3):
                sock_h = None
                try:
                    t_h0 = time.perf_counter()
                    sock_h = self._connect_raw_socket(timeout=5.0, use_tls=use_tls, verbose=(attempt == 1))
                    req_h = (
                        f"GET /health HTTP/1.1\r\n"
                        f"Host: {self.host}:{self.port}\r\n"
                        f"User-Agent: UNetTrackerClient/{APP_VERSION}\r\n"
                        f"Accept: application/json\r\n"
                        f"Connection: close\r\n\r\n"
                    )
                    sock_h.sendall(req_h.encode("ascii"))
                    raw_resp = bytearray()
                    while True:
                        chunk = sock_h.recv(4096)
                        if not chunk:
                            break
                        raw_resp.extend(chunk)
                        if b"\r\n\r\n" in raw_resp and len(raw_resp) > 64:
                            # 检查是否已收完 JSON 主体
                            if raw_resp.endswith(b"}") or raw_resp.endswith(b"}\n"):
                                break
                    sock_h.close()
                    sock_h = None
                    http_rtt_ms = (time.perf_counter() - t_h0) * 1000.0

                    if not raw_resp:
                        self._log(
                            f"⚠️ [HTTP探测 第{attempt}次] 远端 {self.host}:{self.port} 接受了 TCP 连接但返回了 0 字节空响应 (EOF)!\n"
                            f"   👉 [FRP 穿透排查提示]: 这说明公网 frps ({self.port}端口) 已打通，但内网 frpc 转发到本机 127.0.0.1:8765 失败！\n"
                            f"      请检查: 1) 算力机上是否已运行 `--server --port 8765`；2) frpc 控制台是否报错 'connectex: No connection could be made'。"
                        )
                        time.sleep(0.25)
                        continue

                    header_part, _, body_part = bytes(raw_resp).partition(b"\r\n\r\n")
                    status_line = header_part.split(b"\r\n", 1)[0].decode("latin1", errors="ignore")
                    body_txt = body_part.decode("utf-8", errors="ignore").strip()
                    self._log(f"📥 [HTTP /health 响应] 状态行='{status_line}', 耗时={http_rtt_ms:.1f}ms, 内容={body_txt[:120]}")
                    if "200" in status_line and ("{" in body_txt):
                        j_start = body_txt.find("{")
                        j_end = body_txt.rfind("}") + 1
                        info = json.loads(body_txt[j_start:j_end])
                        http_dev_name = info.get("device", "Remote GPU")
                        srv_ver = info.get("version", "V3.x")
                        self._log(f"✅ HTTP 健康检查通过! 远端算力设备='{http_dev_name}', 服务端版本={srv_ver}")
                        break
                except Exception as e_h:
                    self._log(f"⚠️ [HTTP探测 第{attempt}次] 异常: {e_h}")
                    self.last_error = str(e_h)
                    if sock_h is not None:
                        try:
                            sock_h.close()
                        except Exception:
                            pass
                    time.sleep(0.25)

            # 步骤 2: 若用户配置为 ws:// 或 wss://，测试 WebSocket 握手 + 发送全版本兼容的 {"cmd": "reset"} 指令
            if self.scheme in ("ws", "wss"):
                for ws_attempt in range(1, 3):
                    try:
                        t_ws = time.perf_counter()
                        self._ensure_ws_connected(timeout=6.0, use_tls=use_tls, verbose=True)
                        # 注意: 发送 {"cmd": "reset"} 而非 {"cmd": "ping"}，因为 V3.7~V3.7 服务端仅识别 "reset"，发送 "ping" 会触发 KeyError: 'image_base64' 导致连接被服务端断开！
                        reset_cmd = json.dumps({"cmd": "reset", "fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                        self._ws_send_frame(self._ws_sock, reset_cmd, opcode=0x01)
                        resp_bytes = self._ws_recv_frame()
                        rtt_ms = (time.perf_counter() - t_ws) * 1000.0
                        resp_txt = resp_bytes.decode("utf-8", errors="ignore")
                        self._log(f"📥 [WebSocket 校验响应] 耗时={rtt_ms:.1f}ms, 内容={resp_txt[:120]}")
                        info = json.loads(resp_txt)
                        dev = info.get("device") or http_dev_name or "Remote GPU"
                        self._force_http_fallback = False
                        self.remote_device_name = f"远程 {dev}"
                        self.last_rtt_ms = rtt_ms
                        self.connected = True
                        self.last_error = ""
                        self._log(f"🎉 WebSocket 远程算力连接成功! 设备={dev}, RTT={rtt_ms:.1f}ms")
                        self._log("=" * 68)
                        return True, f"在线(WS): {dev} ({rtt_ms:.1f}ms)", rtt_ms
                    except Exception as e_ws:
                        self.last_error = str(e_ws)
                        self._log(f"❌ [WebSocket 第{ws_attempt}次握手/校验失败]: {e_ws}")
                        self.close()
                        time.sleep(0.25)

                # 若 HTTP /health 已经成功，说明 frpc 穿透正常但中间件拦截了 WebSocket Upgrade，自动无缝启用 HTTP Keep-Alive 模式！
                if http_dev_name is not None:
                    self._force_http_fallback = True
                    self.remote_device_name = f"远程 {http_dev_name}"
                    self.last_rtt_ms = http_rtt_ms
                    self.connected = True
                    self.last_error = ""
                    self._log(f"✅ 已自动切换为 HTTP Keep-Alive 穿透模式! 设备={http_dev_name}, RTT={http_rtt_ms:.1f}ms")
                    self._log("=" * 68)
                    return True, f"在线(HTTP穿透): {http_dev_name} ({http_rtt_ms:.1f}ms)", http_rtt_ms
            else:
                if http_dev_name is not None:
                    self.remote_device_name = f"远程 {http_dev_name}"
                    self.last_rtt_ms = http_rtt_ms
                    self.connected = True
                    self.last_error = ""
                    self._log(f"🎉 HTTP 远程算力连接成功! 设备={http_dev_name}, RTT={http_rtt_ms:.1f}ms")
                    self._log("=" * 68)
                    return True, f"在线(HTTP): {http_dev_name} ({http_rtt_ms:.1f}ms)", http_rtt_ms

            err_msg = self.last_error or "目标端口未响应"
            if "10061" in err_msg or "Connection refused" in err_msg:
                hint = f"端口 {self.port} 拒绝连接 (请检查 frps/服务端是否启动)"
            elif "空响应" in err_msg or "EOF" in err_msg or "远端关闭" in err_msg:
                hint = f"frps:{self.port} 已通，但内网 frpc 未连上本机 8765 服务端"
            elif "timed out" in err_msg or "Timeout" in err_msg:
                hint = f"连接 {self.host}:{self.port} 超时 (请检查 frps 端口/防火墙)"
            else:
                hint = err_msg[:46]
            self._log(f"❌ 远程算力连接最终失败: {hint} (原始错误: {err_msg})")
            self._log("=" * 68)
            return False, f"失败: {hint}", 0.0

    def _ensure_ws_connected(self, timeout: Optional[float] = None, use_tls: Optional[bool] = None, verbose: bool = False) -> None:
        if self._ws_sock is not None:
            return
        sock = self._connect_raw_socket(timeout=timeout, use_tls=use_tls, verbose=verbose)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        ws_path = self.path if self.path else "/ws"
        # 标准 Host 头: 对非标准端口附带 :port，兼容 frps vhost 与 tcp 转发
        req = (
            f"GET {ws_path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"User-Agent: UNetTrackerClient/{APP_VERSION}\r\n\r\n"
        )
        if verbose:
            self._log(f"📤 发送 WebSocket Upgrade 握手: GET {ws_path} HTTP/1.1 (Host: {self.host}:{self.port})")
        sock.sendall(req.encode("ascii"))
        resp = bytearray()
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(1024)
            if not chunk:
                sock.close()
                raise ConnectionError(
                    f"WebSocket 握手收到空响应 (EOF)! 公网 {self.host}:{self.port} (frps) 已断开，请确认算力机已启动 8765 端口服务端且 frpc 映射正常"
                )
            resp.extend(chunk)
            if len(resp) > 16384:
                sock.close()
                raise ConnectionError("WebSocket 握手响应头过长")

        sep_idx = resp.index(b"\r\n\r\n") + 4
        header_bytes = bytes(resp[:sep_idx])
        extra_bytes = bytes(resp[sep_idx:])
        first_line = header_bytes.split(b"\r\n", 1)[0].decode("latin1", errors="ignore")
        if verbose:
            self._log(f"📥 收到 WebSocket 握手响应头: '{first_line}'")

        if "101" not in first_line:
            body_preview = extra_bytes[:120].decode("utf-8", errors="ignore")
            sock.close()
            if "200" in first_line:
                self._force_http_fallback = True
            raise ConnectionError(f"WebSocket 握手未返回 101 (状态行: {first_line}, 内容: {body_preview})")

        self._ws_sock = sock
        self._ws_rx_buf = bytearray(extra_bytes)
        self.connected = True

    @staticmethod
    def _ws_send_frame(sock: socket.socket, payload: bytes, opcode: int = 0x02) -> None:
        """客户端向服务端发送带掩码的 RFC 6455 WebSocket 数据帧 (0x01=Text, 0x02=Binary, 0x0A=Pong)"""
        header = bytearray([0x80 | (opcode & 0x0F)])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask_key = os.urandom(4)
        header.extend(mask_key)
        masked = bytearray(payload)
        for i in range(length):
            masked[i] ^= mask_key[i & 3]
        sock.sendall(header + masked)

    def _ws_recv_exact_buffered(self, n: int) -> bytes:
        while len(self._ws_rx_buf) < n:
            if self._ws_sock is None:
                raise ConnectionError("WebSocket 套接字未连接")
            chunk = self._ws_sock.recv(max(4096, n - len(self._ws_rx_buf)))
            if not chunk:
                raise ConnectionError("WebSocket 连接已由远端关闭 (EOF)")
            self._ws_rx_buf.extend(chunk)
        out = bytes(self._ws_rx_buf[:n])
        del self._ws_rx_buf[:n]
        return out

    @staticmethod
    def _ws_recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("WebSocket 连接已断开")
            buf.extend(chunk)
        return bytes(buf)

    @staticmethod
    def _ws_recv_from_rfile(rfile, n: int) -> bytes:
        """服务端专用: 从 BaseHTTPRequestHandler.rfile 缓冲流中精确读取 n 字节，防止 frpc 粘包数据滞留在 rfile 缓冲区中丢失"""
        buf = bytearray()
        while len(buf) < n:
            chunk = rfile.read(n - len(buf))
            if not chunk:
                raise ConnectionError("WebSocket 客户端流已关闭")
            buf.extend(chunk)
        return bytes(buf)

    def _ws_recv_frame(self, sock: Optional[socket.socket] = None) -> bytes:
        """读取完整的 WebSocket 数据帧，自动响应反向代理的 Ping(0x09) 帧并忽略 Pong(0x0A) 帧"""
        while True:
            b1, b2 = self._ws_recv_exact_buffered(2)
            opcode = b1 & 0x0F
            masked = (b2 & 0x80) != 0
            length = b2 & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._ws_recv_exact_buffered(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._ws_recv_exact_buffered(8))[0]
            mask_key = self._ws_recv_exact_buffered(4) if masked else None
            data = bytearray(self._ws_recv_exact_buffered(length))
            if masked and mask_key:
                for i in range(length):
                    data[i] ^= mask_key[i & 3]
            payload = bytes(data)

            if opcode == 0x09:  # Ping -> 自动回复 Pong (0x0A)
                if self._ws_sock is not None:
                    try:
                        self._ws_send_frame(self._ws_sock, payload, opcode=0x0A)
                    except Exception:
                        pass
                continue
            if opcode == 0x0A:  # Pong -> 忽略
                continue
            if opcode == 0x08:  # Close
                raise ConnectionError("远端主动关闭了 WebSocket 会话 (Opcode 0x08)")
            return payload

    def reset(self) -> None:
        self.canonical_contour = None
        self.last_unet_prob_s = None
        with self._lock:
            for attempt in range(2):
                try:
                    if self.scheme in ("ws", "wss") and (not self._force_http_fallback):
                        self._ensure_ws_connected(verbose=(attempt == 1))
                        cmd = json.dumps({"cmd": "reset", "fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                        self._ws_send_frame(self._ws_sock, cmd, opcode=0x01)
                        _ = self._ws_recv_frame()
                    else:
                        conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=self._ssl_ctx) if self.scheme in ("https", "wss") else http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                        body = json.dumps({"fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                        conn.request(
                            "POST",
                            "/reset",
                            body=body,
                            headers={"Content-Type": "application/json", "Host": f"{self.host}:{self.port}"},
                        )
                        _ = conn.getresponse().read()
                        conn.close()
                    return
                except Exception as e:
                    if attempt == 1:
                        self._log(f"⚠️ 远程 reset 失败: {e}")
                    self.close()

    def get_current_polygon_full(self, cx: float, cy: float, angle_deg: float) -> np.ndarray:
        if self.canonical_contour is None:
            box = cv2.boxPoints(((cx, cy), (self.ref_w, self.ref_h), angle_deg))
            return box.astype(np.int32)
        rad = np.radians(angle_deg)
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        rot_pts = (self.canonical_contour @ R.T) + np.array([[cx, cy]], dtype=np.float32)
        return np.round(rot_pts).astype(np.int32)

    def _send_track_http(self, jpg_bytes: bytes, dt_val: float) -> dict:
        if self._http_conn is None:
            is_https = self.scheme in ("https", "wss")
            self._http_conn = (
                http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=self._ssl_ctx)
                if is_https else http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            )
        post_path = "/track" if self.path in ("", "/", "/ws") else self.path
        headers = {
            "Host": f"{self.host}:{self.port}",
            "Content-Type": "image/jpeg",
            "Connection": "keep-alive",
            "X-Dt-Sec": f"{dt_val:.6f}",
            "X-Fps": f"{self.fps:.2f}",
            "X-Mode": self.perf_mode,
        }
        self._http_conn.request("POST", post_path, body=jpg_bytes, headers=headers)
        resp = self._http_conn.getresponse()
        resp_body = resp.read()
        if resp.status != 200:
            raise ConnectionError(f"HTTP {resp.status}: {resp_body[:100]!r}")
        return json.loads(resp_body.decode("utf-8"))

    def update(self, frame_bgr: np.ndarray, dt_sec: Optional[float] = None) -> dict:
        t0 = time.perf_counter()
        ok, jpg_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ValueError("图像 JPEG 编码失败")
        jpg_bytes = jpg_buf.tobytes()
        dt_val = float(dt_sec) if (dt_sec is not None and dt_sec > 0) else (1.0 / max(1.0, self.fps))

        with self._lock:
            res = None
            last_exc = None
            for attempt in range(2):
                try:
                    if self.scheme in ("ws", "wss") and (not self._force_http_fallback):
                        self._ensure_ws_connected(verbose=(attempt == 1))
                        pkt = struct.pack("!ff", dt_val, float(self.fps)) + jpg_bytes
                        self._ws_send_frame(self._ws_sock, pkt, opcode=0x02)
                        resp_bytes = self._ws_recv_frame()
                        res = json.loads(resp_bytes.decode("utf-8"))
                    else:
                        res = self._send_track_http(jpg_bytes, dt_val)
                    break
                except Exception as e:
                    last_exc = e
                    self._log(f"⚠️ 帧传输异常 (第{attempt+1}次尝试): {e}")
                    self.close()
                    if self._force_http_fallback:
                        try:
                            res = self._send_track_http(jpg_bytes, dt_val)
                            break
                        except Exception as e2:
                            last_exc = e2
                            self.close()

            if res is None:
                H, W = frame_bgr.shape[:2]
                err_str = str(last_exc) if last_exc else "Unknown"
                self.last_error = err_str
                self._log(f"❌ 远程推理失败: {err_str}")
                return {
                    "frame_idx": 0, "phase": "REMOTE_ERR", "shape": f"ERR:{err_str[:22]}",
                    "cx": W * 0.5, "cy": H * 0.5, "w": self.ref_w, "h": self.ref_h, "angle": 0.0,
                    "vx": 0.0, "vy": 0.0, "omega": 0.0, "snr": 0.0, "conf": 0.0,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0, "n_step": 1.0,
                    "device": f"远程断开: {err_str[:28]}"
                }

        rtt_ms = (time.perf_counter() - t0) * 1000.0
        self.last_rtt_ms = rtt_ms
        self.connected = True
        self.ref_w = float(res.get("w", self.ref_w))
        self.ref_h = float(res.get("h", self.ref_h))
        self.shape_desc = str(res.get("shape", self.shape_desc))
        if "poly_pts" in res and res["poly_pts"] is not None:
            self.canonical_contour = np.array(res["poly_pts"], dtype=np.float32)
        remote_dev = res.get("device", "Remote GPU")
        self.remote_device_name = f"远程 {remote_dev} (RTT:{rtt_ms:.1f}ms)"
        res["remote_eval_ms"] = res.get("latency_ms", 0.0)
        res["latency_ms"] = rtt_ms
        res["device"] = self.remote_device_name
        return res


class RemoteTrackerServer:
    """
    全图形自适应 U-Net 远程算力服务端 V3.7:
    - 采用原生 IPv4 (`0.0.0.0:8765`, `AF_INET`) 作为主监听套接字，100% 保证 Windows/Linux 下 `frpc` (`127.0.0.1:8765 -> 38765`) 零障碍连通；
    - 同时后台自动挂载独立 IPv6 (`[::]:8765`) 与 `38765` 备用端口监听；
    - WebSocket 读取采用 `self.rfile` 缓冲流无缝衔接，彻底消除 `frpc` 隧道合并发送握手与首帧时的粘包丢帧；
    - 控制台实时打印客户端/frpc 接入日志。
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        device: str = "cuda",
        gpu_id: int = 0,
        perf_mode: str = "ultra",
        fps: float = 30.0,
    ):
        self.host = host
        self.port = int(port)
        self.tracker = UniversalUNetTracker(fps=fps, perf_mode=perf_mode, device=device, gpu_id=gpu_id)
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._extra_servers: List[ThreadingHTTPServer] = []

    def process_frame_bytes(self, jpg_bytes: bytes, dt_sec: Optional[float] = None, fps: Optional[float] = None) -> dict:
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("无法解码客户端发送的 JPEG 图像")
        with self._lock:
            if fps is not None and fps > 1.0:
                self.tracker.fps = float(fps)
            st = self.tracker.update(frame_bgr, dt_sec=dt_sec)
            if self.tracker.canonical_contour is not None and st["frame_idx"] <= 25:
                st["poly_pts"] = self.tracker.canonical_contour.tolist()
            return st

    def _build_handler_class(self):
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def _send_json(self, obj: dict, status: int = 200) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                upg = self.headers.get("Upgrade", "").lower()
                conn_hdr = self.headers.get("Connection", "").lower()
                if upg == "websocket" or ("upgrade" in conn_hdr and self.headers.get("Sec-WebSocket-Key")):
                    self._handle_websocket()
                    return
                print(f"[{time.strftime('%H:%M:%S')}] [Server] 收到健康检查 GET {self.path} 来自 {self.client_address}", flush=True)
                self._send_json({
                    "status": "ok",
                    "version": APP_VERSION,
                    "device": server_ref.tracker.cuda_backend.device_name,
                    "mode": server_ref.tracker.perf_mode,
                    "frame_idx": server_ref.tracker.frame_idx,
                })

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b""

                if self.path.startswith("/reset"):
                    with server_ref._lock:
                        if raw_body:
                            try:
                                cfg = json.loads(raw_body.decode("utf-8"))
                                if "fps" in cfg:
                                    server_ref.tracker.fps = float(cfg["fps"])
                                if "mode" in cfg:
                                    server_ref.tracker.perf_mode = str(cfg["mode"])
                            except Exception:
                                pass
                        server_ref.tracker.reset()
                    print(f"[{time.strftime('%H:%M:%S')}] [Server] 收到 HTTP /reset 来自 {self.client_address}", flush=True)
                    self._send_json({
                        "status": "reset_ok",
                        "version": APP_VERSION,
                        "device": server_ref.tracker.cuda_backend.device_name,
                    })
                    return

                try:
                    ctype = self.headers.get("Content-Type", "").lower()
                    if "application/json" in ctype:
                        payload = json.loads(raw_body.decode("utf-8"))
                        if payload.get("cmd") == "reset":
                            with server_ref._lock:
                                server_ref.tracker.reset()
                            self._send_json({
                                "status": "reset_ok",
                                "version": APP_VERSION,
                                "device": server_ref.tracker.cuda_backend.device_name,
                            })
                            return
                        if payload.get("cmd") == "ping":
                            self._send_json({
                                "status": "ok",
                                "version": APP_VERSION,
                                "device": server_ref.tracker.cuda_backend.device_name,
                            })
                            return
                        jpg_bytes = base64.b64decode(payload["image_base64"])
                        dt_sec = payload.get("dt_sec", None)
                        fps_val = payload.get("fps", None)
                    else:
                        jpg_bytes = raw_body
                        dt_hdr = self.headers.get("X-Dt-Sec", None)
                        fps_hdr = self.headers.get("X-Fps", None)
                        dt_sec = float(dt_hdr) if dt_hdr else None
                        fps_val = float(fps_hdr) if fps_hdr else None
                    st = server_ref.process_frame_bytes(jpg_bytes, dt_sec=dt_sec, fps=fps_val)
                    self._send_json(st)
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [Server] POST 处理异常: {e}", flush=True)
                    self._send_json({"error": str(e)}, status=400)

            @staticmethod
            def _ws_send_server_frame(conn: socket.socket, payload_bytes: bytes, opcode: int = 0x01) -> None:
                hdr = bytearray([0x80 | (opcode & 0x0F)])
                rlen = len(payload_bytes)
                if rlen < 126:
                    hdr.append(rlen)
                elif rlen < 65536:
                    hdr.append(126)
                    hdr.extend(struct.pack("!H", rlen))
                else:
                    hdr.append(127)
                    hdr.extend(struct.pack("!Q", rlen))
                conn.sendall(hdr + payload_bytes)

            def _handle_websocket(self) -> None:
                key = self.headers.get("Sec-WebSocket-Key", "").strip()
                accept = base64.b64encode(
                    hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
                ).decode("ascii")
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()

                conn = self.connection
                try:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                print(f"[{time.strftime('%H:%M:%S')}] [Server] ✅ WebSocket 隧道已建立! 客户端={self.client_address}, 路径={self.path}", flush=True)
                try:
                    while True:
                        b1_b2 = RemoteTrackerClient._ws_recv_from_rfile(self.rfile, 2)
                        opcode = b1_b2[0] & 0x0F
                        if opcode == 0x08:  # Close
                            print(f"[{time.strftime('%H:%M:%S')}] [Server] 客户端关闭了 WebSocket 连接: {self.client_address}", flush=True)
                            break
                        masked = (b1_b2[1] & 0x80) != 0
                        length = b1_b2[1] & 0x7F
                        if length == 126:
                            length = struct.unpack("!H", RemoteTrackerClient._ws_recv_from_rfile(self.rfile, 2))[0]
                        elif length == 127:
                            length = struct.unpack("!Q", RemoteTrackerClient._ws_recv_from_rfile(self.rfile, 8))[0]
                        mask_key = RemoteTrackerClient._ws_recv_from_rfile(self.rfile, 4) if masked else None
                        payload = bytearray(RemoteTrackerClient._ws_recv_from_rfile(self.rfile, length))
                        if masked and mask_key:
                            for i in range(length):
                                payload[i] ^= mask_key[i & 3]

                        if opcode == 0x09:  # Ping -> Pong
                            self._ws_send_server_frame(conn, bytes(payload), opcode=0x0A)
                            continue
                        if opcode == 0x0A:  # Pong
                            continue

                        if opcode == 0x02:  # Binary frame: 8-byte float32(dt_sec, fps) + JPEG bytes
                            dt_val, fps_val = struct.unpack("!ff", payload[:8])
                            jpg_bytes = bytes(payload[8:])
                            st = server_ref.process_frame_bytes(jpg_bytes, dt_sec=dt_val, fps=fps_val)
                            resp_bytes = json.dumps(st, ensure_ascii=False).encode("utf-8")
                        else:  # Text JSON frame
                            txt = payload.decode("utf-8", errors="ignore").strip()
                            if not txt:
                                continue
                            msg = json.loads(txt)
                            cmd_type = msg.get("cmd", "")
                            if cmd_type == "ping":
                                resp_bytes = json.dumps({
                                    "status": "ok",
                                    "version": APP_VERSION,
                                    "device": server_ref.tracker.cuda_backend.device_name,
                                    "mode": server_ref.tracker.perf_mode,
                                }, ensure_ascii=False).encode("utf-8")
                            elif cmd_type == "reset":
                                with server_ref._lock:
                                    if "fps" in msg:
                                        server_ref.tracker.fps = float(msg["fps"])
                                    if "mode" in msg:
                                        server_ref.tracker.perf_mode = str(msg["mode"])
                                    server_ref.tracker.reset()
                                resp_bytes = json.dumps({
                                    "status": "reset_ok",
                                    "version": APP_VERSION,
                                    "device": server_ref.tracker.cuda_backend.device_name,
                                }, ensure_ascii=False).encode("utf-8")
                            elif "image_base64" in msg:
                                jpg_bytes = base64.b64decode(msg["image_base64"])
                                st = server_ref.process_frame_bytes(
                                    jpg_bytes, dt_sec=msg.get("dt_sec"), fps=msg.get("fps")
                                )
                                resp_bytes = json.dumps(st, ensure_ascii=False).encode("utf-8")
                            else:
                                resp_bytes = json.dumps({
                                    "status": "ok",
                                    "version": APP_VERSION,
                                    "device": server_ref.tracker.cuda_backend.device_name,
                                }, ensure_ascii=False).encode("utf-8")

                        self._ws_send_server_frame(conn, resp_bytes, opcode=0x01)
                except Exception as e_ws:
                    print(f"[{time.strftime('%H:%M:%S')}] [Server] WebSocket 会话结束 ({self.client_address}): {e_ws}", flush=True)

        return _Handler

    def start_forever(self) -> None:
        handler_cls = self._build_handler_class()
        bind_ipv4 = "0.0.0.0" if self.host in ("0.0.0.0", "::", "", "*") else self.host
        # 1. 主监听器: 原生 IPv4 (AF_INET) 0.0.0.0:port，确保 frpc -> 127.0.0.1:8765 100% 稳定连通
        self._httpd = FastIPv4ThreadingHTTPServer((bind_ipv4, self.port), handler_cls)

        # 2. 伴随监听器: 独立 IPv6 [::]:port + 备用端口 (38765 / 8765)
        aux_port = 38765 if self.port == 8765 else (8765 if self.port == 38765 else None)
        bound_Descs = [f"IPv4 0.0.0.0:{self.port}"]

        if self.host in ("0.0.0.0", "::", "", "*"):
            try:
                srv_v6 = FastIPv6ThreadingHTTPServer(("::", self.port), handler_cls)
                threading.Thread(target=srv_v6.serve_forever, daemon=True).start()
                self._extra_servers.append(srv_v6)
                bound_Descs.append(f"IPv6 [::]:{self.port}")
            except Exception:
                pass

        if aux_port is not None:
            try:
                srv_aux4 = FastIPv4ThreadingHTTPServer((bind_ipv4, aux_port), handler_cls)
                threading.Thread(target=srv_aux4.serve_forever, daemon=True).start()
                self._extra_servers.append(srv_aux4)
                bound_Descs.append(f"IPv4 0.0.0.0:{aux_port}")
            except Exception:
                pass
            if self.host in ("0.0.0.0", "::", "", "*"):
                try:
                    srv_aux6 = FastIPv6ThreadingHTTPServer(("::", aux_port), handler_cls)
                    threading.Thread(target=srv_aux6.serve_forever, daemon=True).start()
                    self._extra_servers.append(srv_aux6)
                    bound_Descs.append(f"IPv6 [::]:{aux_port}")
                except Exception:
                    pass

        print("=" * 80, flush=True)
        print(f"🚀 U-Net 远程算力服务端 {APP_VERSION} (FRP 8765->38765 穿透增强 + IPv4/IPv6 双栈版) 已启动!", flush=True)
        print(f"   加速硬件 : {self.tracker.cuda_backend.device_name}", flush=True)
        print(f"   计算档位 : {self.tracker.perf_mode.upper()}", flush=True)
        print(f"   监听列表 : {', '.join(bound_Descs)}", flush=True)
        print(f"   本机自检 : http://127.0.0.1:{self.port}/health (供 frpc local_port={self.port} 穿透使用)", flush=True)
        print(f"   公网示例 : ws://home.52507000.xyz:38765/ws (frpc remote_port=38765)", flush=True)
        print("=" * 80, flush=True)
        self._httpd.serve_forever()


