#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.0 远程 GPU 算力模块：支持 WebSocket 二进制流与 HTTP Keep-Alive 双协议单端口复用。
"""

import os
import json
import time
import threading
from typing import Optional, Tuple
import cv2
import numpy as np
from .tracker import UniversalUNetTracker

# ==============================================================================
# 3.5 远程算力服务端与客户端接口 (支持 IPv4/IPv6 双栈 + 8765/38765 双端口 + WS/WSS/HTTP/HTTPS 自动协商与重连)
# ==============================================================================
import base64
import hashlib
import http.client
import socket
import ssl
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class DualStackThreadingHTTPServer(ThreadingHTTPServer):
    """支持 IPv4 (0.0.0.0) 与 IPv6 (::) 双栈同时监听的高并发 HTTP/WebSocket 服务器 (兼容家庭宽带 IPv6 DDNS 与 NAT 端口映射)"""
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except Exception:
                pass
        super().server_bind()


def create_dual_stack_server(host: str, port: int, handler_cls) -> ThreadingHTTPServer:
    """优先创建同时监听 IPv6 (::) 与 IPv4 (0.0.0.0) 的双栈服务端；若系统未启用 IPv6 则自动回退至 IPv4"""
    bind_host = (host or "0.0.0.0").strip()
    if bind_host in ("0.0.0.0", "::", "", "*"):
        try:
            class _IPv6Server(DualStackThreadingHTTPServer):
                address_family = socket.AF_INET6
            return _IPv6Server(("::", int(port)), handler_cls)
        except Exception:
            pass
        bind_host = "0.0.0.0"
    return DualStackThreadingHTTPServer((bind_host, int(port)), handler_cls)


class RemoteTrackerClient:
    """
    远程 GPU 算力客户端 (与 UniversalUNetTracker 接口 100% 兼容，可在 GUI 操作台直接无缝切换):
    - 专为公网 DDNS (如 ws://home.52507000.xyz:38765/ws)、IPv6 直连、FRP/Lucky/Nginx/Cloudflare 反向代理优化:
      1) 支持 IPv4 / IPv6 双栈自动优选寻址 (`getaddrinfo` 多地址快速故障转移)。
      2) 支持 `ws://`, `wss://`, `http://`, `https://` 四种协议及自签名证书免校验，若反向代理拦截 WebSocket Upgrade 则自动无缝降级为 HTTP Keep-Alive。
      3) 完整实现 RFC 6455 WebSocket 控制帧 (`Ping 0x09` 自动回复 `Pong 0x0A`、粘包缓冲区保留、断线自动重连重试)，防止反向代理心跳包导致 JSON 解析报错断开。
    """

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8765/ws",
        fps: float = 30.0,
        perf_mode: str = "ultra",
        jpeg_quality: int = 92,
        timeout: float = 6.0,
    ):
        self.server_url = (server_url or "ws://127.0.0.1:8765/ws").strip()
        self.fps = float(fps)
        self.perf_mode = perf_mode
        self.jpeg_quality = int(jpeg_quality)
        self.timeout = float(timeout)

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

    @staticmethod
    def _clean_url_string(url: str) -> str:
        """清理从微信/QQ/浏览器复制 URL 时可能夹带的零宽字符、全角冒号、首尾空格与引号"""
        if not url:
            return "ws://127.0.0.1:8765/ws"
        for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\r", "\n", "\t", "\"", "\'"):
            url = url.replace(ch, "")
        url = url.strip().replace("：", ":")
        return url

    def _parse_url(self, url: str) -> None:
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

    def _connect_raw_socket(self, timeout: Optional[float] = None, use_tls: Optional[bool] = None) -> socket.socket:
        """支持 IPv4 / IPv6 双栈多地址轮询连接与可选 TLS/SSL 握手"""
        tm = float(timeout if timeout is not None else self.timeout)
        tls_flag = (self.scheme in ("wss", "https")) if use_tls is None else bool(use_tls)
        try:
            addr_infos = socket.getaddrinfo(self.host, self.port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror as e:
            raise ConnectionError(f"域名解析失败 ({self.host}): {e}") from e

        last_err = None
        for family, socktype, proto, _, sockaddr in addr_infos:
            s = None
            try:
                s = socket.socket(family, socktype, proto)
                s.settimeout(tm)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except Exception:
                    pass
                s.connect(sockaddr)
                if tls_flag:
                    s = self._ssl_ctx.wrap_socket(s, server_hostname=self.host)
                return s
            except Exception as e:
                last_err = e
                if s is not None:
                    try:
                        s.close()
                    except Exception:
                        pass
        raise ConnectionError(f"无法连接到 {self.host}:{self.port} ({last_err})")

    def _open_http_connection(self, timeout: Optional[float] = None, use_https: Optional[bool] = None):
        tm = float(timeout if timeout is not None else self.timeout)
        is_https = (self.scheme in ("https", "wss")) if use_https is None else bool(use_https)
        if is_https:
            return http.client.HTTPSConnection(self.host, self.port, timeout=tm, context=self._ssl_ctx)
        return http.client.HTTPConnection(self.host, self.port, timeout=tm)

    def ping_health(self) -> Tuple[bool, str, float]:
        """
        全方位测试远程算力连通性 (适配 DDNS 端口映射、反向代理仅转发 /ws、IPv4/IPv6 双栈及 TLS 自动探测):
        1) 若配置为 ws:// 或 wss://，优先直接在目标路径 (如 /ws) 发起 WebSocket 握手并发送 ping 指令；
        2) 若 WebSocket 握手被代理拦截或配置为 http://，自动测试 GET /health 与 GET /ws；
        3) 若明文连接被重置，自动探测 TLS (wss/https) 端口。
        """
        t0 = time.perf_counter()
        with self._lock:
            self.close()
            # 1. 优先尝试 WebSocket 握手 + ping 指令 (100% 匹配 ws://home.52507000.xyz:38765/ws 等仅映射了 /ws 的反代环境)
            if self.scheme in ("ws", "wss"):
                for try_tls in ([self.scheme == "wss", not (self.scheme == "wss")]):
                    try:
                        t_ws = time.perf_counter()
                        self._ensure_ws_connected(timeout=4.5, use_tls=try_tls)
                        ping_cmd = json.dumps({"cmd": "ping"}).encode("utf-8")
                        self._ws_send_frame(self._ws_sock, ping_cmd, opcode=0x01)
                        resp_bytes = self._ws_recv_frame()
                        rtt_ms = (time.perf_counter() - t_ws) * 1000.0
                        info = json.loads(resp_bytes.decode("utf-8", errors="ignore"))
                        dev = info.get("device", "Remote GPU")
                        if try_tls and self.scheme == "ws":
                            self.scheme = "wss"
                        elif (not try_tls) and self.scheme == "wss":
                            self.scheme = "ws"
                        self._force_http_fallback = False
                        self.remote_device_name = f"远程 {dev}"
                        self.last_rtt_ms = rtt_ms
                        self.connected = True
                        self.last_error = ""
                        return True, f"在线(WS): {dev} ({rtt_ms:.1f}ms)", rtt_ms
                    except Exception as e_ws:
                        self.last_error = str(e_ws)
                        self.close()

            # 2. 尝试 HTTP/HTTPS GET /health 与当前路径 (兼容仅支持 HTTP 的穿透网关)
            for try_https in ([self.scheme in ("https", "wss"), not (self.scheme in ("https", "wss"))]):
                for check_path in ("/health", self.path, "/"):
                    try:
                        t_ht = time.perf_counter()
                        conn = self._open_http_connection(timeout=4.5, use_https=try_https)
                        conn.request(
                            "GET",
                            check_path,
                            headers={
                                "Host": f"{self.host}:{self.port}",
                                "User-Agent": "UNetTrackerClient/3.6",
                                "Accept": "application/json",
                            },
                        )
                        resp = conn.getresponse()
                        body = resp.read().decode("utf-8", errors="ignore")
                        conn.close()
                        rtt_ms = (time.perf_counter() - t_ht) * 1000.0
                        if resp.status == 200 and ("status" in body or "device" in body):
                            info = json.loads(body)
                            dev = info.get("device", "Remote GPU")
                            if self.scheme in ("ws", "wss"):
                                self._force_http_fallback = True
                            self.remote_device_name = f"远程 {dev}"
                            self.last_rtt_ms = rtt_ms
                            self.connected = True
                            self.last_error = ""
                            return True, f"在线(HTTP): {dev} ({rtt_ms:.1f}ms)", rtt_ms
                    except Exception as e_ht:
                        self.last_error = str(e_ht)

            err_msg = self.last_error or "目标端口未响应"
            if "10061" in err_msg or "Connection refused" in err_msg:
                hint = f"端口 {self.port} 拒绝连接(请确认服务端已启动且监听 {self.port}/8765)"
            elif "timed out" in err_msg or "Timeout" in err_msg:
                hint = f"连接 {self.host}:{self.port} 超时(请检查路由器端口转发/防火墙)"
            elif "gaierror" in err_msg or "域名解析" in err_msg:
                hint = f"域名 {self.host} 解析失败"
            else:
                hint = err_msg[:48]
            return False, f"失败: {hint}", 0.0

    def _ensure_ws_connected(self, timeout: Optional[float] = None, use_tls: Optional[bool] = None) -> None:
        if self._ws_sock is not None:
            return
        sock = self._connect_raw_socket(timeout=timeout, use_tls=use_tls)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        ws_path = self.path if self.path else "/ws"
        scheme_origin = "https" if (use_tls or self.scheme in ("wss", "https")) else "http"
        req = (
            f"GET {ws_path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"Origin: {scheme_origin}://{self.host}:{self.port}\r\n"
            f"User-Agent: UNetTrackerClient/3.6\r\n\r\n"
        )
        sock.sendall(req.encode("ascii"))
        resp = bytearray()
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(1024)
            if not chunk:
                sock.close()
                raise ConnectionError("WebSocket 握手被远端中断 (空响应)")
            resp.extend(chunk)
            if len(resp) > 16384:
                sock.close()
                raise ConnectionError("WebSocket 握手响应头过长")

        sep_idx = resp.index(b"\r\n\r\n") + 4
        header_bytes = bytes(resp[:sep_idx])
        extra_bytes = bytes(resp[sep_idx:])
        first_line = header_bytes.split(b"\r\n", 1)[0].decode("latin1", errors="ignore")

        if "101" not in first_line:
            sock.close()
            # 若服务端直接返回了 200 JSON (说明中间代理剥离了 Upgrade 头)，抛出特定标记以便自动切换 HTTP 模式
            if "200" in first_line:
                self._force_http_fallback = True
            raise ConnectionError(f"WebSocket 握手未返回 101 ({first_line})")

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
                raise ConnectionError("WebSocket 连接已由远端关闭")
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

            if opcode == 0x09:  # Ping -> 自动回复 Pong (0x0A) 并继续读取真实业务帧
                if self._ws_sock is not None:
                    try:
                        self._ws_send_frame(self._ws_sock, payload, opcode=0x0A)
                    except Exception:
                        pass
                continue
            if opcode == 0x0A:  # Pong -> 忽略并继续读取下一帧
                continue
            if opcode == 0x08:  # Close -> 抛出连接关闭以便外层重连
                raise ConnectionError("远端主动关闭了 WebSocket 会话 (Opcode 0x08)")
            return payload

    def reset(self) -> None:
        self.canonical_contour = None
        self.last_unet_prob_s = None
        with self._lock:
            for attempt in range(2):
                try:
                    if self.scheme in ("ws", "wss") and (not self._force_http_fallback):
                        self._ensure_ws_connected()
                        cmd = json.dumps({"cmd": "reset", "fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                        self._ws_send_frame(self._ws_sock, cmd, opcode=0x01)
                        _ = self._ws_recv_frame()
                    else:
                        conn = self._open_http_connection()
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
                except Exception:
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
            self._http_conn = self._open_http_connection()
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
            # 自动支持断线重连 1 次 + WebSocket 被网关拦截时自动降级 HTTP Keep-Alive
            for attempt in range(2):
                try:
                    if self.scheme in ("ws", "wss") and (not self._force_http_fallback):
                        self._ensure_ws_connected()
                        pkt = struct.pack("!ff", dt_val, float(self.fps)) + jpg_bytes
                        self._ws_send_frame(self._ws_sock, pkt, opcode=0x02)
                        resp_bytes = self._ws_recv_frame()
                        res = json.loads(resp_bytes.decode("utf-8"))
                    else:
                        res = self._send_track_http(jpg_bytes, dt_val)
                    break
                except Exception as e:
                    last_exc = e
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
    全图形自适应 U-Net 远程算力服务端 (支持 IPv4/IPv6 双栈 + 8765/38765 双端口同时监听 + HTTP/WebSocket 全路径兼容):
    - 可部署在 RTX 3060 (12GB) 主机、家庭宽带 DDNS 主机 (`home.52507000.xyz:38765`) 或 Google Colab T4 GPU 上:
      `python unet_universal_tracker_v2.py --server --host 0.0.0.0 --port 38765 --device cuda --mode ultra`
    - 当指定端口为 8765 (默认) 时，自动同时在后台监听 38765 端口 (反之亦然)，无论路由器转发至 8765 还是直通 38765 均可直接连通！
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
        self._aux_httpd: Optional[ThreadingHTTPServer] = None

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
                if upg == "websocket" or "upgrade" in conn_hdr and self.headers.get("Sec-WebSocket-Key"):
                    self._handle_websocket()
                    return
                # 兼容任意路径 (/health, /ws, /, /status) 的健康检查请求
                self._send_json({
                    "status": "ok",
                    "version": "V3.6",
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
                    self._send_json({"status": "reset_ok", "device": server_ref.tracker.cuda_backend.device_name})
                    return

                # 兼容 /track, /ws, / 等任意 POST 路径
                try:
                    ctype = self.headers.get("Content-Type", "").lower()
                    if "application/json" in ctype:
                        payload = json.loads(raw_body.decode("utf-8"))
                        if payload.get("cmd") == "reset":
                            with server_ref._lock:
                                server_ref.tracker.reset()
                            self._send_json({"status": "reset_ok", "device": server_ref.tracker.cuda_backend.device_name})
                            return
                        if payload.get("cmd") == "ping":
                            self._send_json({"status": "ok", "device": server_ref.tracker.cuda_backend.device_name})
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
                try:
                    while True:
                        b1_b2 = RemoteTrackerClient._ws_recv_exact(conn, 2)
                        opcode = b1_b2[0] & 0x0F
                        if opcode == 0x08:  # Close
                            break
                        masked = (b1_b2[1] & 0x80) != 0
                        length = b1_b2[1] & 0x7F
                        if length == 126:
                            length = struct.unpack("!H", RemoteTrackerClient._ws_recv_exact(conn, 2))[0]
                        elif length == 127:
                            length = struct.unpack("!Q", RemoteTrackerClient._ws_recv_exact(conn, 8))[0]
                        mask_key = RemoteTrackerClient._ws_recv_exact(conn, 4) if masked else None
                        payload = bytearray(RemoteTrackerClient._ws_recv_exact(conn, length))
                        if masked and mask_key:
                            for i in range(length):
                                payload[i] ^= mask_key[i & 3]

                        if opcode == 0x09:  # Ping -> 立即回复 Pong (0x0A)
                            self._ws_send_server_frame(conn, bytes(payload), opcode=0x0A)
                            continue
                        if opcode == 0x0A:  # Pong -> 忽略
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
                                    "version": "V3.6",
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
                                    "device": server_ref.tracker.cuda_backend.device_name,
                                }, ensure_ascii=False).encode("utf-8")
                            else:
                                jpg_bytes = base64.b64decode(msg["image_base64"])
                                st = server_ref.process_frame_bytes(
                                    jpg_bytes, dt_sec=msg.get("dt_sec"), fps=msg.get("fps")
                                )
                                resp_bytes = json.dumps(st, ensure_ascii=False).encode("utf-8")

                        self._ws_send_server_frame(conn, resp_bytes, opcode=0x01)
                except Exception:
                    pass

        return _Handler

    def start_forever(self) -> None:
        handler_cls = self._build_handler_class()
        self._httpd = create_dual_stack_server(self.host, self.port, handler_cls)

        # 自动双端口监听: 若主端口为 8765 则自动同时监听 38765 (若主端口为 38765 则同时监听 8765)
        aux_port = 38765 if self.port == 8765 else (8765 if self.port == 38765 else None)
        aux_port_str = ""
        if aux_port is not None:
            try:
                self._aux_httpd = create_dual_stack_server(self.host, aux_port, handler_cls)
                t_aux = threading.Thread(target=self._aux_httpd.serve_forever, daemon=True)
                t_aux.start()
                aux_port_str = f" & {aux_port} (双端口自动同开)"
            except Exception:
                self._aux_httpd = None

        stack_mode = "IPv4 + IPv6 双栈 (0.0.0.0 & [::])" if self._httpd.address_family == socket.AF_INET6 else f"IPv4 ({self.host})"
        print("=" * 78)
        print(f"🚀 U-Net 远程算力服务端 V3.6 (IPv4/IPv6 双栈 + 双端口兼容版) 已启动!")
        print(f"   加速硬件 : {self.tracker.cuda_backend.device_name}")
        print(f"   计算档位 : {self.tracker.perf_mode.upper()}")
        print(f"   网络监听 : {stack_mode} | 端口: {self.port}{aux_port_str}")
        print(f"   HTTP 地址: http://{self.host}:{self.port}/health")
        print(f"   WS   地址: ws://{self.host}:{self.port}/ws (及 38765 端口)")
        print("=" * 78)
        self._httpd.serve_forever()

