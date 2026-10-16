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
# 3.5 远程算力服务端与客户端接口 (支持 HTTP Keep-Alive 与 WebSocket 双协议单端口复用)
# ==============================================================================
import base64
import hashlib
import http.client
import socket
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class RemoteTrackerClient:
    """
    远程 GPU 算力客户端 (与 UniversalUNetTracker 接口 100% 兼容，可在 GUI 操作台直接无缝切换):
    - 支持两种低延迟传输协议:
      1) WebSocket 二进制流协议 (`ws://IP:8765/ws` 或 `wss://...`):
         建立长连接后，每帧直接发送 JPEG 二进制字节流，无 Base64 编码开销与 HTTP 头开销，局域网通信延迟 < 1.0 ms。
      2) HTTP Keep-Alive 长连接协议 (`http://IP:8765` 或 `https://...`):
         复用底层 TCP 套接字发送 `/track` 二进制帧，完美兼容 Cloudflare Tunnel / ngrok / FRP 等内网穿透网关。
    """

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8765/ws",
        fps: float = 30.0,
        perf_mode: str = "ultra",
        jpeg_quality: int = 92,
        timeout: float = 5.0,
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

        self.ema_alpha = 0.50
        self.gate_radius_half = 44.0
        self.max_step_px = 21.0

        self._lock = threading.Lock()
        self._ws_sock: Optional[socket.socket] = None
        self._http_conn: Optional[http.client.HTTPConnection] = None
        self._parse_url(self.server_url)

    def _parse_url(self, url: str) -> None:
        if "://" not in url:
            url = "ws://" + url
        parsed = urlparse(url)
        self.scheme = parsed.scheme.lower()
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme in ("https", "wss") else 8765)
        self.path = parsed.path if parsed.path else ("/ws" if self.scheme in ("ws", "wss") else "/track")

    def set_server_url(self, url: str) -> None:
        with self._lock:
            self.close()
            self.server_url = url.strip()
            self._parse_url(self.server_url)

    def close(self) -> None:
        self.connected = False
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

    def ping_health(self) -> Tuple[bool, str, float]:
        """测试远程算力服务器连通性、获取远端 GPU 型号并测量网络 RTT (ms)"""
        t0 = time.perf_counter()
        try:
            conn_cls = http.client.HTTPSConnection if self.scheme in ("https", "wss") else http.client.HTTPConnection
            conn = conn_cls(self.host, self.port, timeout=2.5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="ignore")
            conn.close()
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status == 200:
                info = json.loads(body)
                dev = info.get("device", "Remote GPU")
                self.remote_device_name = f"远程 {dev}"
                self.last_rtt_ms = rtt_ms
                self.connected = True
                return True, f"在线: {dev} (RTT {rtt_ms:.1f}ms)", rtt_ms
            return False, f"HTTP {resp.status}", rtt_ms
        except Exception as e:
            self.connected = False
            return False, f"连接失败: {e}", 0.0

    def _ensure_ws_connected(self) -> None:
        if self._ws_sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        ws_path = self.path if self.path else "/ws"
        req = (
            f"GET {ws_path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(req.encode("ascii"))
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(1024)
            if not chunk:
                sock.close()
                raise ConnectionError("WebSocket 握手失败")
            resp += chunk
        if b"101" not in resp.split(b"\r\n")[0]:
            sock.close()
            raise ConnectionError(f"WebSocket 握手被拒绝: {resp[:80]!r}")
        self._ws_sock = sock
        self.connected = True

    @staticmethod
    def _ws_send_frame(sock: socket.socket, payload: bytes, opcode: int = 0x02) -> None:
        """客户端向服务端发送带掩码的 RFC 6455 WebSocket 数据帧 (0x01=Text, 0x02=Binary)"""
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

    @staticmethod
    def _ws_recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("WebSocket 连接已断开")
            buf.extend(chunk)
        return bytes(buf)

    def _ws_recv_frame(self, sock: socket.socket) -> bytes:
        b1, b2 = self._ws_recv_exact(sock, 2)
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._ws_recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._ws_recv_exact(sock, 8))[0]
        mask_key = self._ws_recv_exact(sock, 4) if masked else None
        data = bytearray(self._ws_recv_exact(sock, length))
        if masked and mask_key:
            for i in range(length):
                data[i] ^= mask_key[i & 3]
        return bytes(data)

    def reset(self) -> None:
        self.canonical_contour = None
        self.last_unet_prob_s = None
        with self._lock:
            try:
                if self.scheme in ("ws", "wss"):
                    self._ensure_ws_connected()
                    cmd = json.dumps({"cmd": "reset", "fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                    self._ws_send_frame(self._ws_sock, cmd, opcode=0x01)
                    _ = self._ws_recv_frame(self._ws_sock)
                else:
                    conn_cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
                    conn = conn_cls(self.host, self.port, timeout=self.timeout)
                    body = json.dumps({"fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                    conn.request("POST", "/reset", body=body, headers={"Content-Type": "application/json"})
                    _ = conn.getresponse().read()
                    conn.close()
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

    def update(self, frame_bgr: np.ndarray, dt_sec: Optional[float] = None) -> dict:
        t0 = time.perf_counter()
        ok, jpg_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ValueError("图像 JPEG 编码失败")
        jpg_bytes = jpg_buf.tobytes()
        dt_val = float(dt_sec) if (dt_sec is not None and dt_sec > 0) else (1.0 / max(1.0, self.fps))

        with self._lock:
            try:
                if self.scheme in ("ws", "wss"):
                    self._ensure_ws_connected()
                    # 二进制帧头: 8字节 float64 (dt_sec, fps) + JPEG 字节流
                    pkt = struct.pack("!ff", dt_val, float(self.fps)) + jpg_bytes
                    self._ws_send_frame(self._ws_sock, pkt, opcode=0x02)
                    resp_bytes = self._ws_recv_frame(self._ws_sock)
                    res = json.loads(resp_bytes.decode("utf-8"))
                else:
                    if self._http_conn is None:
                        conn_cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
                        self._http_conn = conn_cls(self.host, self.port, timeout=self.timeout)
                    headers = {
                        "Content-Type": "image/jpeg",
                        "Connection": "keep-alive",
                        "X-Dt-Sec": f"{dt_val:.6f}",
                        "X-Fps": f"{self.fps:.2f}",
                        "X-Mode": self.perf_mode,
                    }
                    self._http_conn.request("POST", "/track", body=jpg_bytes, headers=headers)
                    resp = self._http_conn.getresponse()
                    resp_body = resp.read()
                    if resp.status != 200:
                        raise ConnectionError(f"HTTP {resp.status}: {resp_body[:100]!r}")
                    res = json.loads(resp_body.decode("utf-8"))
            except Exception as e:
                self.close()
                H, W = frame_bgr.shape[:2]
                return {
                    "frame_idx": 0, "phase": "REMOTE_ERR", "shape": f"ERR:{str(e)[:16]}",
                    "cx": W * 0.5, "cy": H * 0.5, "w": self.ref_w, "h": self.ref_h, "angle": 0.0,
                    "vx": 0.0, "vy": 0.0, "omega": 0.0, "snr": 0.0, "conf": 0.0,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0, "n_step": 1.0,
                    "device": f"远程断开 ({self.server_url})"
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
    全图形自适应 U-Net 远程算力服务端 (单端口同时支持 HTTP REST 与 WebSocket 实时二进制流):
    - 可部署在 RTX 3060 (12GB) 主机、局域网算力服务器或 Google Colab T4 GPU 上:
      `python unet_universal_tracker_v2.py --server --host 0.0.0.0 --port 8765 --device cuda --mode ultra`
    - HTTP 接口:
      - `GET  /health` : 返回服务器状态与 CUDA 显卡型号 JSON
      - `POST /reset`  : 重置追踪器状态
      - `POST /track`  : 接收原始 `image/jpeg` 二进制或 JSON `{"image_base64": ...}`，返回追踪状态 JSON
    - WebSocket 接口:
      - `ws://IP:8765/ws`: 支持全双工二进制帧 (`8字节 header + JPEG bytes`) 或 JSON 文本帧，零握手开销极速推理
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

    def start_forever(self) -> None:
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

            def do_GET(self):
                if self.headers.get("Upgrade", "").lower() == "websocket":
                    self._handle_websocket()
                    return
                if self.path.startswith("/health") or self.path == "/":
                    self._send_json({
                        "status": "ok",
                        "version": "V3.0",
                        "device": server_ref.tracker.cuda_backend.device_name,
                        "mode": server_ref.tracker.perf_mode,
                        "frame_idx": server_ref.tracker.frame_idx,
                    })
                else:
                    self._send_json({"error": "Not Found"}, status=404)

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

                if self.path.startswith("/track"):
                    try:
                        ctype = self.headers.get("Content-Type", "").lower()
                        if "application/json" in ctype:
                            payload = json.loads(raw_body.decode("utf-8"))
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
                    return

                self._send_json({"error": "Unknown endpoint"}, status=404)

            def _handle_websocket(self) -> None:
                key = self.headers.get("Sec-WebSocket-Key", "")
                accept = base64.b64encode(
                    hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
                ).decode("ascii")
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()

                conn = self.connection
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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

                        if opcode == 0x02:  # Binary frame: 8-byte float32(dt_sec, fps) + JPEG bytes
                            dt_val, fps_val = struct.unpack("!ff", payload[:8])
                            jpg_bytes = bytes(payload[8:])
                            st = server_ref.process_frame_bytes(jpg_bytes, dt_sec=dt_val, fps=fps_val)
                            resp_bytes = json.dumps(st, ensure_ascii=False).encode("utf-8")
                        else:  # Text JSON frame
                            msg = json.loads(payload.decode("utf-8"))
                            if msg.get("cmd") == "reset":
                                with server_ref._lock:
                                    if "fps" in msg:
                                        server_ref.tracker.fps = float(msg["fps"])
                                    if "mode" in msg:
                                        server_ref.tracker.perf_mode = str(msg["mode"])
                                    server_ref.tracker.reset()
                                resp_bytes = json.dumps({"status": "reset_ok"}).encode("utf-8")
                            else:
                                jpg_bytes = base64.b64decode(msg["image_base64"])
                                st = server_ref.process_frame_bytes(
                                    jpg_bytes, dt_sec=msg.get("dt_sec"), fps=msg.get("fps")
                                )
                                resp_bytes = json.dumps(st, ensure_ascii=False).encode("utf-8")

                        # Send unmasked WebSocket text frame back to client
                        hdr = bytearray([0x81])
                        rlen = len(resp_bytes)
                        if rlen < 126:
                            hdr.append(rlen)
                        elif rlen < 65536:
                            hdr.append(126)
                            hdr.extend(struct.pack("!H", rlen))
                        else:
                            hdr.append(127)
                            hdr.extend(struct.pack("!Q", rlen))
                        conn.sendall(hdr + resp_bytes)
                except Exception:
                    pass

        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        print("=" * 78)
        print(f"🚀 U-Net 远程算力服务端 V3.0 已启动!")
        print(f"   加速硬件 : {self.tracker.cuda_backend.device_name}")
        print(f"   计算档位 : {self.tracker.perf_mode.upper()}")
        print(f"   HTTP 地址: http://{self.host}:{self.port}  (/health, /track, /reset)")
        print(f"   WS   地址: ws://{self.host}:{self.port}/ws (低延迟二进制流长连接)")
        print("=" * 78)
        self._httpd.serve_forever()


