import os
import re
import time
import json
import uuid
import random
import socket
import tempfile
import threading
import subprocess
import concurrent.futures
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler

# Try to import rich for enhanced UI, fall back to basic if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.style import Style
    from rich.layout import Layout
    from rich.logging import RichHandler
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class VPNStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

class VPNManager:
    def __init__(self):
        self.current_process = None
        self.connection_thread = None
        self.auth_file = None
        self.configs = []
        self.username = '26awmx0l4Rz3WG91'
        self.password = 'x5n8HURoFPNOqtAetR1Q0w37jYUcbJ0j'
        self.connect_timeout = 10
        self.failed_servers = {}
        self.server_blacklist = set()
        self.connection_status = VPNStatus.DISCONNECTED
        self.connection_info = ""
        self.connection_start_time = None
        self.vpn_connection_duration = "00:00:00"
        self.log_history = []
        self.active_threads = {}
        self.auto_connect_enabled = False
        self.auto_connect_retry_delay = 10
        self.auto_connect_max_retries = 1
        self.auto_connect_shuffle = False
        self.auto_connect_failure_threshold = 1
        self.is_auto_connecting = False
        self.auto_connect_servers = []
        self.current_auto_connect_index = 0
        self.auto_connect_retry_count = 0
        self.current_auto_connect_server = None
        self.last_health_check = 0
        self.connection_timer = None
        self.auto_connect_timer = None
        self.lock = threading.Lock()
        self.logger = self.setup_logger()
        
        # Rich console if available
        if RICH_AVAILABLE:
            self.console = Console()
        
        self.load_config()
        self.load_vpn_configs()

    def setup_logger(self):
        """Setup advanced logging with file rotation and rich console output"""
        logger = logging.getLogger("QuantumVPN")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_dir / "quantum_vpn.log", 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler with rich if available
        if RICH_AVAILABLE:
            console_handler = RichHandler(
                show_time=False,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False
            )
            console_handler.setLevel(logging.INFO)
        else:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def stop_all(self):
        """Stop all VPN-related activities"""
        self.stop_auto_connect()
        self.disconnect_vpn(restart_auto_connect=False)
        
        # Stop any timers
        if self.connection_timer:
            self.connection_timer.cancel()
            
        if self.auto_connect_timer:
            self.auto_connect_timer.cancel()

    def load_config(self):
        default_config = {
            'vpn_configs_dir': 'vpn_configs',
            'username': '26awmx0l4Rz3WG91',
            'password': 'x5n8HURoFPNOqtAetR1Q0w37jYUcbJ0j',
            'connect_timeout': 10,
            'auto_connect_enabled': False,
            'auto_connect_retry_delay': 10,
            'auto_connect_max_retries': 1,
            'auto_connect_shuffle': False,
            'auto_connect_failure_threshold': 1,
        }
        
        try:
            if os.path.exists("vpn_config.json"):
                with open("vpn_config.json", 'r') as f:
                    config = json.load(f)
                    for key, default_value in default_config.items():
                        setattr(self, key, config.get(key, default_value))
            else:
                for key, value in default_config.items():
                    setattr(self, key, value)
                self.save_config()
        except Exception as e:
            for key, value in default_config.items():
                setattr(self, key, value)
            self.log_message(f"Error loading config: {e}", "error")

    def save_config(self):
        config = {
            'vpn_configs_dir': self.vpn_configs_dir,
            'username': self.username,
            'password': self.password,
            'connect_timeout': self.connect_timeout,
            'auto_connect_enabled': self.auto_connect_enabled,
            'auto_connect_retry_delay': self.auto_connect_retry_delay,
            'auto_connect_max_retries': self.auto_connect_max_retries,
            'auto_connect_shuffle': self.auto_connect_shuffle,
            'auto_connect_failure_threshold': self.auto_connect_failure_threshold,
        }
        with open("vpn_config.json", 'w') as f:
            json.dump(config, f)

    def load_vpn_configs(self):
        self.configs = []
        if os.path.exists(self.vpn_configs_dir) and os.path.isdir(self.vpn_configs_dir):
            for root, _, files in os.walk(self.vpn_configs_dir):
                for file in files:
                    if file.endswith(".ovpn"):
                        self.configs.append(os.path.join(root, file))
            self.log_message(f"Loaded {len(self.configs)} VPN configurations", "info")

    def create_auth_file(self):
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(f"{self.username}\n{self.password}\n")
                auth_file = f.name
            os.chmod(auth_file, 0o600)
            return auth_file
        except Exception as e:
            self.log_message(f"Error creating auth file: {e}", "error")
            return None

    def connect_to_vpn(self, config_path):
        if not self.username or not self.password:
            self.log_message("VPN credentials are not configured!", "error")
            return False
            
        if self.is_auto_connecting:
            self.stop_auto_connect()
            
        self.auth_file = self.create_auth_file()
        if not self.auth_file:
            return False
        
        self.log_message(f"Connecting to {os.path.basename(config_path)}...", "info")
        self.connection_status = VPNStatus.CONNECTING
        self.connection_info = f"Connecting to: {os.path.basename(config_path)}"
        
        try:
            self.connection_thread = VPNConnectionThread(
                config_path,
                self.auth_file,
                self.connect_timeout,
                self.handle_log,
                self.handle_status
            )
            
            thread_id = uuid.uuid4().hex
            self.active_threads[thread_id] = self.connection_thread
            self.connection_thread.start()
            return True

        except Exception as e:
            if self.auth_file and os.path.exists(self.auth_file):
                os.remove(self.auth_file)
                self.auth_file = None
            self.log_message(f"Connection failed: {str(e)}", "error")
            return False

    def disconnect_vpn(self, restart_auto_connect=True):
        if self.connection_thread and self.connection_thread.is_alive():
            self.log_message("Disconnecting...", "info")
            try:
                self.connection_thread.stop()
                self.connection_thread.join(timeout=5)
                self.connection_status = VPNStatus.DISCONNECTED
                self.connection_info = "No active connection"
                self.vpn_connection_duration = "00:00:00"
                self.log_message("Disconnected successfully", "info")
                
                if self.auth_file and os.path.exists(self.auth_file):
                    try:
                        os.remove(self.auth_file)
                        self.auth_file = None
                    except Exception as e:
                        self.log_message(f"Error removing auth file: {str(e)}", "error")
                
                if restart_auto_connect and self.auto_connect_enabled:
                    self.start_auto_connect()
                    
                return True
            except Exception as e:
                self.log_message(f"Error during disconnection: {str(e)}", "error")
                return False
                
        self.connection_status = VPNStatus.DISCONNECTED
        self.connection_info = "No active connection"
        return False

    def handle_log(self, message):
        self.log_message(message)
        
    def handle_status(self, event_type, message):
        with self.lock:
            if event_type == "connected":
                self.connection_status = VPNStatus.CONNECTED
                self.connection_info = message
                self.connection_start_time = datetime.now()
                self.last_health_check = time.time()
                self.start_connection_timer()
                self.log_message(f"Connected successfully: {message}", "success")
                
            elif event_type == "disconnected":
                self.connection_status = VPNStatus.DISCONNECTED
                self.connection_info = message
                self.stop_connection_timer()
                self.log_message(f"Disconnected: {message}", "info")
                
            elif event_type == "error":
                self.connection_status = VPNStatus.ERROR
                self.connection_info = message
                self.log_message(f"Connection error: {message}", "error")
                
                if self.current_auto_connect_server:
                    server = self.current_auto_connect_server
                    self.failed_servers.setdefault(server, {"count": 0, "last_error": message})
                    self.failed_servers[server]["count"] += 1
                    self.failed_servers[server]["last_error"] = message
                    
                    if "AUTH_FAILED" in message or "TLS Error" in message:
                        self.server_blacklist.add(server)
                        self.log_message(f"Blacklisted server due to critical error: {message}", "warning")
                
                if self.is_auto_connecting:
                    if self.auto_connect_timer:
                        self.auto_connect_timer.cancel()
                    
                    if "AUTH_FAILED" in message or "TLS Error" in message:
                        self.log_message("Critical error, skipping server", "error")
                        self.next_auto_connect_server()
                    else:
                        # Add the missing auto_connect_attempt reference
                        if self.auto_connect_retry_count < self.auto_connect_max_retries:
                            delay = self.auto_connect_retry_delay
                            self.log_message(f"Retrying in {delay}s (Attempt {self.auto_connect_retry_count + 1}/{self.auto_connect_max_retries})", "warning")
                            self.auto_connect_timer = threading.Timer(delay, self.auto_connect_attempt)
                            self.auto_connect_timer.start()
                        else:
                            self.log_message("Max retries reached, moving to next server", "warning")
                            self.next_auto_connect_server()

    def start_connection_timer(self):
        try:
            if self.connection_status == VPNStatus.CONNECTED and self.connection_start_time:
                self.update_connection_timer()
                if self.connection_timer:
                    self.connection_timer.cancel()
                self.connection_timer = threading.Timer(1.0, self.start_connection_timer)
                self.connection_timer.start()
        except Exception as e:
            self.log_message(f"Connection timer error: {str(e)}", "error")

    def stop_connection_timer(self):
        if hasattr(self, 'connection_timer') and self.connection_timer:
            try:
                self.connection_timer.cancel()
                self.connection_timer = None
            except Exception as e:
                self.log_message(f"Error stopping connection timer: {str(e)}", "error")

    def update_connection_timer(self):
        if self.connection_start_time and self.connection_status == VPNStatus.CONNECTED:
            duration = datetime.now() - self.connection_start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.vpn_connection_duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            current_time = time.time()
            if current_time - self.last_health_check > 30:
                self.check_connection_health()
                self.last_health_check = current_time
    
    def check_connection_health(self):
        if self.connection_status != VPNStatus.CONNECTED:
            return
            
        targets = ["8.8.8.8", "1.1.1.1", "208.67.222.222"]
        
        for target in targets:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3)
                    s.connect((target, 53))
                    return
            except Exception:
                continue
                
        self.log_message("All health checks failed", "error")
        self.handle_status("error", "Network unreachable")
    
    def validate_connection(self):
        try:
            result = subprocess.run(["ip", "route", "show", "default"], 
                                   capture_output=True, text=True)
            if "default" not in result.stdout:
                self.log_message("No default route found", "error")
                return False
                
            socket.gethostbyname("google.com")
            return True
        except Exception as e:
            self.log_message(f"Network validation failed: {str(e)}", "error")
            return False
    
    def log_message(self, message, msg_type="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        if len(self.log_history) >= 500:
            self.log_history = self.log_history[-250:]
        self.log_history.append({"timestamp": timestamp, "message": message, "type": msg_type})
        
        # Use logger for advanced logging
        if msg_type == "error":
            self.logger.error(message)
        elif msg_type == "warning":
            self.logger.warning(message)
        elif msg_type == "success":
            self.logger.info(message)
        else:
            self.logger.info(message)
    
    def get_vpn_status(self):
        return {
            "status": self.connection_status.value,
            "info": self.connection_info,
            "duration": self.vpn_connection_duration,
            "auto_connect": {
                "enabled": self.auto_connect_enabled,
                "active": self.is_auto_connecting,
                "current_server": self.current_auto_connect_server,
                "retry_count": self.auto_connect_retry_count
            }
        }

    def toggle_auto_connect(self, state=None):
        if state is None:
            self.auto_connect_enabled = not self.auto_connect_enabled
        else:
            self.auto_connect_enabled = bool(state)
        
        if self.auto_connect_enabled:
            self.start_auto_connect()
        else:
            self.stop_auto_connect()
        
        self.save_config()
        return self.auto_connect_enabled
    
    def start_auto_connect(self):
        if self.is_auto_connecting:
            self.log_message("Auto-connect already running", "warning")
            return False
        
        if not self.configs:
            self.log_message("No VPN configurations available for auto-connect", "error")
            return False
        
        self.is_auto_connecting = True
        self.log_message("Starting multi-threaded server selection...", "success")
        
        # Reset state when starting auto-connect
        self.auto_connect_retry_count = 0
        self.current_auto_connect_server = None
        
        # Start server selection
        selection_thread = threading.Thread(target=self.select_best_server)
        selection_thread.daemon = True
        selection_thread.start()
        return True

    def auto_connect_attempt(self):
        """Attempt to connect to the current auto-connect server"""
        if not self.is_auto_connecting or not self.current_auto_connect_server:
            return
            
        self.log_message(f"Attempting connection to {os.path.basename(self.current_auto_connect_server)}", "info")
        self.connect_to_best_server(self.current_auto_connect_server)

    def select_best_server(self):
        """Многопоточный выбор лучшего сервера на основе ping и скорости"""
        available_servers = [s for s in self.configs if s not in self.server_blacklist]
        
        if not available_servers:
            self.log_message("No available servers after filtering", "error")
            self.is_auto_connecting = False
            return

        # Этап 1: Быстрая проверка ping до всех серверов
        self.log_message("Testing server latency (ping)...", "info")
        ping_results = self.ping_servers(available_servers)
        
        # Выбираем топ-5 серверов с наилучшим ping
        top_servers = sorted(ping_results.items(), key=lambda x: x[1])[:5]
        top_servers = [server for server, ping in top_servers if ping < float('inf')]
        
        if not top_servers:
            self.log_message("No responsive servers found", "error")
            self.is_auto_connecting = False
            return

        # Этап 2: Проверка скорости соединения к лучшим серверам
        self.log_message("Testing connection speed to best servers...", "info")
        speed_results = self.test_connection_speed(top_servers)
        
        # Выбираем сервер с лучшей скоростью
        best_server = min(speed_results.items(), key=lambda x: x[1])[0] if speed_results else top_servers[0]
        
        self.log_message(f"Selected best server: {os.path.basename(best_server)}", "success")
        self.connect_to_best_server(best_server)

    def ping_servers(self, servers):
        """Многопоточная проверка ping до серверов"""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_server = {
                executor.submit(self.ping_server, server): server 
                for server in servers
            }
            
            for future in concurrent.futures.as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    ping_time = future.result()
                    results[server] = ping_time
                except Exception as e:
                    results[server] = float('inf')
                    self.log_message(f"Ping test failed for {os.path.basename(server)}: {e}", "warning")
        
        return results

    def ping_server(self, server_config):
        """Извлечение адреса сервера и проверка ping"""
        try:
            # Извлекаем адрес сервера из конфигурации
            with open(server_config, 'r') as f:
                content = f.read()
            
            # Ищем строку remote в конфигурации
            remote_match = re.search(r'remote\s+([^\s]+)\s+(\d+)', content)
            if not remote_match:
                return float('inf')
                
            server_address = remote_match.group(1)
            
            # Выполняем ping
            param = "-n 1" if os.name == 'nt' else "-c 1"
            command = ["ping", param, server_address]
            
            start_time = time.time()
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return (end_time - start_time) * 1000  # Время в мс
            else:
                return float('inf')
                
        except (subprocess.TimeoutExpired, Exception) as e:
            return float('inf')

    def test_connection_speed(self, servers):
        """Проверка скорости соединения к серверам"""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_server = {
                executor.submit(self.test_single_connection, server): server 
                for server in servers
            }
            
            for future in concurrent.futures.as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    speed = future.result()
                    results[server] = speed
                except Exception as e:
                    results[server] = float('inf')
                    self.log_message(f"Speed test failed for {os.path.basename(server)}: {e}", "warning")
        
        return results

    def test_single_connection(self, server_config):
        """Тест скорости одиночного соединения"""
        # Создаем временный auth файл
        auth_file = self.create_auth_file()
        if not auth_file:
            return float('inf')
        
        try:
            # Запускаем OpenVPN с таймаутом
            cmd = [
                "sudo", "openvpn",
                "--config", server_config,
                "--auth-user-pass", auth_file,
                "--verb", "0",
                "--connect-timeout", "10"
            ]
            
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            
            # Ждем установления соединения или таймаута
            try:
                stdout, _ = process.communicate(timeout=15)
                end_time = time.time()
                
                if "Initialization Sequence Completed" in stdout.decode():
                    return end_time - start_time  # Время установления соединения
                else:
                    return float('inf')
            except subprocess.TimeoutExpired:
                process.kill()
                return float('inf')
                
        except Exception as e:
            return float('inf')
        finally:
            if auth_file and os.path.exists(auth_file):
                os.remove(auth_file)

    def connect_to_best_server(self, server_config):
        """Подключение к выбранному лучшему серверу"""
        self.current_auto_connect_server = server_config
        self.auto_connect_retry_count = 0
        
        self.log_message(f"Connecting to best server: {os.path.basename(server_config)}", "info")
        
        self.auth_file = self.create_auth_file()
        if not self.auth_file:
            self.next_auto_connect_server()
            return
        
        self.connection_thread = VPNConnectionThread(
            server_config, 
            self.auth_file,
            self.connect_timeout,
            self.handle_log,
            self.handle_status
        )
        
        self.connection_thread.start()

    def stop_auto_connect(self):
        if self.is_auto_connecting:
            self.is_auto_connecting = False
            if self.auto_connect_timer:
                self.auto_connect_timer.cancel()
            self.log_message("Auto-connect stopped", "info")
            return True
        return False
 
    def next_auto_connect_server(self):
        """Handle connection failure by restarting server selection"""
        if self.is_auto_connecting:
            self.log_message("Server connection failed, reselecting...", "warning")
            # Reset retry count when moving to a new server
            self.auto_connect_retry_count = 0
            # Add small delay before reselecting to avoid rapid cycles
            time.sleep(2)
            self.select_best_server()

    def handle_server_selection_error(self, error_message):
        """Handle errors during server selection"""
        self.log_message(f"Server selection error: {error_message}", "error")
        if self.is_auto_connecting:
            self.log_message("Will retry server selection in 10 seconds", "warning")
            time.sleep(10)
            self.select_best_server()


class VPNConnectionThread(threading.Thread):
    def __init__(self, ovpn_file, auth_file, connect_timeout, log_callback, status_callback):
        super().__init__()
        self.ovpn_file = ovpn_file
        self.auth_file = auth_file
        self.connect_timeout = connect_timeout
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.process = None
        self.connected = False
        self.stop_flag = threading.Event()
        self.output_lines = []

    def run(self):
        try:
            cmd = [
                "sudo", "openvpn",
                "--config", self.ovpn_file,
                "--auth-user-pass", self.auth_file,
                "--verb", "3"
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            connected = threading.Event()
            error_detected = threading.Event()
            error_message = ""
            
            def monitor_output():
                nonlocal error_message
                while not self.stop_flag.is_set():
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    
                    if "auth-user-pass" in line or "password" in line.lower():
                        line = "[REDACTED CREDENTIALS]"
                    
                    self.output_lines.append(line)
                    self.log_callback(line)
                    
                    if "Initialization Sequence Completed" in line:
                        connected.set()
                        self.connected = True
                        self.status_callback("connected", f"Connected to {os.path.basename(self.ovpn_file)}")
                        self.log_callback("Connection established successfully")
                    
                    elif "AUTH_FAILED" in line:
                        error_detected.set()
                        error_message = "Authentication failed: Check your credentials"
                    elif "TLS Error" in line:
                        error_detected.set()
                        error_message = "TLS handshake failed: Protocol error"
                    elif "timed out" in line:
                        error_detected.set()
                        error_message = "Connection timed out: Server unreachable"
                    elif "Could not resolve host" in line:
                        error_detected.set()
                        error_message = "DNS resolution failed: Invalid hostname"
                    elif "Network unreachable" in line:
                        error_detected.set()
                        error_message = "Network unreachable: Check connectivity"
                    elif "Linux route add command failed because route exists" in line:
                        self.log_callback("Route already exists - continuing connection")
            
            monitor_thread = threading.Thread(target=monitor_output)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            start_time = time.time()
            while not self.stop_flag.is_set():
                if connected.is_set():
                    break
                
                if error_detected.is_set():
                    self.status_callback("error", error_message)
                    self.stop()
                    break
                    
                if time.time() - start_time > self.connect_timeout:
                    self.status_callback("error", "Connection timed out")
                    self.stop()
                    break
                    
                if self.process.poll() is not None:
                    if not connected.is_set() and not error_detected.is_set():
                        last_lines = "\n".join(self.output_lines[-3:]) if self.output_lines else ""
                        if "error" in last_lines.lower() or "failed" in last_lines.lower():
                            error_message = self.parse_openvpn_error(last_lines)
                        else:
                            error_message = "Connection failed unexpectedly"
                        self.status_callback("error", error_message)
                    break
                    
                time.sleep(0.5)
            
            if not self.stop_flag.is_set() and self.process.poll() is None:
                self.process.wait()
                
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            if "No such file or directory" in str(e):
                error_msg = "VPN configuration file not found"
            self.status_callback("error", error_msg)
        finally:
            if self.auth_file and os.path.exists(self.auth_file):
                try:
                    os.remove(self.auth_file)
                except Exception:
                    pass
            self.status_callback("disconnected", "")

    def parse_openvpn_error(self, output):
        if "certificate verify failed" in output:
            return "Certificate verification failed"
        elif "no route to host" in output:
            return "No network route to host"

        if "AUTH_FAILED" in output:
            return "Authentication failed: Check your username and password"
        elif "TLS Error" in output:
            return "TLS handshake failed: Security protocol error"
        elif "timed out" in output:
            return "Connection timed out: Server unreachable"
        elif "Could not resolve host" in output:
            return "DNS resolution failed: Invalid hostname"
        elif "Network unreachable" in output:
            return "Network unreachable: Check your internet connection"
        elif "Permission denied" in output:
            return "Permission denied: Run with administrator privileges"
        elif "No such file or directory" in output:
            return "Configuration file not found"
        elif "SIGTERM" in output or "process exiting" in output:
            return "Connection terminated by user"
        
        error_lines = [line for line in output.splitlines() 
                    if "error" in line.lower() or "fail" in line.lower()]
        
        if error_lines:
            return error_lines[-1]
        
        return "Connection failed: Unknown error"

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.stop_flag.set()


class VPNCLI:
    def __init__(self, vpn_manager):
        self.vpn_manager = vpn_manager
        self.running = True
        self.last_update = time.time()
        
        if RICH_AVAILABLE:
            self.console = Console()
            self.layout = Layout()
            self.setup_layout()
        else:
            self.console = None

    def setup_layout(self):
        # Split the console into three parts
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=2),
            Layout(name="logs", size=10),
        )
        
        # Further split the main section
        self.layout["main"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="servers", ratio=1),
        )

    def display_status_panel(self):
        if not RICH_AVAILABLE:
            return "[Rich not available]"
            
        status = self.vpn_manager.get_vpn_status()
        
        # Create status table
        status_table = Table(show_header=False, box=box.ROUNDED)
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="white")
        
        # Add rows based on connection status
        status_table.add_row("Status", status["status"].upper())
        status_table.add_row("Duration", status["duration"])
        status_table.add_row("Info", status["info"])
        
        # Auto-connect status
        auto_connect = status["auto_connect"]
        ac_status = "Enabled" if auto_connect["enabled"] else "Disabled"
        if auto_connect["active"]:
            ac_status += " (Active)"
        status_table.add_row("Auto-Connect", ac_status)
        
        if auto_connect["current_server"]:
            server_name = os.path.basename(auto_connect["current_server"])
            status_table.add_row("Current Server", server_name)
            status_table.add_row("Retry Count", str(auto_connect["retry_count"]))
        
        return Panel(status_table, title="VPN Status", border_style="blue")

    def display_servers_panel(self):
        if not RICH_AVAILABLE:
            return "[Rich not available]"
            
        servers_table = Table(box=box.ROUNDED)
        servers_table.add_column("Server", style="cyan")
        servers_table.add_column("Status", style="green")
        
        # Add some sample data - in a real implementation, you'd track server status
        if self.vpn_manager.configs:
            for config in self.vpn_manager.configs[:5]:  # Show first 5 servers
                server_name = os.path.basename(config)
                status = "Available"
                
                if config in self.vpn_manager.server_blacklist:
                    status = "[red]Blacklisted[/red]"
                elif config == self.vpn_manager.current_auto_connect_server:
                    status = "[green]Connecting[/green]"
                
                servers_table.add_row(server_name, status)
        
        return Panel(servers_table, title="Servers", border_style="green")

    def display_logs_panel(self):
        if not RICH_AVAILABLE:
            return "[Rich not available]"
            
        # Get last 10 log entries
        recent_logs = self.vpn_manager.log_history[-10:] if self.vpn_manager.log_history else []
        
        log_text = Text()
        for log in recent_logs:
            timestamp = log["timestamp"]
            message = log["message"]
            log_type = log["type"]
            
            # Apply styles based on log type
            if log_type == "error":
                style = "red"
            elif log_type == "warning":
                style = "yellow"
            elif log_type == "success":
                style = "green"
            else:
                style = "white"
            
            # Truncate long messages to prevent UI issues
            if len(message) > 80:
                message = message[:77] + "..."
                
            # Append timestamp and message with appropriate style
            log_text.append(f"{timestamp} ", style="dim")
            log_text.append(f"{message}\n", style=style)
        
        return Panel(log_text, title="Recent Logs", border_style="yellow")

    def display_header(self):
        if not RICH_AVAILABLE:
            return "[Rich not available]"
            
        title_text = Text("QUANTUM VPN", style="bold blue")
        subtitle_text = Text("Secure Connection Manager", style="italic dim")
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_text = Text(now, style="bold green")
        
        header_table = Table(show_header=False, box=box.ROUNDED)
        header_table.add_column("Title", ratio=2)
        header_table.add_column("Time", ratio=1)
        header_table.add_row(title_text, time_text)
        header_table.add_row(subtitle_text, "")
        
        return Panel(header_table, style="blue")

    def update_display(self):
        if not RICH_AVAILABLE:
            # Basic text interface if rich is not available
            if time.time() - self.last_update > 1.0:  # Update every second
                os.system('cls' if os.name == 'nt' else 'clear')
                status = self.vpn_manager.get_vpn_status()
                print("=" * 60)
                print("QUANTUM VPN - Secure Connection Manager")
                print("=" * 60)
                print(f"Status: {status['status'].upper()}")
                print(f"Duration: {status['duration']}")
                print(f"Info: {status['info']}")
                print(f"Auto-Connect: {'Enabled' if status['auto_connect']['enabled'] else 'Disabled'}")
                if status['auto_connect']['current_server']:
                    print(f"Current Server: {os.path.basename(status['auto_connect']['current_server'])}")
                print("-" * 60)
                
                # Show last 5 logs with basic coloring
                print("Recent logs:")
                for log in self.vpn_manager.log_history[-5:]:
                    color_code = ""
                    if log['type'] == "error" or "error" in log['message'].lower():
                        color_code = "\033[91m"  # red
                    elif log['type'] == "warning":
                        color_code = "\033[93m"  # yellow
                    elif log['type'] == "success":
                        color_code = "\033[92m"  # green
                    
                    reset_code = "\033[0m"
                    print(f"{color_code}{log['timestamp']} {log['message']}{reset_code}")
                
                print("=" * 60)
                print("Commands: (c)onnect, (d)isconnect, (a)uto-connect, (q)uit")
                self.last_update = time.time()
            return
        
        # Update Rich layout with actual content
        self.layout["header"].update(self.display_header())
        self.layout["status"].update(self.display_status_panel())
        self.layout["servers"].update(self.display_servers_panel())
        self.layout["logs"].update(self.display_logs_panel())

    def handle_input(self):
        if self.console and RICH_AVAILABLE:
            # Add input handling for Rich interface
            try:
                # Check for input with timeout
                result = self.console.input("", timeout=0.1)
                if result:
                    self.process_command(result.lower())
            except Exception:
                pass
        else:
            # Basic input handling for non-rich interface
            try:
                # Check if there's input available (non-blocking)
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    cmd = sys.stdin.readline().strip().lower()
                    self.process_command(cmd)
            except Exception:
                pass

    def process_command(self, command):
        if command == 'q':
            self.running = False
            self.vpn_manager.stop_all()
            print("Shutting down...")
        elif command == 'c':
            if self.vpn_manager.configs:
                server = random.choice(self.vpn_manager.configs)
                self.vpn_manager.connect_to_vpn(server)
            else:
                print("No VPN configurations available")
        elif command == 'd':
            self.vpn_manager.disconnect_vpn()
        elif command == 'a':
            enabled = self.vpn_manager.toggle_auto_connect()
            status = "enabled" if enabled else "disabled"
            print(f"Auto-connect {status}")
        else:
            print("Unknown command")

    def run(self):
        try:
            if RICH_AVAILABLE:
                with Live(self.layout, refresh_per_second=4, screen=True) as live:
                    while self.running:
                        self.update_display()
                        self.handle_input()
                        time.sleep(0.25)
            else:
                while self.running:
                    self.update_display()
                    self.handle_input()
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.vpn_manager.stop_all()
            print("\nShutting down...")


if __name__ == "__main__":
    vpn_manager = VPNManager()
    
    # Start auto-connect if enabled
    if vpn_manager.auto_connect_enabled:
        vpn_manager.start_auto_connect()

    # Start CLI interface
    cli = VPNCLI(vpn_manager)
    cli.run()