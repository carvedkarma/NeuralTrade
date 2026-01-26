#!/usr/bin/env python3
"""
BTC Futures Trading - GPU Trainer Desktop GUI

A graphical interface for fetching data and training neural networks
on your local GPU without using the command line.

Usage:
    python gui.py
    
    Or double-click gui.py on Windows
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import queue
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

class LogRedirector:
    def __init__(self, widget, queue):
        self.widget = widget
        self.queue = queue
        
    def write(self, text):
        self.queue.put(text)
        
    def flush(self):
        pass

class GPUTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BTC Futures - GPU Neural Network Trainer")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        self.configure_dark_theme()
        
        self.log_queue = queue.Queue()
        self.is_training = False
        self.is_fetching = False
        self.training_thread = None
        self.fetch_thread = None
        self.stop_training_flag = threading.Event()
        
        self.current_model = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_loss = None
        self.val_loss = None
        self.models_completed = []
        self.gpu_name = None
        self.gpu_memory_used = None
        self.gpu_memory_total = None
        
        self.create_widgets()
        self.wire_stdout_to_log()
        self.check_gpu_status()
        self.process_log_queue()
        self.start_status_push()
    
    def wire_stdout_to_log(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = LogRedirector(self.log_text, self.log_queue)
        sys.stderr = LogRedirector(self.log_text, self.log_queue)
        
    def configure_dark_theme(self):
        self.colors = {
            'bg': '#1a1a2e',
            'bg_secondary': '#16213e',
            'bg_tertiary': '#0f3460',
            'accent': '#e94560',
            'accent_hover': '#ff6b6b',
            'text': '#eaeaea',
            'text_secondary': '#a0a0a0',
            'success': '#00d26a',
            'warning': '#ffd93d',
            'border': '#2a2a4a'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Secondary.TFrame', background=self.colors['bg_secondary'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), foreground=self.colors['accent'])
        style.configure('Status.TLabel', font=('Segoe UI', 9), foreground=self.colors['text_secondary'])
        
        style.configure('TButton', 
                       background=self.colors['accent'],
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(15, 8))
        style.map('TButton',
                 background=[('active', self.colors['accent_hover']), ('disabled', self.colors['border'])])
        
        style.configure('Success.TButton', background=self.colors['success'])
        style.map('Success.TButton', background=[('active', '#00b359')])
        
        style.configure('TEntry',
                       fieldbackground=self.colors['bg_secondary'],
                       foreground=self.colors['text'],
                       insertcolor=self.colors['text'])
        
        style.configure('TCombobox',
                       fieldbackground=self.colors['bg_secondary'],
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text'])
        
        style.configure('Horizontal.TProgressbar',
                       background=self.colors['accent'],
                       troughcolor=self.colors['bg_secondary'])
        
    def create_widgets(self):
        main_container = ttk.Frame(self.root, padding=15)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="BTC Futures GPU Trainer", style='Header.TLabel')
        title_label.pack(side=tk.LEFT)
        
        self.gpu_status_label = ttk.Label(header_frame, text="Checking GPU...", style='Status.TLabel')
        self.gpu_status_label.pack(side=tk.RIGHT)
        
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(content_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.create_data_panel(left_panel)
        self.create_training_panel(left_panel)
        
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_log_panel(right_panel)
        
    def create_data_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=" Data Fetching ", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        proxy_frame = ttk.Frame(frame)
        proxy_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(proxy_frame, text="Replit Proxy URL:").pack(anchor=tk.W)
        self.proxy_url_var = tk.StringVar(value=os.getenv("REPLIT_PROXY_URL", "https://99f68291-4a03-450a-9815-ebee9435cee2-00-2os5ge21n6uho.spock.replit.dev"))
        proxy_entry = ttk.Entry(proxy_frame, textvariable=self.proxy_url_var, width=40)
        proxy_entry.pack(fill=tk.X, pady=(2, 0))
        
        candles_frame = ttk.Frame(frame)
        candles_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(candles_frame, text="Candles to Fetch:").pack(anchor=tk.W)
        self.candles_var = tk.StringVar(value="50000")
        candles_combo = ttk.Combobox(candles_frame, textvariable=self.candles_var, 
                                      values=["1000", "10000", "50000", "100000", "200000"], width=15)
        candles_combo.pack(anchor=tk.W, pady=(2, 0))
        
        symbols_frame = ttk.Frame(frame)
        symbols_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(symbols_frame, text="Symbols:").pack(anchor=tk.W)
        self.btc_var = tk.BooleanVar(value=True)
        self.eth_var = tk.BooleanVar(value=True)
        self.sol_var = tk.BooleanVar(value=True)
        self.bnb_var = tk.BooleanVar(value=True)
        
        cb_frame = ttk.Frame(symbols_frame)
        cb_frame.pack(anchor=tk.W, pady=(2, 0))
        
        ttk.Checkbutton(cb_frame, text="BTCUSDT", variable=self.btc_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(cb_frame, text="ETHUSDT", variable=self.eth_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(cb_frame, text="SOLUSDT", variable=self.sol_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(cb_frame, text="BNBUSDT", variable=self.bnb_var).pack(side=tk.LEFT)
        
        self.fetch_progress = ttk.Progressbar(frame, mode='indeterminate')
        self.fetch_progress.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        self.fetch_btn = ttk.Button(btn_frame, text="Fetch Data", command=self.start_fetch)
        self.fetch_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.test_proxy_btn = ttk.Button(btn_frame, text="Test Proxy", command=self.test_proxy)
        self.test_proxy_btn.pack(side=tk.LEFT)
        
    def create_training_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=" Neural Network Training ", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        model_frame = ttk.Frame(frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model Architecture:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="transformer")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                    values=["transformer", "tft", "lstm", "cnn", "vae", "gnn"], width=20)
        model_combo.pack(anchor=tk.W, pady=(2, 0))
        
        params_frame = ttk.Frame(frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row1, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="100")
        epochs_entry = ttk.Entry(row1, textvariable=self.epochs_var, width=8)
        epochs_entry.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_var = tk.StringVar(value="64")
        batch_entry = ttk.Entry(row1, textvariable=self.batch_var, width=8)
        batch_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X)
        
        ttk.Label(row2, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value="0.0001")
        lr_entry = ttk.Entry(row2, textvariable=self.lr_var, width=12)
        lr_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        self.train_progress = ttk.Progressbar(frame, mode='determinate')
        self.train_progress.pack(fill=tk.X, pady=(0, 10))
        
        self.train_status_label = ttk.Label(frame, text="Ready to train", style='Status.TLabel')
        self.train_status_label.pack(anchor=tk.W, pady=(0, 10))
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        self.train_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training, style='Success.TButton')
        self.train_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_train_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_train_btn.pack(side=tk.LEFT)
        
        quick_frame = ttk.LabelFrame(parent, text=" Quick Actions ", padding=10)
        quick_frame.pack(fill=tk.X)
        
        ttk.Button(quick_frame, text="Train All Models", command=self.train_all_models).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(quick_frame, text="Start API Server", command=self.start_api_server).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(quick_frame, text="Train RL Agent", command=self.train_rl_agent).pack(fill=tk.X)
        
    def create_log_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=" Output Log ", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            font=('Consolas', 9),
            height=20
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=(5, 0))
        
        self.log("=" * 50)
        self.log("  BTC Futures GPU Trainer - Ready")
        self.log("=" * 50)
        self.log("")
        
    def check_gpu_status(self):
        def check():
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    mem_used = torch.cuda.memory_allocated(0) / 1024**3
                    status = f"[OK] GPU: {name} ({mem_total:.1f} GB)"
                    self.log(f"GPU Detected: {name} with {mem_total:.1f} GB VRAM")
                    self.gpu_name = name
                    self.gpu_memory_total = mem_total
                    self.gpu_memory_used = mem_used
                else:
                    status = "[WARN] No GPU - Using CPU"
                    self.log("WARNING: No GPU detected. Training will be slow on CPU.")
                    self.gpu_name = None
            except ImportError:
                status = "[ERROR] PyTorch not installed"
                self.log("ERROR: PyTorch not installed. Run: pip install torch")
            except Exception as e:
                status = f"[ERROR] {str(e)[:30]}"
                self.log(f"GPU check error: {e}")
                
            self.root.after(0, lambda: self.gpu_status_label.config(text=status))
            
        threading.Thread(target=check, daemon=True).start()
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}\n")
        
    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.process_log_queue)
        
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        
    def save_log(self):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log(f"Log saved to: {filename}")
    
    def start_status_push(self):
        def push_loop():
            while True:
                self.push_status_to_replit()
                import time
                time.sleep(5)
        
        push_thread = threading.Thread(target=push_loop, daemon=True)
        push_thread.start()
        
    def push_status_to_replit(self):
        try:
            import requests
            proxy_url = self.proxy_url_var.get().strip()
            if not proxy_url:
                return
            
            status = {
                "gpuAvailable": self.gpu_name is not None,
                "gpuName": self.gpu_name,
                "gpuMemoryUsed": self.gpu_memory_used,
                "gpuMemoryTotal": self.gpu_memory_total,
                "isTraining": self.is_training,
                "trainingProgress": (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0,
                "currentModel": self.current_model,
                "currentEpoch": self.current_epoch,
                "totalEpochs": self.total_epochs,
                "trainLoss": self.train_loss,
                "valLoss": self.val_loss,
                "modelsLoaded": [],
                "modelsCompleted": self.models_completed
            }
            
            url = f"{proxy_url}/api/gpu/push-status"
            response = requests.post(url, json=status, timeout=5)
            
        except Exception:
            pass
            
    def test_proxy(self):
        def do_test():
            self.log("Testing proxy connection...")
            proxy_url = self.proxy_url_var.get().strip()
            
            if not proxy_url:
                self.log("ERROR: Please enter a Replit proxy URL")
                return
                
            try:
                import requests
                test_url = f"{proxy_url}/api/data/klines?symbol=BTCUSDT&interval=15m&limit=3"
                self.log(f"Requesting: {test_url}")
                
                response = requests.get(test_url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    count = data.get('count', 0)
                    self.log(f"[OK] SUCCESS! Received {count} candles from proxy")
                    self.log(f"   Latest BTC price: ${data['candles'][-1]['close']:,.2f}")
                else:
                    self.log(f"[ERROR] FAILED: HTTP {response.status_code}")
                    self.log(f"   Response: {response.text[:200]}")
                    
            except requests.exceptions.ConnectionError as e:
                self.log(f"[ERROR] Connection failed: Cannot reach proxy server")
                self.log("   Possible causes:")
                self.log("   - DNS resolution issue (try using requests instead of aiohttp)")
                self.log("   - Proxy server not running")
                self.log("   - Network/firewall blocking")
            except Exception as e:
                self.log(f"[ERROR] {e}")
                
        threading.Thread(target=do_test, daemon=True).start()
        
    def get_selected_symbols(self):
        symbols = []
        if self.btc_var.get():
            symbols.append("BTCUSDT")
        if self.eth_var.get():
            symbols.append("ETHUSDT")
        if self.sol_var.get():
            symbols.append("SOLUSDT")
        if self.bnb_var.get():
            symbols.append("BNBUSDT")
        return symbols
        
    def start_fetch(self):
        if self.is_fetching:
            return
            
        symbols = self.get_selected_symbols()
        if not symbols:
            messagebox.showwarning("No Symbols", "Please select at least one symbol to fetch")
            return
            
        proxy_url = self.proxy_url_var.get().strip()
        if not proxy_url:
            messagebox.showwarning("No Proxy URL", "Please enter your Replit proxy URL")
            return
            
        self.is_fetching = True
        self.fetch_btn.config(state=tk.DISABLED)
        self.fetch_progress.config(mode='determinate', value=0)
        
        def do_fetch():
            try:
                candles = int(self.candles_var.get())
                self.log(f"Starting data fetch: {candles} candles for {', '.join(symbols)}")
                self.log(f"Using proxy: {proxy_url}")
                
                os.environ["REPLIT_PROXY_URL"] = proxy_url
                
                from config import Config
                config = Config()
                config.data.symbols = symbols
                config.replit_proxy_url = proxy_url
                
                from data.pipeline import BinanceDataFetcher
                
                total_pairs = len(symbols) * len(config.data.timeframes)
                pairs_done = [0]
                
                def progress_callback(current, total, symbol, timeframe):
                    pairs_done[0] = current
                    pct = (current / total) * 100
                    self.root.after(0, lambda: self.fetch_progress.config(value=pct))
                
                fetcher = BinanceDataFetcher(
                    symbols, 
                    config.data.timeframes,
                    replit_proxy_url=proxy_url,
                    use_sync=True
                )
                
                data = fetcher.fetch_all_historical_sync(candles, progress_callback=progress_callback)
                
                total_candles = 0
                for symbol, timeframes in data.items():
                    for tf, df in timeframes.items():
                        if len(df) > 0:
                            path = config.data_dir / f"{symbol}_{tf}.parquet"
                            df.to_parquet(path)
                            total_candles += len(df)
                            self.log(f"Saved {len(df)} candles: {symbol} {tf}")
                        else:
                            self.log(f"WARNING: No data for {symbol} {tf}")
                            
                if total_candles > 0:
                    self.log(f"")
                    self.log(f"[OK] Fetch complete! Total: {total_candles:,} candles saved")
                else:
                    self.log(f"[ERROR] No data was fetched. Check proxy connection.")
                    
            except Exception as e:
                self.log(f"[ERROR] Fetch error: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.root.after(0, self.fetch_complete)
                
        self.fetch_thread = threading.Thread(target=do_fetch, daemon=True)
        self.fetch_thread.start()
        
    def fetch_complete(self):
        self.is_fetching = False
        self.fetch_btn.config(state=tk.NORMAL)
        self.fetch_progress.stop()
        
    def start_training(self):
        if self.is_training:
            return
        
        # Define the dataset being trained
        training_symbol = "BTCUSDT"
        training_timeframe = "15m"
        data_filename = f"{training_symbol}_{training_timeframe}.parquet"
            
        data_path = Path(__file__).parent / "data_cache" / data_filename
        if not data_path.exists():
            result = messagebox.askyesno(
                "No Data", 
                f"No training data found for {training_symbol} {training_timeframe}.\nWould you like to fetch data first?"
            )
            if result:
                self.start_fetch()
            return
            
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.NORMAL)
        self.train_progress['value'] = 0
        
        def do_train():
            try:
                model_type = self.model_var.get()
                epochs = int(self.epochs_var.get())
                batch_size = int(self.batch_var.get())
                lr = float(self.lr_var.get())
                
                # PROMINENT DATASET LOGGING - Critical for data isolation awareness
                self.log(f"")
                self.log(f"{'='*70}")
                self.log(f"   TRAINING DATASET INFORMATION")
                self.log(f"{'='*70}")
                self.log(f"   Symbol:    {training_symbol}")
                self.log(f"   Timeframe: {training_timeframe}")
                self.log(f"   Data File: {data_filename}")
                self.log(f"{'='*70}")
                self.log(f"")
                self.log(f"Starting training: {model_type.upper()} model")
                self.log(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
                self.log(f"")
                
                import torch
                import numpy as np
                import pandas as pd
                from config import config
                from data.pipeline import FeatureEngineer, TradingDataset, create_labels
                from torch.utils.data import DataLoader
                from training.trainer import Trainer
                
                df = pd.read_parquet(data_path)
                self.log(f"Loaded {len(df):,} candles from cache")
                
                # Validate loaded data for symbol/timeframe isolation - BLOCKING on critical errors
                self.log(f"")
                self.log(f"[Data Validation] Verifying data integrity...")
                validation_failed = False
                
                if "symbol" in df.columns:
                    unique_symbols = df["symbol"].unique().tolist()
                    if len(unique_symbols) == 1 and unique_symbols[0] == training_symbol:
                        self.log(f"[Data Validation] Symbol check PASSED: {training_symbol}")
                    elif len(unique_symbols) > 1:
                        self.log(f"[Data Validation] CRITICAL: Multiple symbols detected: {unique_symbols}")
                        self.log(f"[Data Validation] ABORTING TRAINING - Data contamination detected!")
                        validation_failed = True
                    else:
                        self.log(f"[Data Validation] WARNING: Unexpected symbol in data: {unique_symbols}")
                
                if "timeframe" in df.columns:
                    unique_tfs = df["timeframe"].unique().tolist()
                    if len(unique_tfs) == 1 and unique_tfs[0] == training_timeframe:
                        self.log(f"[Data Validation] Timeframe check PASSED: {training_timeframe}")
                    elif len(unique_tfs) > 1:
                        self.log(f"[Data Validation] CRITICAL: Multiple timeframes detected: {unique_tfs}")
                        self.log(f"[Data Validation] ABORTING TRAINING - Data contamination detected!")
                        validation_failed = True
                    else:
                        self.log(f"[Data Validation] WARNING: Unexpected timeframe in data: {unique_tfs}")
                
                if "timestamp" in df.columns:
                    n_unique = df["timestamp"].nunique()
                    if n_unique == len(df):
                        self.log(f"[Data Validation] No duplicate timestamps: {n_unique:,} unique")
                    else:
                        dup_count = len(df) - n_unique
                        self.log(f"[Data Validation] WARNING: {dup_count} duplicate timestamps found")
                
                if validation_failed:
                    self.log(f"[Data Validation] FAILED - Training aborted for data safety")
                    self.log(f"")
                    self.root.after(0, self.training_complete)
                    return
                
                self.log(f"[Data Validation] PASSED - Data integrity verified")
                self.log(f"")
                
                engineer = FeatureEngineer()
                features_df = engineer.compute_technical_features(df)
                features_df = features_df.fillna(0)
                
                engineer.fit_scalers(features_df)
                scaled_features = engineer.transform(features_df)
                
                labels = create_labels(df, horizon=5, threshold=0.001)
                labels = (labels + 1).astype(int)
                
                features_np = scaled_features.values.astype(np.float32)
                labels_np = labels.astype(np.int64)
                
                valid_start = config.data.sequence_length
                features_np = features_np[valid_start:]
                labels_np = labels_np[valid_start:]
                
                n_train = int(len(features_np) * 0.8)
                n_val = int(len(features_np) * 0.1)
                
                train_dataset = TradingDataset(features_np[:n_train], labels_np[:n_train], config.data.sequence_length)
                val_dataset = TradingDataset(features_np[n_train:n_train+n_val], labels_np[n_train:n_train+n_val], config.data.sequence_length)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                input_dim = features_np.shape[1]
                self.log(f"Features: {input_dim}, Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
                
                if model_type == "transformer":
                    from models.transformer import TransformerPriceModel
                    model = TransformerPriceModel(input_dim=input_dim, d_model=256, nhead=8, num_layers=6)
                elif model_type == "tft":
                    from models.transformer import TemporalFusionTransformer
                    model = TemporalFusionTransformer(input_dim=input_dim, d_model=256, nhead=8)
                elif model_type == "lstm":
                    from models.lstm import BidirectionalLSTM
                    model = BidirectionalLSTM(input_dim=input_dim, hidden_dim=256, num_layers=3)
                elif model_type == "cnn":
                    from models.cnn import ResNetPrice
                    model = ResNetPrice(input_dim=input_dim, channels=[64, 128, 256, 512])
                elif model_type == "vae":
                    from models.vae import MarketVAE
                    model = MarketVAE(input_dim=input_dim, sequence_length=config.data.sequence_length, latent_dim=64)
                elif model_type == "gnn":
                    from models.gnn import CrossAssetGNN
                    model = CrossAssetGNN(input_dim=input_dim, num_assets=4)
                else:
                    self.log(f"Unknown model: {model_type}")
                    return
                    
                self.log(f"Model parameters: {model.count_parameters():,}")
                
                config.training.epochs = epochs
                config.training.learning_rate = lr
                
                trainer = Trainer(model, train_loader, val_loader, config, device=config.device)
                
                self.current_model = model_type
                self.total_epochs = epochs
                
                def progress_callback(epoch, train_loss, val_loss):
                    progress = (epoch + 1) / epochs * 100
                    self.current_epoch = epoch + 1
                    self.train_loss = train_loss
                    self.val_loss = val_loss
                    
                    self.root.after(0, lambda: self.train_progress.config(value=progress))
                    self.root.after(0, lambda: self.train_status_label.config(
                        text=f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                    ))
                    self.log(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                    
                    if not self.is_training:
                        return False
                    return True
                    
                trainer.epoch_callback = progress_callback
                
                history = trainer.train(epochs=epochs)
                
                if self.is_training:
                    self.models_completed.append(model_type)
                    save_path = config.model_dir / f"{model_type}_trained.pt"
                    model.save(str(save_path))
                    self.log(f"")
                    self.log(f"[OK] Training complete! Model saved to: {save_path}")
                    
                    engineer.save_scalers(str(config.model_dir / f"{model_type}_scalers.joblib"))
                else:
                    self.log("Training stopped by user")
                    
            except Exception as e:
                self.log(f"[ERROR] Training error: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.root.after(0, self.training_complete)
                
        self.training_thread = threading.Thread(target=do_train, daemon=True)
        self.training_thread.start()
        
    def training_complete(self):
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_train_btn.config(state=tk.DISABLED)
        self.train_status_label.config(text="Ready to train")
        
    def stop_training(self):
        if self.is_training:
            self.log("Stopping training... (will complete current epoch)")
            self.is_training = False
            
    def train_all_models(self):
        if self.is_training:
            messagebox.showinfo("Busy", "Training already in progress")
            return
            
        result = messagebox.askyesno(
            "Train All Models",
            "This will train all 6 model architectures sequentially.\n\nThis may take several hours. Continue?"
        )
        if not result:
            return
            
        def train_sequence():
            models = ["transformer", "tft", "lstm", "cnn", "vae", "gnn"]
            for i, model in enumerate(models):
                if not self.is_training:
                    self.log(f"Training sequence stopped at {model}")
                    break
                self.log(f"")
                self.log(f"=== Training model {i+1}/6: {model.upper()} ===")
                self.model_var.set(model)
                self.root.after(0, self.start_training)
                
                while self.is_training:
                    import time
                    time.sleep(1)
                    
            self.log("")
            self.log("=== All models training complete! ===")
            
        self.is_training = True
        threading.Thread(target=train_sequence, daemon=True).start()
        
    def train_rl_agent(self):
        if self.is_training:
            messagebox.showinfo("Busy", "Training already in progress")
            return
            
        self.log("Starting RL Agent training...")
        
        def do_rl_train():
            try:
                self.is_training = True
                self.root.after(0, lambda: self.train_btn.config(state=tk.DISABLED))
                
                import torch
                from config import config
                from models.rl_agent import PPOAgent, TradingEnvironment
                import numpy as np
                
                dummy_data = np.random.randn(10000, 5)
                env = TradingEnvironment(
                    data=dummy_data,
                    initial_balance=config.rl.initial_capital,
                    transaction_cost=config.rl.transaction_cost
                )
                
                agent = PPOAgent(
                    state_dim=env._get_state().shape[0],
                    action_dim=3,
                    hidden_dim=256,
                    gamma=config.rl.gamma,
                    gae_lambda=config.rl.gae_lambda,
                    clip_epsilon=config.rl.clip_epsilon,
                    device=config.device
                )
                
                episodes = 500
                self.log(f"Training for {episodes} episodes...")
                
                for episode in range(episodes):
                    if not self.is_training:
                        break
                        
                    state = env.reset()
                    done = False
                    total_reward = 0
                    
                    while not done:
                        action, log_prob, value = agent.select_action(state)
                        next_state, reward, done, info = env.step(action)
                        
                        from models.rl_agent import Experience
                        exp = Experience(state, action, reward, next_state, done, log_prob, value)
                        agent.store_experience(exp)
                        
                        state = next_state
                        total_reward += reward
                        
                    if len(agent.buffer) >= 256:
                        agent.update()
                        
                    if (episode + 1) % 50 == 0:
                        progress = (episode + 1) / episodes * 100
                        self.root.after(0, lambda p=progress: self.train_progress.config(value=p))
                        self.log(f"Episode {episode+1}: Reward={total_reward:.2f}, Trades={info['num_trades']}")
                        
                if self.is_training:
                    save_path = config.model_dir / "ppo_agent.pt"
                    agent.save(str(save_path))
                    self.log(f"[OK] RL Agent saved to: {save_path}")
                    
            except Exception as e:
                self.log(f"[ERROR] RL Training error: {e}")
            finally:
                self.root.after(0, self.training_complete)
                
        threading.Thread(target=do_rl_train, daemon=True).start()
        
    def start_api_server(self):
        self.log("Starting FastAPI prediction server on port 8000...")
        
        def do_serve():
            try:
                from api.server import start_server
                start_server(host="0.0.0.0", port=8000)
            except Exception as e:
                self.log(f"[ERROR] Server error: {e}")
                
        threading.Thread(target=do_serve, daemon=True).start()
        self.log("Server thread started. API will be available at http://localhost:8000")

def main():
    root = tk.Tk()
    
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
        
    app = GPUTrainerGUI(root)
    
    def on_closing():
        sys.stdout = app.original_stdout
        sys.stderr = app.original_stderr
        
        if app.is_training or app.is_fetching:
            if messagebox.askokcancel("Quit", "Training/Fetching in progress. Are you sure you want to quit?"):
                app.is_training = False
                app.is_fetching = False
                root.destroy()
        else:
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
