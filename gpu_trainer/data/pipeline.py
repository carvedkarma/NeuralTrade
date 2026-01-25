import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import aiohttp
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
import pywt
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from tqdm import tqdm

class BinanceDataFetcher:
    BINANCE_VISION_URL = "https://data-api.binance.vision/api/v3"
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2"
    
    def __init__(self, symbols: List[str], timeframes: List[str], replit_proxy_url: Optional[str] = None):
        self.symbols = symbols
        self.timeframes = timeframes
        self.session = None
        self.working_source = None
        self.replit_proxy_url = replit_proxy_url
        
    async def _get_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _try_fetch(self, url: str, params: Dict, source_name: str) -> Optional[Any]:
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    print(f"[{source_name}] Rate limited (429)")
                elif resp.status == 403:
                    print(f"[{source_name}] Forbidden (403) - possibly blocked")
                elif resp.status == 451:
                    print(f"[{source_name}] Geoblocked (451)")
                else:
                    print(f"[{source_name}] HTTP {resp.status}")
        except aiohttp.ClientConnectorError as e:
            print(f"[{source_name}] Connection failed: {e}")
        except asyncio.TimeoutError:
            print(f"[{source_name}] Timeout")
        except Exception as e:
            print(f"[{source_name}] Error: {e}")
        return None
    
    async def _fetch_replit_proxy(self, symbol: str, timeframe: str, limit: int,
                                   end_time: Optional[int] = None) -> List[Dict]:
        if not self.replit_proxy_url:
            return []
        
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": min(limit, 1000)
        }
        if end_time:
            params["endTime"] = end_time
        
        url = f"{self.replit_proxy_url}/api/data/klines"
        data = await self._try_fetch(url, params, "Replit Proxy")
        
        if data and "candles" in data:
            candles = []
            for c in data["candles"]:
                candles.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": c["timestamp"],
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c["volume"]),
                    "close_time": c["closeTime"],
                    "quote_volume": float(c["quoteVolume"]),
                    "trades": c["trades"],
                    "taker_buy_base": float(c["takerBuyBase"]),
                    "taker_buy_quote": float(c["takerBuyQuote"])
                })
            if candles:
                print(f"[Replit Proxy] Successfully fetched {len(candles)} candles")
            return candles
        return []
    
    async def fetch_klines(self, symbol: str, timeframe: str, limit: int = 1000, 
                          start_time: Optional[int] = None, 
                          end_time: Optional[int] = None) -> List[Dict]:
        # If we already have a working source, try it first
        if self.working_source == "Replit Proxy":
            proxy_data = await self._fetch_replit_proxy(symbol, timeframe, limit, end_time)
            if proxy_data:
                return proxy_data
            self.working_source = None
        
        if self.working_source == "CryptoCompare":
            cc_data = await self._fetch_cryptocompare(symbol, timeframe, limit, end_time)
            if cc_data:
                return cc_data
            self.working_source = None
        
        # Try Replit Proxy first (bypasses Australia Binance block)
        if self.replit_proxy_url:
            proxy_data = await self._fetch_replit_proxy(symbol, timeframe, limit, end_time)
            if proxy_data:
                self.working_source = "Replit Proxy"
                return proxy_data
        
        # Then try direct Binance access
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        sources = [
            (f"{self.BINANCE_VISION_URL}/klines", params, "Binance Vision"),
            (f"{self.BINANCE_API_URL}/klines", params, "Binance API"),
        ]
        
        if self.working_source and self.working_source not in ["CryptoCompare", "Replit Proxy"]:
            sources = [s for s in sources if s[2] == self.working_source] + \
                      [s for s in sources if s[2] != self.working_source]
        
        for url, p, source_name in sources:
            data = await self._try_fetch(url, p, source_name)
            if data:
                self.working_source = source_name
                print(f"[{source_name}] Successfully fetched {len(data)} candles")
                return [self._parse_kline(k, symbol, timeframe) for k in data]
        
        # Last resort: CryptoCompare
        cc_data = await self._fetch_cryptocompare(symbol, timeframe, limit, end_time)
        if cc_data:
            self.working_source = "CryptoCompare"
            return cc_data
        
        return []
    
    async def _fetch_cryptocompare(self, symbol: str, timeframe: str, limit: int, 
                                    end_time: Optional[int] = None) -> List[Dict]:
        if end_time:
            return []
        
        fsym = symbol.replace("USDT", "")
        tsym = "USDT"
        
        tf_map = {"1m": "minute", "5m": "minute", "15m": "minute", "1h": "hour", "4h": "hour", "1d": "day"}
        endpoint = tf_map.get(timeframe, "minute")
        
        aggregate = 1
        if timeframe == "5m":
            aggregate = 5
        elif timeframe == "15m":
            aggregate = 15
        elif timeframe == "4h":
            aggregate = 4
        
        url = f"{self.CRYPTOCOMPARE_URL}/histo{endpoint}"
        params: Dict[str, Any] = {
            "fsym": fsym, 
            "tsym": tsym, 
            "limit": 2000,
            "aggregate": aggregate
        }
        
        data = await self._try_fetch(url, params, "CryptoCompare")
        if data and "Data" in data and "Data" in data["Data"]:
            candles = []
            for c in data["Data"]["Data"]:
                if c.get("close", 0) == 0 and c.get("open", 0) == 0:
                    continue
                ts_ms = c["time"] * 1000
                candles.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": ts_ms,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c.get("volumefrom", 0)),
                    "close_time": ts_ms,
                    "quote_volume": float(c.get("volumeto", 0)),
                    "trades": 0,
                    "taker_buy_base": 0,
                    "taker_buy_quote": 0
                })
            candles.sort(key=lambda x: x["timestamp"])
            if candles:
                print(f"[CryptoCompare] Fetched {len(candles)} most recent candles (max 2000, no pagination)")
            return candles
        return []
    
    def _parse_kline(self, kline: List, symbol: str, timeframe: str) -> Dict:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": kline[0],
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
            "close_time": kline[6],
            "quote_volume": float(kline[7]),
            "trades": int(kline[8]),
            "taker_buy_base": float(kline[9]),
            "taker_buy_quote": float(kline[10])
        }
    
    async def fetch_all_historical(self, lookback_candles: int = 50000) -> Dict[str, Dict[str, pd.DataFrame]]:
        all_data = {}
        
        for symbol in tqdm(self.symbols, desc="Fetching symbols"):
            all_data[symbol] = {}
            for timeframe in self.timeframes:
                candles = []
                oldest_ts = None
                
                while len(candles) < lookback_candles:
                    batch = await self.fetch_klines(
                        symbol, timeframe, limit=1000,
                        end_time=oldest_ts
                    )
                    if not batch:
                        if self.working_source == "CryptoCompare":
                            print(f"[CryptoCompare] Limited to {len(candles)} candles (no pagination)")
                        break
                    
                    batch.sort(key=lambda x: x["timestamp"])
                    
                    if oldest_ts is None:
                        candles = batch + candles
                    else:
                        new_candles = [c for c in batch if c["timestamp"] < oldest_ts]
                        if not new_candles:
                            break
                        candles = new_candles + candles
                    
                    oldest_ts = candles[0]["timestamp"] - 1
                    await asyncio.sleep(0.1)
                
                candles.sort(key=lambda x: x["timestamp"])
                candles = candles[-lookback_candles:] if len(candles) > lookback_candles else candles
                    
                if candles:
                    df = pd.DataFrame(candles)
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("datetime", inplace=True)
                    all_data[symbol][timeframe] = df
                    print(f"Fetched {len(candles)} total candles for {symbol} {timeframe}")
                else:
                    all_data[symbol][timeframe] = pd.DataFrame()
                
        return all_data
    
    async def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
        params = {"symbol": symbol, "limit": limit}
        
        sources = [
            (f"{self.BINANCE_VISION_URL}/depth", "Binance Vision"),
            (f"{self.BINANCE_API_URL}/depth", "Binance API"),
        ]
        
        for url, source_name in sources:
            data = await self._try_fetch(url, params, source_name)
            if data and "bids" in data and "asks" in data:
                bids = np.array([[float(p), float(q)] for p, q in data["bids"]])
                asks = np.array([[float(p), float(q)] for p, q in data["asks"]])
                
                bid_volume = bids[:, 1].sum() if len(bids) > 0 else 0
                ask_volume = asks[:, 1].sum() if len(asks) > 0 else 0
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
                
                return {
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "imbalance": imbalance,
                    "spread": (asks[0, 0] - bids[0, 0]) / bids[0, 0] if len(bids) > 0 and len(asks) > 0 else 0
                }
        return {}
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


class FeatureEngineer:
    def __init__(self, wavelet: str = "db4", wavelet_level: int = 4):
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.scalers = {}
        
    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        features["returns"] = df["close"].pct_change()
        features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        for period in [5, 10, 20, 50, 100]:
            features[f"sma_{period}"] = df["close"].rolling(period).mean()
            features[f"ema_{period}"] = df["close"].ewm(span=period).mean()
            features[f"std_{period}"] = df["close"].rolling(period).std()
            features[f"return_{period}"] = df["close"].pct_change(period)
            
        features["rsi_14"] = self._compute_rsi(df["close"], 14)
        features["rsi_7"] = self._compute_rsi(df["close"], 7)
        
        macd, signal, hist = self._compute_macd(df["close"])
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = hist
        
        bb_upper, bb_middle, bb_lower = self._compute_bollinger(df["close"])
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle
        features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        features["atr_14"] = self._compute_atr(df, 14)
        features["atr_7"] = self._compute_atr(df, 7)
        
        features["volume_sma_20"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / features["volume_sma_20"]
        
        features["adx_14"] = self._compute_adx(df, 14)
        
        stoch_k, stoch_d = self._compute_stochastic(df)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
        
        features["obv"] = self._compute_obv(df)
        features["obv_sma"] = features["obv"].rolling(20).mean()
        
        return features
    
    def compute_wavelet_features(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        coeffs = pywt.wavedec(prices, self.wavelet, level=self.wavelet_level)
        
        trend = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], self.wavelet)
        
        features = {"trend": trend[:len(prices)]}
        
        for i, c in enumerate(coeffs[1:], 1):
            detail_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(cc) for cc in coeffs[1:]]
            detail_coeffs[i] = c
            detail = pywt.waverec(detail_coeffs, self.wavelet)
            features[f"cycle_{i}"] = detail[:len(prices)]
            
        features["noise"] = prices - trend[:len(prices)]
        
        return features
    
    def compute_cross_asset_features(self, data: Dict[str, pd.DataFrame], 
                                     target_symbol: str = "BTCUSDT") -> pd.DataFrame:
        target_df = data[target_symbol]
        features = pd.DataFrame(index=target_df.index)
        
        for symbol, df in data.items():
            if symbol == target_symbol:
                continue
                
            aligned_df = df.reindex(target_df.index, method="ffill")
            
            symbol_short = symbol.replace("USDT", "")
            
            features[f"{symbol_short}_returns"] = aligned_df["close"].pct_change()
            
            features[f"{symbol_short}_corr_20"] = (
                target_df["close"].pct_change()
                .rolling(20)
                .corr(aligned_df["close"].pct_change())
            )
            
            features[f"{symbol_short}_lead_1"] = aligned_df["close"].pct_change().shift(1)
            features[f"{symbol_short}_lead_5"] = aligned_df["close"].pct_change(5).shift(1)
            
            btc_returns = target_df["close"].pct_change()
            alt_returns = aligned_df["close"].pct_change()
            features[f"{symbol_short}_relative_strength"] = btc_returns - alt_returns
            
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _compute_bollinger(self, prices: pd.Series, period: int = 20, 
                          std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._compute_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(period).mean()
        return adx
    
    def _compute_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                           d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()
        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-8)
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d
    
    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        return obv
    
    def fit_scalers(self, features: pd.DataFrame, method: str = "robust"):
        for col in features.columns:
            if method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            valid_data = features[col].dropna().values.reshape(-1, 1)
            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.scalers[col] = scaler
                
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = features.copy()
        for col in features.columns:
            if col in self.scalers:
                valid_mask = ~features[col].isna()
                if valid_mask.any():
                    transformed.loc[valid_mask, col] = self.scalers[col].transform(
                        features.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()
        return transformed
    
    def save_scalers(self, path: str):
        joblib.dump(self.scalers, path)
        
    def load_scalers(self, path: str):
        self.scalers = joblib.load(path)


class TradingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 sequence_length: int = 100):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length]
        return x, y


class MultiTimeframeDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], labels: np.ndarray,
                 sequence_lengths: Dict[str, int]):
        self.data = {tf: torch.FloatTensor(d) for tf, d in data.items()}
        self.labels = torch.FloatTensor(labels)
        self.sequence_lengths = sequence_lengths
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = {tf: self.data[tf][idx] for tf in self.data}
        y = self.labels[idx]
        return x, y


def create_labels(df: pd.DataFrame, horizon: int = 5, 
                  threshold: float = 0.001) -> np.ndarray:
    future_returns = df["close"].pct_change(horizon).shift(-horizon)
    
    labels = np.zeros(len(df))
    labels[future_returns > threshold] = 1
    labels[future_returns < -threshold] = -1
    
    return labels


def prepare_data_loaders(features: np.ndarray, labels: np.ndarray,
                        config, shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n = len(features)
    train_end = int(n * config.data.train_split)
    val_end = int(n * (config.data.train_split + config.data.val_split))
    
    train_dataset = TradingDataset(
        features[:train_end], 
        labels[:train_end],
        config.data.sequence_length
    )
    val_dataset = TradingDataset(
        features[train_end:val_end],
        labels[train_end:val_end],
        config.data.sequence_length
    )
    test_dataset = TradingDataset(
        features[val_end:],
        labels[val_end:],
        config.data.sequence_length
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
