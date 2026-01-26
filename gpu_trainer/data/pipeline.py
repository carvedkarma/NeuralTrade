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
    
    def __init__(self, symbols: List[str], timeframes: List[str], replit_proxy_url: Optional[str] = None, use_sync: bool = False):
        self.symbols = symbols
        self.timeframes = timeframes
        self.session = None
        self.working_source = None
        self.replit_proxy_url = replit_proxy_url
        self.use_sync = use_sync
    
    def _fetch_replit_proxy_sync(self, symbol: str, timeframe: str, limit: int,
                                  end_time: Optional[int] = None) -> List[Dict]:
        if not self.replit_proxy_url:
            return []
        
        try:
            import requests
            
            params: Dict[str, Any] = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": min(limit, 1000)
            }
            if end_time:
                params["endTime"] = end_time
            
            url = f"{self.replit_proxy_url}/api/data/klines"
            print(f"[Replit Proxy] Fetching {symbol} {timeframe} (limit={params['limit']})...")
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if "candles" in data:
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
                        print(f"[Replit Proxy] Successfully fetched {len(candles)} candles for {symbol} {timeframe}")
                    return candles
            else:
                print(f"[Replit Proxy] HTTP {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"[Replit Proxy] Sync fetch error: {e}")
        return []
    
    def fetch_klines_sync(self, symbol: str, timeframe: str, limit: int = 1000,
                          end_time: Optional[int] = None) -> List[Dict]:
        if self.replit_proxy_url:
            return self._fetch_replit_proxy_sync(symbol, timeframe, limit, end_time)
        
        print(f"[Sync] No Replit proxy configured - cannot fetch {symbol}")
        return []
    
    def fetch_historical_sync(self, symbol: str, timeframe: str, 
                               num_candles: int = 175000) -> pd.DataFrame:
        print(f"[Sync] Fetching {num_candles:,} candles for {symbol} {timeframe}...")
        
        all_candles = []
        end_time = None
        remaining = num_candles
        batch_size = 1000
        fetched = 0
        
        while remaining > 0:
            fetch_count = min(batch_size, remaining)
            candles = self.fetch_klines_sync(symbol, timeframe, fetch_count, end_time)
            
            if not candles:
                print(f"[Sync] No more data available for {symbol} {timeframe}")
                break
            
            all_candles = candles + all_candles
            remaining -= len(candles)
            fetched += len(candles)
            
            pct = (fetched / num_candles) * 100
            print(f"[{symbol} {timeframe}] Progress: {fetched:,}/{num_candles:,} ({pct:.1f}%)")
            
            if len(candles) < batch_size:
                break
                
            end_time = candles[0]["timestamp"] - 1
            
            import time
            time.sleep(0.1)
        
        if not all_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"])
        
        print(f"[Sync] Completed {symbol} {timeframe}: {len(df):,} unique candles")
        return df
    
    def fetch_all_historical_sync(self, num_candles: int = 175000, 
                                    progress_callback=None) -> Dict[str, Dict[str, pd.DataFrame]]:
        results = {}
        
        total_pairs = len(self.symbols) * len(self.timeframes)
        current = 0
        
        print(f"[Sync] Fetching {num_candles:,} candles for {len(self.symbols)} symbols, {len(self.timeframes)} timeframes")
        print(f"[Sync] Total pairs to fetch: {total_pairs}")
        
        for symbol in self.symbols:
            results[symbol] = {}
            for timeframe in self.timeframes:
                current += 1
                print(f"")
                print(f"=== [{current}/{total_pairs}] Fetching {symbol} {timeframe} ===")
                
                if progress_callback:
                    progress_callback(current, total_pairs, symbol, timeframe)
                
                df = self.fetch_historical_sync(symbol, timeframe, num_candles)
                results[symbol][timeframe] = df
        
        if progress_callback:
            progress_callback(total_pairs, total_pairs, "", "")
        
        print(f"")
        print(f"[Sync] All fetches complete!")
        return results
        
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
        # If Replit Proxy is configured, use ONLY that source
        # This avoids DNS/connection errors from trying blocked Binance APIs
        if self.replit_proxy_url:
            proxy_data = await self._fetch_replit_proxy(symbol, timeframe, limit, end_time)
            if proxy_data:
                self.working_source = "Replit Proxy"
                return proxy_data
            # If proxy fails, don't fall back to Binance (it's likely blocked)
            print(f"[Replit Proxy] Failed to fetch {symbol} {timeframe} - no fallback when proxy is configured")
            return []
        
        # No proxy configured - try direct Binance access (for non-geoblocked regions)
        if self.working_source == "CryptoCompare":
            cc_data = await self._fetch_cryptocompare(symbol, timeframe, limit, end_time)
            if cc_data:
                return cc_data
            self.working_source = None
        
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


class DashboardAPIFetcher:
    """
    Fetches multi-timeframe aligned data from the Replit dashboard's GPU Export API.
    This is the recommended way to get training data as it provides properly aligned
    candles with as-of joins across timeframes.
    """
    
    def __init__(self, dashboard_url: str):
        self.dashboard_url = dashboard_url.rstrip('/')
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_timeframes(self) -> Dict[str, List[str]]:
        """Get available timeframes and symbols from dashboard."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/timeframes") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("timeframes", [])
            return []
    
    async def get_data_range(self, symbol: str = "BTCUSDT", timeframe: str = "1m") -> Dict:
        """Get data range for a symbol/timeframe."""
        session = await self._get_session()
        params = {"symbol": symbol, "timeframe": timeframe}
        async with session.get(f"{self.dashboard_url}/api/gpu-export/data-range", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {}
    
    async def get_trainer_config(self) -> Dict:
        """Get optimal GPU trainer configuration."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/trainer-config") as resp:
            if resp.status == 200:
                return await resp.json()
            return {}
    
    async def get_feature_specs(self) -> List[Dict]:
        """Get feature engineering specifications."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/feature-specs") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("features", [])
            return []
    
    async def get_walk_forward_folds(self, symbol: str = "BTCUSDT", timeframe: str = "1m",
                                      train_months: int = 12, val_months: int = 2, 
                                      test_months: int = 2) -> List[Dict]:
        """Get walk-forward validation fold timestamps."""
        session = await self._get_session()
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "trainMonths": train_months,
            "valMonths": val_months,
            "testMonths": test_months
        }
        async with session.get(f"{self.dashboard_url}/api/gpu-export/walk-forward-folds", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("folds", [])
            return []
    
    async def fetch_multi_tf_candles(self, symbol: str, base_tf: str, 
                                      start_ts: int, end_ts: int, 
                                      limit: int = 100000) -> pd.DataFrame:
        """Fetch multi-timeframe aligned candles with as-of joins."""
        session = await self._get_session()
        params = {
            "symbol": symbol,
            "baseTF": base_tf,
            "startTs": start_ts,
            "endTs": end_ts,
            "limit": limit
        }
        print(f"[Dashboard API] Fetching {symbol} {base_tf} from {start_ts} to {end_ts}...")
        
        async with session.get(f"{self.dashboard_url}/api/gpu-export/multi-tf", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                candles = data.get("candles", [])
                if candles:
                    df = pd.DataFrame(candles)
                    print(f"[Dashboard API] Received {len(df)} candles with columns: {list(df.columns)}")
                    return df
            print(f"[Dashboard API] Error: HTTP {resp.status}")
            return pd.DataFrame()
    
    async def fetch_cross_asset_candles(self, base_tf: str, start_ts: int, end_ts: int,
                                         limit: int = 100000) -> pd.DataFrame:
        """Fetch cross-asset aligned candles (BTC/ETH/SOL/BNB)."""
        session = await self._get_session()
        params = {
            "baseTF": base_tf,
            "startTs": start_ts,
            "endTs": end_ts,
            "limit": limit
        }
        print(f"[Dashboard API] Fetching cross-asset data for {base_tf}...")
        
        async with session.get(f"{self.dashboard_url}/api/gpu-export/cross-asset", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                candles = data.get("candles", [])
                if candles:
                    df = pd.DataFrame(candles)
                    print(f"[Dashboard API] Received {len(df)} cross-asset rows")
                    return df
            return pd.DataFrame()
    
    async def push_predictions(self, predictions: List[Dict], model_id: str) -> bool:
        """Push model predictions back to dashboard for ensemble integration."""
        session = await self._get_session()
        payload = {
            "predictions": predictions,
            "modelId": model_id,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        async with session.post(f"{self.dashboard_url}/api/gpu-export/predictions", json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"[Dashboard API] Pushed {result.get('received', 0)} predictions")
                return True
            return False
    
    def fetch_multi_tf_candles_sync(self, symbol: str, base_tf: str,
                                     start_ts: int, end_ts: int,
                                     limit: int = 100000) -> pd.DataFrame:
        """Synchronous version of fetch_multi_tf_candles."""
        import requests
        params = {
            "symbol": symbol,
            "baseTF": base_tf,
            "startTs": start_ts,
            "endTs": end_ts,
            "limit": limit
        }
        print(f"[Dashboard API Sync] Fetching {symbol} {base_tf}...")
        
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/multi-tf", params=params, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                candles = data.get("candles", [])
                if candles:
                    df = pd.DataFrame(candles)
                    print(f"[Dashboard API Sync] Received {len(df)} candles")
                    return df
        except Exception as e:
            print(f"[Dashboard API Sync] Error: {e}")
        return pd.DataFrame()
    
    def get_trainer_config_sync(self) -> Dict:
        """Synchronous version of get_trainer_config."""
        import requests
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/trainer-config", timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[Dashboard API Sync] Error getting config: {e}")
        return {}
    
    def get_trading_costs_sync(self) -> Dict:
        """Get trading costs for cost-adjusted edge computation."""
        import requests
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/trading-costs", timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[Dashboard API Sync] Error getting trading costs: {e}")
        return {"costs": {"totalRoundTrip": 0.0009}}
    
    async def get_trading_costs(self) -> Dict:
        """Get trading costs for cost-adjusted edge computation."""
        session = await self._get_session()
        async with session.get(f"{self.dashboard_url}/api/gpu-export/trading-costs") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"costs": {"totalRoundTrip": 0.0009}}
    
    async def get_enhanced_labels(self, symbol: str = "BTCUSDT", timeframe: str = "15m",
                                   start_ts: int = None, end_ts: int = None,
                                   limit: int = 10000) -> pd.DataFrame:
        """Get enhanced training labels with cost-adjusted edge and trade-worthiness."""
        session = await self._get_session()
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts
        
        async with session.get(f"{self.dashboard_url}/api/gpu-export/enhanced-labels", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                labels = data.get("labels", [])
                if labels:
                    df = pd.DataFrame(labels)
                    print(f"[Dashboard API] Received {len(df)} enhanced labels")
                    return df
            return pd.DataFrame()
    
    def get_enhanced_labels_sync(self, symbol: str = "BTCUSDT", timeframe: str = "15m",
                                  start_ts: int = None, end_ts: int = None,
                                  limit: int = 10000) -> pd.DataFrame:
        """Synchronous version of get_enhanced_labels."""
        import requests
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts
        
        try:
            resp = requests.get(f"{self.dashboard_url}/api/gpu-export/enhanced-labels", 
                               params=params, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                labels = data.get("labels", [])
                if labels:
                    df = pd.DataFrame(labels)
                    print(f"[Dashboard API Sync] Received {len(df)} enhanced labels")
                    return df
        except Exception as e:
            print(f"[Dashboard API Sync] Error getting enhanced labels: {e}")
        return pd.DataFrame()


def compute_features_from_spec(df: pd.DataFrame, feature_specs: List[Dict]) -> pd.DataFrame:
    """
    Compute features according to the specification from the dashboard.
    This ensures features match exactly between dashboard and GPU trainer.
    """
    features = pd.DataFrame(index=df.index)
    
    close = df.get('1m_close', df.get('close', df.get('btc_close')))
    if close is None:
        raise ValueError("No close price column found")
    
    high = df.get('1m_high', df.get('high', df.get('btc_high')))
    low = df.get('1m_low', df.get('low', df.get('btc_low')))
    open_price = df.get('1m_open', df.get('open', df.get('btc_open')))
    volume = df.get('1m_volume', df.get('volume', df.get('btc_volume')))
    
    for spec in feature_specs:
        name = spec['name']
        formula = spec['formula']
        window = spec.get('window', 20)
        
        try:
            if 'log_return' in name:
                lookback = int(name.split('_')[-1]) if '_' in name else 1
                features[name] = np.log(close / close.shift(lookback))
            
            elif name.startswith('volatility_') and name != 'volatility_regime':
                # Rolling volatility (std of log returns)
                log_ret = np.log(close / close.shift(1))
                features[name] = log_ret.rolling(window).std()
            
            elif 'ema_ratio' in name:
                ema = close.ewm(span=window, adjust=False).mean()
                features[name] = close / ema - 1
            
            elif name == 'rsi_14':
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                features[name] = 100 - (100 / (1 + rs))
            
            elif 'macd' in name:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                
                if name == 'macd_line':
                    features[name] = macd_line
                elif name == 'macd_signal':
                    features[name] = macd_signal
                elif name == 'macd_hist':
                    features[name] = macd_line - macd_signal
            
            elif name == 'atr_14_norm':
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                features[name] = atr / close
            
            elif name == 'body_ratio':
                features[name] = (close - open_price) / open_price
            
            elif name == 'wick_up_ratio':
                body_top = pd.concat([open_price, close], axis=1).max(axis=1)
                features[name] = (high - body_top) / open_price
            
            elif name == 'wick_dn_ratio':
                body_bottom = pd.concat([open_price, close], axis=1).min(axis=1)
                features[name] = (body_bottom - low) / open_price
            
            elif name == 'volume_log':
                features[name] = np.log1p(volume)
            
            elif name == 'volume_zscore':
                vol_mean = volume.rolling(window).mean()
                vol_std = volume.rolling(window).std()
                features[name] = (volume - vol_mean) / vol_std.replace(0, 1)
            
            elif 'trend_slope' in name:
                ema20 = close.ewm(span=20, adjust=False).mean()
                features[name] = ema20.diff(window) / window
            
            elif '_return_1' in name and name.startswith(('eth', 'sol', 'bnb')):
                asset = name.split('_')[0]
                asset_close = df.get(f'{asset}_close')
                if asset_close is not None:
                    features[name] = np.log(asset_close / asset_close.shift(1))
            
            elif '_corr_' in name:
                # Rolling correlations between BTC and other assets
                # e.g., btc_eth_corr_20 -> rolling(20).corr() of BTC and ETH returns
                parts = name.split('_')
                if len(parts) >= 4:
                    asset1 = parts[0]  # btc
                    asset2 = parts[1]  # eth, sol, bnb
                    corr_window = int(parts[-1])  # 20
                    
                    close1 = df.get(f'{asset1}_close', close)
                    close2 = df.get(f'{asset2}_close')
                    
                    if close2 is not None:
                        ret1 = np.log(close1 / close1.shift(1))
                        ret2 = np.log(close2 / close2.shift(1))
                        features[name] = ret1.rolling(corr_window).corr(ret2)
                    else:
                        features[name] = 0.0
            
            elif name == 'volatility_regime':
                # Quantile-bucket volatility_20 with thresholds [0.33, 0.67]
                # Matches FEATURE_SPECS: quantile_bucket(volatility_20, [0.33, 0.67])
                log_ret = np.log(close / close.shift(1))
                vol_20 = log_ret.rolling(20).std()
                
                # Rolling quantile thresholds
                q33 = vol_20.rolling(100, min_periods=20).quantile(0.33)
                q67 = vol_20.rolling(100, min_periods=20).quantile(0.67)
                
                # Map to bucket: 0=low, 1=medium, 2=high
                def map_bucket(row_idx):
                    v = vol_20.iloc[row_idx]
                    t33 = q33.iloc[row_idx]
                    t67 = q67.iloc[row_idx]
                    if pd.isna(v) or pd.isna(t33) or pd.isna(t67):
                        return 1  # medium (default)
                    if v < t33:
                        return 0  # low
                    elif v < t67:
                        return 1  # medium
                    else:
                        return 2  # high
                
                features[name] = pd.Series([map_bucket(i) for i in range(len(vol_20))], index=df.index)
            
            elif 'relative_strength' in name:
                asset = name.split('_')[0]
                asset_close = df.get(f'{asset}_close')
                if asset_close is not None:
                    btc_ret = np.log(close / close.shift(20))
                    asset_ret = np.log(asset_close / asset_close.shift(20))
                    features[name] = asset_ret - btc_ret
            
        except Exception as e:
            print(f"[Feature] Error computing {name}: {e}")
            features[name] = np.nan
    
    return features.fillna(0)


def compute_enhanced_labels(
    df: pd.DataFrame,
    horizons: List[int] = [15, 60, 240],
    trading_costs: float = 0.0009
) -> pd.DataFrame:
    """
    Compute enhanced training labels with:
    1. Cost-adjusted edge (return - trading costs)
    2. Trade-worthiness labels (positive edge + clean move)
    3. Sample weights (prioritize high-move samples)
    
    Args:
        df: DataFrame with close/high/low prices
        horizons: List of forward horizons in candles
        trading_costs: Round-trip trading cost (default 0.09%)
    
    Returns:
        DataFrame with enhanced labels
    """
    close = df.get('1m_close', df.get('close', df.get('btc_close')))
    high = df.get('1m_high', df.get('high', df.get('btc_high')))
    low = df.get('1m_low', df.get('low', df.get('btc_low')))
    
    if close is None:
        raise ValueError("No close price column found")
    
    labels = pd.DataFrame(index=df.index)
    
    for h in horizons:
        raw_return = np.log(close.shift(-h) / close)
        
        labels[f'return_{h}'] = raw_return
        
        edge = np.abs(raw_return) - trading_costs
        labels[f'edge_{h}'] = edge
        
        direction = np.where(
            raw_return > trading_costs, 1,
            np.where(raw_return < -trading_costs, -1, 0)
        )
        labels[f'direction_{h}'] = direction
        
        max_favorable = pd.Series(np.nan, index=df.index)
        max_adverse = pd.Series(np.nan, index=df.index)
        
        for i in range(len(df) - h):
            entry = close.iloc[i]
            exit_dir = np.sign(raw_return.iloc[i])
            
            future_highs = high.iloc[i+1:i+h+1]
            future_lows = low.iloc[i+1:i+h+1]
            
            if exit_dir >= 0:
                max_favorable.iloc[i] = (future_highs.max() - entry) / entry
                max_adverse.iloc[i] = (entry - future_lows.min()) / entry
            else:
                max_favorable.iloc[i] = (entry - future_lows.min()) / entry
                max_adverse.iloc[i] = (future_highs.max() - entry) / entry
        
        clean_move_ratio = max_favorable / (max_favorable + max_adverse + 1e-8)
        
        trade_worthy = ((edge > 0.001) & (clean_move_ratio > 0.5)).astype(float)
        labels[f'trade_worthy_{h}'] = trade_worthy
    
    max_abs_return = np.maximum.reduce([np.abs(labels[f'return_{h}']) for h in horizons])
    
    move_weight = np.power(max_abs_return / 0.01, 0.5)
    
    log_ret = np.log(close / close.shift(1))
    vol_20 = log_ret.rolling(20).std()
    
    volatility_multiplier = np.where(
        vol_20 > 0.015, 1.5,
        np.where(vol_20 < 0.005, 0.3, 1.0)
    )
    
    raw_weight = move_weight * volatility_multiplier
    labels['sample_weight'] = np.clip(raw_weight, 0.1, 5.0)
    
    labels['sample_weight'] = labels['sample_weight'].fillna(1.0)
    
    return labels


class WeightedTrainingDataset(Dataset):
    """
    PyTorch Dataset with sample weighting for improved training.
    Addresses the '90% chop' problem by weighting samples by move size.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        seq_len: int = 256,
        horizons: List[str] = ['edge_15', 'edge_60', 'edge_240']
    ):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.seq_len = seq_len
        self.horizons = horizons
        
        self.valid_indices = np.arange(seq_len, len(features) - max(240, 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]
        
        x = self.features[actual_idx - self.seq_len:actual_idx]
        
        y = self.labels[actual_idx]
        
        w = self.weights[actual_idx]
        
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(w, dtype=torch.float32)
        )
    
    @staticmethod
    def create_weighted_sampler(weights: np.ndarray, num_samples: Optional[int] = None):
        """
        Create a WeightedRandomSampler for DataLoader.
        Higher weights = more likely to be sampled.
        """
        from torch.utils.data import WeightedRandomSampler
        
        normalized = weights / weights.sum()
        
        if num_samples is None:
            num_samples = len(weights)
        
        return WeightedRandomSampler(
            weights=normalized,
            num_samples=num_samples,
            replacement=True
        )
