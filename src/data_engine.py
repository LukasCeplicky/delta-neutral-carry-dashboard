import sqlite3
import pandas as pd
import requests
import time
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataEngine:
    def __init__(self, db_name="market_data.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._init_db()
        
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_data (
                ticker TEXT,
                timestamp INTEGER,
                price REAL,
                funding REAL,
                PRIMARY KEY (ticker, timestamp)
            );
        """)
        self.conn.commit()

    # --- PUBLIC METHODS ---
    def get_universe(self):
        """
        Fetches the asset list specifically from the 'xyz' DEX.
        """
        print("--- RESOLVING HIP-3 UNIVERSE (DEX: xyz) ---")
        url = "https://api.hyperliquid.xyz/info"
        headers = {"Content-Type": "application/json"}
        
        try:
            # FIX: We now know the API wants the NAME 'xyz', not the ID.
            payload = {"type": "metaAndAssetCtxs", "dex": "xyz"}
            
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            data = r.json()
            
            # The structure is [PerpMeta, SpotMeta]
            universe = data[0]['universe']
            names = [a['name'] for a in universe]
            
            print(f"-> Found {len(names)} assets in 'xyz'.")
            print(f"-> Example: {names[0]}")
            return names
            
        except Exception as e:
            print(f"Error fetching universe: {e}")
            return []

    def update_data(self, tickers):
        """Master sync function"""
        print(f"\n--- SYNCING {len(tickers)} ASSETS ---")
        for i, ticker in enumerate(tickers):
            # Display simpler name for progress bar
            clean_name = ticker.split(':')[-1] if ':' in ticker else ticker
            progress = f"[{i+1}/{len(tickers)}] {clean_name.ljust(10)}"
            
            self._fetch_perp(ticker, progress)
            # Small sleep to be nice to the API
            time.sleep(0.1)

    def get_data(self, ticker):
        query = "SELECT * FROM hourly_data WHERE ticker = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, self.conn, params=(ticker,))
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['funding_apr'] = df['funding'] * 24 * 365 * 100
        return df

    # --- INTERNAL WORKERS ---
    def _fetch_perp(self, ticker, progress_msg):
        # 1. Determine Start Time
        cursor = self.conn.execute("SELECT MAX(timestamp) FROM hourly_data WHERE ticker = ?", (ticker,))
        last_ts = cursor.fetchone()[0]
        start_default = 1730419200000 # Nov 1 2024
        curr = (last_ts + 3600000) if last_ts else start_default

        print(f"\r{progress_msg} | DL...", end="", flush=True)

        # --- FETCH CANDLES ---
        candles = []
        seen_times = set()
        
        for _ in range(50): 
            try:
                payload = {
                    "type": "candleSnapshot", 
                    "req": {"coin": ticker, "interval": "1h", "startTime": curr}
                }
                
                r = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=5)
                
                if r.status_code != 200: break
                batch = r.json()
                if not batch: break
                
                last_candle_time = batch[-1]['t']
                if last_candle_time in seen_times: break
                seen_times.add(last_candle_time)
                
                candles.extend(batch)
                curr = last_candle_time + 1
                if curr > time.time() * 1000: break
                time.sleep(0.05)
            except: time.sleep(1); continue

        # --- FETCH FUNDING ---
        funding = []
        curr = (last_ts + 3600000) if last_ts else start_default
        seen_times = set()

        for _ in range(50):
            try:
                payload = {"type": "fundingHistory", "coin": ticker, "startTime": curr}
                r = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=5)
                
                if r.status_code != 200: break
                batch = r.json()
                if not batch: break
                
                batch = sorted(batch, key=lambda x: x['time'])
                last_fund_time = batch[-1]['time']
                if last_fund_time in seen_times: break
                seen_times.add(last_fund_time)
                
                funding.extend(batch)
                curr = last_fund_time + 1
                if curr > time.time() * 1000: break
                time.sleep(0.05)
            except: time.sleep(1); continue

        if not candles and not funding:
            print(f" OK (NoData)", flush=True)
            return

        # --- PROCESS & SAVE ---
        try:
            df_p = pd.DataFrame()
            if candles:
                df_p = pd.DataFrame(candles).rename(columns={'t': 'timestamp', 'c': 'price'})
                df_p['price'] = pd.to_numeric(df_p['price'])
                df_p['timestamp'] = (df_p['timestamp'] // 3600000) * 3600000
                df_p = df_p.drop_duplicates(subset=['timestamp'])

            df_f = pd.DataFrame()
            if funding:
                df_f = pd.DataFrame(funding).rename(columns={'time': 'timestamp', 'fundingRate': 'funding'})
                df_f['funding'] = pd.to_numeric(df_f['funding'])
                df_f['timestamp'] = (df_f['timestamp'] // 3600000) * 3600000 
                df_f = df_f.groupby('timestamp')['funding'].sum().reset_index()

            if not df_p.empty:
                if not df_f.empty:
                    merged = pd.merge(df_p, df_f, on='timestamp', how='left').fillna(0)
                else:
                    merged = df_p
                    merged['funding'] = 0.0
            else:
                return 

            merged['ticker'] = ticker
            merged[['ticker', 'timestamp', 'price', 'funding']].to_sql('hourly_data', self.conn, if_exists='append', index=False)
            print(f" +{len(merged)}h", flush=True)
            
        except Exception as e:
            print(f" Err Save: {str(e)[:10]}", flush=True)