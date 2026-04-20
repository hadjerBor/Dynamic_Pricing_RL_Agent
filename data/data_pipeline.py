"""
FILE: data/data_pipeline.py  [UPDATED v2]
CHANGES: Fixed UCI download (direct HTTP), outlier removal in elasticity fit
"""
import os, io, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

def download_uci_retail():
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx",
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/online_retail.csv",
    ]
    for url in urls:
        try:
            import urllib.request
            print(f"[DATA] Trying: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            if url.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(raw), sheet_name=0, engine="openpyxl")
            else:
                df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")))
            df.columns = [c.strip().replace(" ", "") for c in df.columns]
            rename_map = {"UnitPrice":"Price","Unitprice":"Price","InvoiceNo":"Invoice"}
            df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
            print(f"[DATA] Downloaded {len(df):,} rows")
            return df
        except Exception as e:
            print(f"[DATA] Failed ({type(e).__name__}: {e})")
    print("[DATA] All downloads failed — using synthetic data only.")
    return None

def clean_real_data(df):
    if df is None: return None
    df = df.copy()
    inv_col = "Invoice" if "Invoice" in df.columns else "InvoiceNo"
    if inv_col in df.columns:
        df = df[~df[inv_col].astype(str).str.startswith("C")]
    for col in ["Quantity","Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df.get("Quantity",pd.Series([1]))>0)&(df.get("Price",pd.Series([1]))>0)].dropna(
        subset=[c for c in ["Quantity","Price","StockCode","InvoiceDate"] if c in df.columns])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["Date"] = df["InvoiceDate"].dt.date
    top = df.groupby("StockCode")["Quantity"].count().nlargest(20).index.tolist()
    df  = df[df["StockCode"].isin(top)]
    print(f"[CLEAN] {len(df):,} rows, {df['StockCode'].nunique()} products")
    return df

def compute_daily_demand(df):
    if df is None: return {}
    product_data = {}
    for code, grp in df.groupby("StockCode"):
        daily = (grp.groupby("Date")
                    .agg(avg_price=("Price","mean"), total_qty=("Quantity","sum"))
                    .reset_index().sort_values("Date"))
        if len(daily) >= 30:
            product_data[code] = daily
    return product_data

def estimate_elasticity(daily_df):
    df = daily_df[(daily_df["avg_price"]>0)&(daily_df["total_qty"]>0)].copy()
    if len(df) < 10: return -1.5, df["total_qty"].mean(), df["avg_price"].mean()
    lp = np.log(df["avg_price"].values)
    lq = np.log(df["total_qty"].values)
    mask = (np.abs(lp-lp.mean())<3*lp.std())&(np.abs(lq-lq.mean())<3*lq.std())
    lp, lq = lp[mask], lq[mask]
    X = np.column_stack([np.ones(len(lp)), lp])
    try:
        coef,*_ = np.linalg.lstsq(X, lq, rcond=None)
        elasticity = np.clip(float(coef[1]), -5.0, -0.1)
    except: elasticity = -1.5
    return elasticity, float(df["total_qty"].mean()), float(df["avg_price"].mean())

def generate_synthetic_products(n_products=10, n_days=1095, seed=42):
    rng = np.random.RandomState(seed)
    start_date = datetime(2020,1,1)
    categories = [
        {"name":"Electronics",   "price_range":(50,300),  "elasticity_range":(-2.5,-0.8)},
        {"name":"Clothing",      "price_range":(15,80),   "elasticity_range":(-3.0,-1.0)},
        {"name":"Home & Garden", "price_range":(10,100),  "elasticity_range":(-2.0,-0.5)},
        {"name":"Books",         "price_range":(5,40),    "elasticity_range":(-1.5,-0.3)},
        {"name":"Toys",          "price_range":(8,60),    "elasticity_range":(-2.5,-1.2)},
    ]
    products = []
    dates    = [start_date+timedelta(days=i) for i in range(n_days)]
    date_arr = np.arange(n_days)
    for i in range(n_products):
        cat=categories[i%len(categories)]
        base_price=rng.uniform(*cat["price_range"]); elasticity=rng.uniform(*cat["elasticity_range"])
        base_demand=rng.uniform(20,200); trend_rate=rng.uniform(-0.0002,0.0003)
        noise_std=rng.uniform(0.05,0.20); weekly_amp=rng.uniform(0.05,0.20)
        annual_amp=rng.uniform(0.10,0.40); annual_peak=rng.uniform(300,360)
        seasonal_factor=(1+weekly_amp*np.sin(2*np.pi*date_arr/7)
                          +annual_amp*np.sin(2*np.pi*(date_arr-annual_peak)/365)
                          +trend_rate*date_arr)
        price_noise=rng.randn(n_days)*0.02*base_price
        hist_prices=np.clip(base_price+np.cumsum(price_noise)*0.1, base_price*0.5, base_price*2.0)
        demand_at_hist=base_demand*seasonal_factor*(hist_prices/base_price)**elasticity
        noisy_demand=np.array([max(0,rng.poisson(max(1,d))) for d in demand_at_hist])
        df=pd.DataFrame({"date":[d.strftime("%Y-%m-%d") for d in dates],
            "day_of_week":[(start_date+timedelta(days=int(d))).weekday() for d in date_arr],
            "day_of_year":[(start_date+timedelta(days=int(d))).timetuple().tm_yday for d in date_arr],
            "price":hist_prices,"demand":noisy_demand,"revenue":hist_prices*noisy_demand,"seasonal":seasonal_factor})
        products.append({"id":f"P{i:03d}","name":f"{cat['name']} Product {i+1}","category":cat["name"],
            "base_price":round(base_price,2),"elasticity":round(elasticity,3),"base_demand":round(base_demand,1),
            "price_min":round(base_price*0.5,2),"price_max":round(base_price*2.0,2),"noise_std":round(noise_std,3),
            "timeseries":df})
    print(f"[SYNTH] Generated {n_products} products × {n_days} days")
    return products

def augment_with_real_data(synthetic_products, real_product_data):
    if not real_product_data: return synthetic_products
    for i, code in enumerate(list(real_product_data.keys())[:len(synthetic_products)]):
        elast,base_demand,base_price=estimate_elasticity(real_product_data[code])
        synthetic_products[i].update({"id":code,"base_price":round(base_price,2),
            "elasticity":round(elast,3),"base_demand":round(base_demand,1),
            "price_min":round(base_price*0.5,2),"price_max":round(base_price*2.0,2)})
        print(f"  [{code}] elast={elast:.2f}  base_price=${base_price:.2f}")
    return synthetic_products

def save_products(products, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    metadata=[]
    for p in products:
        ts_path=os.path.join(output_dir,f"{p['id']}_timeseries.csv")
        p["timeseries"].to_csv(ts_path,index=False)
        meta={k:v for k,v in p.items() if k!="timeseries"}
        meta["timeseries_file"]=ts_path; metadata.append(meta)
    with open(os.path.join(output_dir,"products_metadata.json"),"w") as f:
        json.dump(metadata,f,indent=2)
    print(f"[SAVE] Saved {len(products)} products → {output_dir}/")
    return metadata

def load_products(data_dir="data/processed"):
    with open(os.path.join(data_dir,"products_metadata.json")) as f:
        metadata=json.load(f)
    for p in metadata:
        p["timeseries"]=pd.read_csv(p["timeseries_file"])
    return metadata

def build_dataset(n_products=10, n_days=1095, output_dir="data/processed"):
    print("="*60+"\nDYNAMIC PRICING — DATA PIPELINE v2\n"+"="*60)
    raw_df=download_uci_retail(); clean_df=clean_real_data(raw_df)
    real_data=compute_daily_demand(clean_df) if clean_df is not None else {}
    products=generate_synthetic_products(n_products=n_products, n_days=n_days)
    products=augment_with_real_data(products, real_data)
    save_products(products, output_dir=output_dir)
    print(f"\n[DONE] {len(products)} products × {n_days} days (~{n_days//365} years)")
    return products

if __name__=="__main__":
    build_dataset(n_products=10, n_days=1095)
