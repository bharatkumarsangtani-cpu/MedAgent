# =====================================================================
#  E-COMMERCE SALES ANALYTICS DASHBOARD
#  Data Handling and Visualization Lab  |  Pandas + Plotly + Gradio
#  Paste this whole block into ONE Jupyter cell and run it.
# =====================================================================
import os
import tempfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

warnings.filterwarnings("ignore")

# If you re-run this cell, close the previous dashboard first
if "demo" in globals():
    try:
        demo.close()
    except Exception:
        pass

# ---------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------
CURRENCY = "$"   # change to "₹", "€" etc. if you like
REQUIRED_COLUMNS = [
    "Order ID", "Order Date", "Product Name", "Category", "Quantity",
    "Unit Price", "Unit Cost", "Discount", "Region", "Customer ID", "Order Status",
]


def _key(name):
    """Normalise a column name so 'order_date', 'Order Date' and 'ORDERDATE' all match."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


COLUMN_LOOKUP = {_key(c): c for c in REQUIRED_COLUMNS}


# ---------------------------------------------------------------------
# 2. SAMPLE DATA GENERATOR (fixed seed -> same data every run)
# ---------------------------------------------------------------------
def generate_sample_data(n=800, seed=42):
    """Create a realistic (slightly 'dirty') e-commerce dataset with n rows."""
    rng = np.random.default_rng(seed)

    # (product, category, base price, cost as fraction of price)
    catalog = [
        ("Smartphone X", "Electronics", 699, 0.72), ("Laptop Pro 14", "Electronics", 1299, 0.78),
        ("Wireless Earbuds", "Electronics", 89, 0.55), ("Smart Watch", "Electronics", 199, 0.60),
        ("Running Shoes", "Fashion", 79, 0.45), ("Denim Jacket", "Fashion", 65, 0.50),
        ("Leather Wallet", "Fashion", 35, 0.40), ("Sunglasses", "Fashion", 55, 0.38),
        ("Air Fryer", "Home & Kitchen", 119, 0.62), ("Coffee Maker", "Home & Kitchen", 85, 0.58),
        ("Vacuum Cleaner", "Home & Kitchen", 249, 0.68), ("Cookware Set", "Home & Kitchen", 99, 0.52),
        ("Yoga Mat", "Sports", 25, 0.35), ("Dumbbell Set", "Sports", 75, 0.55),
        ("Cricket Bat", "Sports", 60, 0.50), ("Fitness Band", "Sports", 45, 0.48),
        ("Python Programming Book", "Books", 30, 0.45), ("Data Science Handbook", "Books", 40, 0.47),
        ("Notebook Pack", "Books", 12, 0.40), ("Novel Collection", "Books", 28, 0.42),
    ]
    names = np.array([c[0] for c in catalog])
    cats = np.array([c[1] for c in catalog])
    base = np.array([c[2] for c in catalog], dtype=float)
    ratio = np.array([c[3] for c in catalog], dtype=float)

    # More orders in later years and during Nov/Dec (festive season)
    dates = pd.date_range("2023-01-01", "2025-12-31")
    w = np.linspace(1, 1.8, len(dates)) * np.where(dates.month.isin([11, 12]), 1.6, 1.0)
    order_dates = rng.choice(dates, size=n, p=w / w.sum())

    idx = rng.integers(0, len(catalog), n)
    df = pd.DataFrame({
        "Order ID": [f"ORD-{10001 + i}" for i in range(n)],
        "Order Date": pd.to_datetime(order_dates).strftime("%Y-%m-%d"),
        "Product Name": names[idx],
        "Category": cats[idx],
        "Quantity": rng.integers(1, 8, n),
        "Unit Price": np.round(base[idx] * rng.uniform(0.95, 1.05, n), 2),
        "Unit Cost": np.round(base[idx] * ratio[idx] * rng.uniform(0.95, 1.05, n), 2),
        # Discount is stored as a fraction (0.10 = 10 % off the order line)
        "Discount": rng.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20, 0.25], n),
        "Region": rng.choice(["North", "South", "East", "West", "Central"], n,
                             p=[0.25, 0.20, 0.20, 0.25, 0.10]),
        "Customer ID": [f"CUST-{x:04d}" for x in rng.integers(1, 301, n)],
        "Order Status": rng.choice(["Completed", "Shipped", "Cancelled", "Returned"], n,
                                   p=[0.72, 0.15, 0.08, 0.05]),
    })

    # Add a little "dirt" so the cleaning step has something to do
    df.loc[df.sample(15, random_state=seed).index, "Discount"] = np.nan
    df = pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)  # 10 duplicates
    return df


# ---------------------------------------------------------------------
# 3. DATA CLEANING + CALCULATED COLUMNS
# ---------------------------------------------------------------------
def _blank_to_na(series):
    """Turn empty strings / 'nan' / 'None' text into real missing values."""
    s = series.astype("string").str.strip()
    return s.mask(s.isin(["", "nan", "NaN", "None", "null"]))


def clean_data(raw):
    """Validate columns, clean values and add Revenue / Profit columns.
    Returns (clean_dataframe, list_of_log_messages). Raises ValueError on bad input."""
    log = []
    if raw is None or raw.empty:
        raise ValueError("The file contains no data.")
    df = raw.copy()

    # --- Validate and rename columns -----------------------------------
    df.columns = [COLUMN_LOOKUP.get(_key(c), str(c).strip()) for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    df = df[REQUIRED_COLUMNS]
    start_rows = len(df)

    # --- Remove duplicates ---------------------------------------------
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["Order ID", "Product Name"], keep="first")
    log.append(f"Duplicates removed: **{start_rows - len(df)}**")

    # --- Convert data types (bad values become NaN instead of crashing) --
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    for col in ["Quantity", "Unit Price", "Unit Cost", "Discount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Fill missing values -------------------------------------------
    cost_ratio = (df["Unit Cost"] / df["Unit Price"]).replace([np.inf, -np.inf], np.nan).median()
    if pd.isna(cost_ratio):
        cost_ratio = 0.7
    n_cost = int(df["Unit Cost"].isna().sum())
    df["Unit Cost"] = df["Unit Cost"].fillna(df["Unit Price"] * cost_ratio)  # typical cost ratio
    n_disc = int(df["Discount"].isna().sum())
    df["Discount"] = df["Discount"].fillna(0)                                # no discount
    log.append(f"Missing values filled: **{n_cost}** unit costs, **{n_disc}** discounts")

    # Discount rule: values above 1 are treated as percentages (10 -> 10 % -> 0.10).
    # The rate is always clipped to the safe range 0 - 1.
    if df["Discount"].max() > 1:
        df["Discount"] = df["Discount"] / 100
    df["Discount"] = df["Discount"].clip(0, 1)

    # --- Text columns --------------------------------------------------
    df["Order ID"] = _blank_to_na(df["Order ID"])
    for col in ["Product Name", "Category", "Region", "Customer ID"]:
        df[col] = _blank_to_na(df[col]).fillna("Unknown")
    df["Order Status"] = _blank_to_na(df["Order Status"]).str.title().fillna("Completed")

    # --- Drop rows that cannot be used ---------------------------------
    before = len(df)
    df = df.dropna(subset=["Order ID", "Order Date", "Quantity", "Unit Price"])
    df = df[(df["Quantity"] > 0) & (df["Unit Price"] >= 0) & (df["Unit Cost"] >= 0)]
    log.append(f"Invalid rows dropped: **{before - len(df)}**")
    if df.empty:
        raise ValueError("No valid rows remain after cleaning.")

    for col in ["Order ID", "Product Name", "Category", "Region", "Customer ID", "Order Status"]:
        df[col] = df[col].astype(str)
    df["Quantity"] = df["Quantity"].round().astype(int)

    # --- Calculated columns --------------------------------------------
    # Gross Sales      = Quantity x Unit Price
    # Discount Amount  = Gross Sales x Discount rate
    # Revenue          = Gross Sales - Discount Amount
    # Profit           = Revenue - (Quantity x Unit Cost)
    df["Gross Sales"] = df["Quantity"] * df["Unit Price"]
    df["Discount Amount"] = df["Gross Sales"] * df["Discount"]
    df["Revenue"] = df["Gross Sales"] - df["Discount Amount"]
    df["Profit"] = df["Revenue"] - df["Quantity"] * df["Unit Cost"]
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["Order Date"].dt.year

    return df.reset_index(drop=True), log


# ---------------------------------------------------------------------
# 4. FILTERS AND ANALYSIS
# ---------------------------------------------------------------------
def _to_ts(value):
    """Safely convert text from the date boxes into a Timestamp (or NaT)."""
    if value is None or str(value).strip() == "":
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors="coerce")


def apply_filters(df, start, end, cats, regs):
    """Filter by date range, category and region (empty selection = all)."""
    out = df
    s, e = _to_ts(start), _to_ts(end)
    if pd.notna(s):
        out = out[out["Order Date"] >= s]
    if pd.notna(e):
        out = out[out["Order Date"] < e + pd.Timedelta(days=1)]  # end date is inclusive
    if cats:
        out = out[out["Category"].isin(cats)]
    if regs:
        out = out[out["Region"].isin(regs)]
    return out


def completed_sales(df):
    """Completed-sales view: every order that is NOT cancelled."""
    return df[df["Order Status"].str.lower() != "cancelled"]


def compute_kpis(sales):
    revenue = float(sales["Revenue"].sum())
    profit = float(sales["Profit"].sum())
    orders = int(sales["Order ID"].nunique())
    return {
        "revenue": revenue,
        "profit": profit,
        "orders": orders,
        "units": int(sales["Quantity"].sum()),
        "aov": revenue / orders if orders else 0.0,
        "margin": profit / revenue * 100 if revenue else 0.0,
    }


def product_summary(sales):
    """Per-product performance table."""
    if sales.empty:
        return pd.DataFrame(columns=["Product Name", "Category", "Units Sold", "Orders",
                                     "Revenue", "Profit", "Profit Margin %"])
    t = sales.groupby(["Product Name", "Category"], as_index=False).agg(
        **{"Units Sold": ("Quantity", "sum"), "Orders": ("Order ID", "nunique"),
           "Revenue": ("Revenue", "sum"), "Profit": ("Profit", "sum")})
    t["Profit Margin %"] = np.where(t["Revenue"] > 0, t["Profit"] / t["Revenue"] * 100, 0)
    return t.sort_values("Profit", ascending=False).round(2).reset_index(drop=True)


# ---------------------------------------------------------------------
# 5. CHARTS (Plotly Express)
# ---------------------------------------------------------------------
def _style(fig, title):
    fig.update_layout(template="plotly_white", title=title, height=380,
                      margin=dict(l=20, r=20, t=60, b=20), legend_title_text="")
    return fig


def empty_fig(title):
    fig = go.Figure()
    fig.add_annotation(text="No data for the selected filters", x=0.5, y=0.5,
                       xref="paper", yref="paper", showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title)


def make_charts(sales):
    """Return the 7 required charts as a list of Plotly figures."""
    titles = ["Monthly Revenue Trend", "Yearly Revenue Comparison", "Sales by Category",
              "Sales by Region", "Top 10 Profitable Products", "Top 10 Best-Selling Products",
              "Revenue vs Profit by Category"]
    if sales.empty:
        return [empty_fig(t) for t in titles]

    # 1. Monthly revenue trend
    monthly = sales.groupby("Month", as_index=False)["Revenue"].sum()
    f1 = px.line(monthly, x="Month", y="Revenue", markers=True)
    f1.update_traces(line_color="#4f46e5")

    # 2. Yearly revenue comparison
    yearly = sales.groupby("Year", as_index=False)["Revenue"].sum()
    yearly["Year"] = yearly["Year"].astype(str)
    f2 = px.bar(yearly, x="Year", y="Revenue", color="Year", text_auto=".3s")

    # 3. Sales by category
    cat = sales.groupby("Category", as_index=False)["Revenue"].sum()
    f3 = px.pie(cat, names="Category", values="Revenue", hole=0.45)

    # 4. Sales by region
    reg = sales.groupby("Region", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    f4 = px.bar(reg, x="Region", y="Revenue", color="Region", text_auto=".3s")

    # 5 & 6. Top-10 products
    prod = product_summary(sales)
    top_p = prod.nlargest(10, "Profit").sort_values("Profit")
    f5 = px.bar(top_p, x="Profit", y="Product Name", orientation="h",
                color="Profit", color_continuous_scale="Teal")
    top_u = prod.nlargest(10, "Units Sold").sort_values("Units Sold")
    f6 = px.bar(top_u, x="Units Sold", y="Product Name", orientation="h",
                color="Units Sold", color_continuous_scale="Sunset")

    # 7. Revenue vs profit
    rp = sales.groupby("Category", as_index=False)[["Revenue", "Profit"]].sum()
    rp = rp.melt(id_vars="Category", var_name="Metric", value_name="Amount")
    f7 = px.bar(rp, x="Category", y="Amount", color="Metric", barmode="group",
                color_discrete_map={"Revenue": "#4f46e5", "Profit": "#10b981"})

    figs = [f1, f2, f3, f4, f5, f6, f7]
    for fig, t in zip(figs, titles):
        _style(fig, t)
    f2.update_layout(showlegend=False)
    f4.update_layout(showlegend=False)
    f5.update_layout(coloraxis_showscale=False)
    f6.update_layout(coloraxis_showscale=False)
    return figs


# ---------------------------------------------------------------------
# 6. KPI CARDS (HTML)
# ---------------------------------------------------------------------
def kpi_html(k, cancelled):
    cards = [
        ("Total Revenue", f"{CURRENCY}{k['revenue']:,.2f}", "#4f46e5, #7c3aed"),
        ("Total Profit", f"{CURRENCY}{k['profit']:,.2f}", "#059669, #10b981"),
        ("Total Orders", f"{k['orders']:,}", "#2563eb, #0ea5e9"),
        ("Units Sold", f"{k['units']:,}", "#d97706, #f59e0b"),
        ("Avg Order Value", f"{CURRENCY}{k['aov']:,.2f}", "#db2777, #ec4899"),
        ("Profit Margin", f"{k['margin']:.2f}%", "#0f766e, #14b8a6"),
        ("Cancelled Orders", f"{cancelled:,}", "#b91c1c, #ef4444"),
    ]
    html = "<div class='kpi-grid'>"
    for label, value, grad in cards:
        html += (f"<div class='kpi-card' style='background:linear-gradient(135deg,{grad})'>"
                 f"<div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>")
    return html + "</div>"


# ---------------------------------------------------------------------
# 7. GRADIO EVENT HANDLERS
# ---------------------------------------------------------------------
def _ui_after_load(df, message):
    """Updates for state, status text, date boxes and dropdowns after a dataset loads."""
    cats = sorted(df["Category"].unique())
    regs = sorted(df["Region"].unique())
    return (df, message,
            df["Order Date"].min().strftime("%Y-%m-%d"), df["Order Date"].max().strftime("%Y-%m-%d"),
            gr.update(choices=cats, value=cats), gr.update(choices=regs, value=regs))


def load_sample(current_df):
    try:
        df, log = clean_data(generate_sample_data())
        msg = f"✅ **Sample dataset loaded** — {len(df):,} clean rows.<br>" + " · ".join(log)
        return _ui_after_load(df, msg)
    except Exception as err:
        return (current_df, f"❌ Could not load sample data: {err}",
                gr.update(), gr.update(), gr.update(), gr.update())


def load_csv(path, current_df):
    if path is None:  # file box was cleared
        return (current_df, gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    try:
        path = getattr(path, "name", path)
        try:
            raw = pd.read_csv(path)
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="latin-1")
        df, log = clean_data(raw)
        msg = f"✅ **{os.path.basename(path)} loaded** — {len(df):,} clean rows.<br>" + " · ".join(log)
        return _ui_after_load(df, msg)
    except Exception as err:
        return (current_df, f"❌ **Upload failed:** {err}",
                gr.update(), gr.update(), gr.update(), gr.update())


def reset_filters(df):
    if df is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    cats, regs = sorted(df["Category"].unique()), sorted(df["Region"].unique())
    return (df["Order Date"].min().strftime("%Y-%m-%d"), df["Order Date"].max().strftime("%Y-%m-%d"),
            gr.update(value=cats), gr.update(value=regs))


def build_dashboard(df, start, end, cats, regs):
    """Recalculate EVERYTHING from the filtered data so KPIs, charts and tables always agree."""
    if df is None:
        empty = pd.DataFrame()
        return ["<p>Load the sample dataset or upload a CSV to begin.</p>"] + \
               [empty_fig("No data")] * 7 + [empty, empty]

    filtered = apply_filters(df, start, end, cats, regs)
    sales = completed_sales(filtered)
    cancelled = int(filtered.loc[~filtered.index.isin(sales.index), "Order ID"].nunique())

    html = kpi_html(compute_kpis(sales), cancelled)
    figs = make_charts(sales)
    table = product_summary(sales)

    preview_cols = REQUIRED_COLUMNS + ["Discount Amount", "Revenue", "Profit"]
    preview = filtered[preview_cols].head(200).copy()
    preview["Order Date"] = preview["Order Date"].dt.strftime("%Y-%m-%d")
    preview = preview.round(2)
    return [html] + figs + [table, preview]


def _new_path(prefix, ext):
    return os.path.join(tempfile.mkdtemp(), f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.{ext}")


def export_clean(df, start, end, cats, regs):
    """Save the (filtered) cleaned dataset as CSV."""
    if df is None:
        raise gr.Error("Load a dataset first.")
    out = apply_filters(df, start, end, cats, regs).copy()
    if out.empty:
        raise gr.Error("No rows match the current filters.")
    out["Order Date"] = out["Order Date"].dt.strftime("%Y-%m-%d")
    out["Month"] = out["Month"].dt.strftime("%Y-%m")
    path = _new_path("cleaned_sales", "csv")
    out.round(2).to_csv(path, index=False)
    return path


def export_report(df, start, end, cats, regs):
    """Save an Excel sales report (KPIs + summary tables) for the current filters."""
    if df is None:
        raise gr.Error("Load a dataset first.")
    sales = completed_sales(apply_filters(df, start, end, cats, regs))
    if sales.empty:
        raise gr.Error("No completed sales match the current filters.")
    k = compute_kpis(sales)
    kpi_df = pd.DataFrame({"Metric": ["Total Revenue", "Total Profit", "Total Orders", "Units Sold",
                                      "Average Order Value", "Profit Margin %"],
                           "Value": [k["revenue"], k["profit"], k["orders"], k["units"],
                                     k["aov"], k["margin"]]}).round(2)
    monthly = sales.groupby("Month", as_index=False)[["Revenue", "Profit"]].sum()
    monthly["Month"] = monthly["Month"].dt.strftime("%Y-%m")
    yearly = sales.groupby("Year", as_index=False)[["Revenue", "Profit"]].sum()
    cat = sales.groupby("Category", as_index=False)[["Revenue", "Profit"]].sum()
    reg = sales.groupby("Region", as_index=False)[["Revenue", "Profit"]].sum()
    path = _new_path("sales_report", "xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        kpi_df.to_excel(writer, "KPIs", index=False)
        monthly.round(2).to_excel(writer, "Monthly", index=False)
        yearly.round(2).to_excel(writer, "Yearly", index=False)
        cat.round(2).to_excel(writer, "Category", index=False)
        reg.round(2).to_excel(writer, "Region", index=False)
        product_summary(sales).to_excel(writer, "Products", index=False)
    return path


# ---------------------------------------------------------------------
# 8. USER INTERFACE
# ---------------------------------------------------------------------
CSS = """
<style>
.hero{background:linear-gradient(135deg,#1e1b4b,#4f46e5 60%,#06b6d4);color:#fff;
      padding:22px 26px;border-radius:16px;margin-bottom:8px}
.hero h1{margin:0;font-size:1.8rem;color:#fff}
.hero p{margin:6px 0 0;opacity:.92}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:6px 0 10px}
.kpi-card{border-radius:14px;padding:16px 18px;color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.18)}
.kpi-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;opacity:.92}
.kpi-value{font-size:1.55rem;font-weight:700;margin-top:4px;word-break:break-word}
</style>
"""

with gr.Blocks(title="E-Commerce Sales Analytics Dashboard") as demo:
    df_state = gr.State(None)  # holds the cleaned dataset for this session

    gr.HTML(CSS + "<div class='hero'><h1>🛒 E-Commerce Sales Analytics Dashboard</h1>"
                  "<p>Explore revenue, profit, products, categories and regions. Load the sample data "
                  "or upload your own CSV, then use the filters. Cancelled orders are excluded from "
                  "all sales metrics.</p></div>")

    with gr.Row():
        # ---------------- Left panel: data, filters, downloads ----------------
        with gr.Column(scale=1, min_width=290):
            gr.Markdown("### 📂 Data")
            file_in = gr.File(label="Upload CSV dataset", file_types=[".csv"], type="filepath")
            btn_sample = gr.Button("Load Sample Dataset", variant="primary")
            status = gr.Markdown("Loading sample data…")

            gr.Markdown("### 🔎 Filters")
            start_box = gr.Textbox(label="Start date (YYYY-MM-DD)")
            end_box = gr.Textbox(label="End date (YYYY-MM-DD)")
            cat_dd = gr.Dropdown(label="Category (empty = all)", choices=[], multiselect=True)
            reg_dd = gr.Dropdown(label="Region (empty = all)", choices=[], multiselect=True)
            with gr.Row():
                btn_apply = gr.Button("Apply Filters", variant="primary")
                btn_reset = gr.Button("Reset")

            gr.Markdown("### ⬇️ Downloads")
            btn_clean = gr.Button("Prepare Cleaned Dataset (CSV)")
            file_clean = gr.File(label="Cleaned dataset", interactive=False)
            btn_report = gr.Button("Prepare Sales Report (Excel)")
            file_report = gr.File(label="Sales report", interactive=False)

        # ---------------- Right panel: KPIs, charts, tables ----------------
        with gr.Column(scale=3):
            kpi_out = gr.HTML()
            with gr.Tabs():
                with gr.Tab("📈 Trends"):
                    with gr.Row():
                        p1 = gr.Plot(label="Monthly Revenue Trend")
                        p2 = gr.Plot(label="Yearly Revenue Comparison")
                with gr.Tab("🧭 Category & Region"):
                    with gr.Row():
                        p3 = gr.Plot(label="Sales by Category")
                        p4 = gr.Plot(label="Sales by Region")
                    p7 = gr.Plot(label="Revenue vs Profit")
                with gr.Tab("🏆 Products"):
                    with gr.Row():
                        p5 = gr.Plot(label="Top 10 Profitable Products")
                        p6 = gr.Plot(label="Top 10 Best-Selling Products")
                    gr.Markdown("#### Product Performance Table")
                    product_tbl = gr.Dataframe(interactive=False, wrap=True)
                with gr.Tab("🧾 Cleaned Data"):
                    gr.Markdown("Preview of the first 200 cleaned rows (after filters, cancelled orders included).")
                    preview_tbl = gr.Dataframe(interactive=False, wrap=True)

    # ---------------- Wiring: connect buttons / filters to functions ----------------
    load_outputs = [df_state, status, start_box, end_box, cat_dd, reg_dd]
    filter_inputs = [df_state, start_box, end_box, cat_dd, reg_dd]
    dash_outputs = [kpi_out, p1, p2, p3, p4, p5, p6, p7, product_tbl, preview_tbl]

    btn_sample.click(load_sample, [df_state], load_outputs).then(build_dashboard, filter_inputs, dash_outputs)
    file_in.change(load_csv, [file_in, df_state], load_outputs).then(build_dashboard, filter_inputs, dash_outputs)
    demo.load(load_sample, [df_state], load_outputs).then(build_dashboard, filter_inputs, dash_outputs)

    btn_apply.click(build_dashboard, filter_inputs, dash_outputs)
    start_box.submit(build_dashboard, filter_inputs, dash_outputs)
    end_box.submit(build_dashboard, filter_inputs, dash_outputs)
    cat_dd.change(build_dashboard, filter_inputs, dash_outputs)
    reg_dd.change(build_dashboard, filter_inputs, dash_outputs)
    btn_reset.click(reset_filters, [df_state], [start_box, end_box, cat_dd, reg_dd]) \
             .then(build_dashboard, filter_inputs, dash_outputs)

    btn_clean.click(export_clean, filter_inputs, file_clean)
    btn_report.click(export_report, filter_inputs, file_report)

# Launch inside the notebook (no command line needed)
demo.launch(share=False, inline=True, theme = "seafoam")