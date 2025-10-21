# app.py
# Masabroso – Dashboard Unificado (flujo centralizado reactivo)
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
import unicodedata
import re

# ============================ CONFIG =========================================
st.set_page_config(page_title="Masabroso – Dashboard Unificado", layout="wide")
# Soporta ejecutar streamlit desde distintos cwd (VSCode, terminal, etc.)
DATA_DIRS = [
    Path(__file__).resolve().parent / "data",
    Path.cwd() / "data",
]

# Alias compatible para el resto del código y para escribir archivos
DATA_DIR = next((p for p in DATA_DIRS if p.exists()), DATA_DIRS[0])
DATA_DIR.mkdir(parents=True, exist_ok=True)

MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
N = 12
PAN_PIEZAS_POR_KG = 10  # 10 piezas de 100 g = 1 kg

def clp(x) -> str:
    try:
        return f"$ {int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return str(x)

# === Formato dinero (CLP) ===========================
def fmt_money(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].map(clp)
    return df2

def show_df_money(df: pd.DataFrame, money_cols: list[str] | None = None, **kwargs):
    # Si no se especifican, detecta columnas de dinero por nombre
    if money_cols is None:
        money_cols = [c for c in df.columns
                      if re.search(r"(precio|venta|ventas|cogs|costo|clp|iva|o?pex|ebitda|aporte|cuota|interes|amort|saldo|caja|flujo|material|capex)", str(c), re.I)]
    st.dataframe(fmt_money(df, money_cols), **kwargs)

def metric_clp(label: str, value):
    st.metric(label, clp(value))
# ====================================================



# ============================ HELPERS ========================================
def _norm_txt(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-zA-ZÁÉÍÓÚáéíóúñÑ ]+", " ", s)
    s = " ".join(s.lower().split())
    return s

def _num(x):
    try:
        if x is None: return np.nan
        if isinstance(x,(int,float,np.number)): return float(x)
        s = str(x).replace("$","").replace(".","").replace(",","").strip()
        return float(s) if s else np.nan
    except Exception:
        return np.nan

def read_first(fname: str, default: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Lector robusto para CSV en data/.
    - Busca el archivo en varias bases (DATA_DIRS)
    - Soporta UTF-8/UTF-8-SIG/latin1
    - Autodetecta delimitador (incluye ';')
    - Ignora filas problemáticas
    - Limpia BOM en headers
    """
    # 1) localizar el archivo en alguna base de datos
    p = None
    for base in DATA_DIRS:
        cand = base / fname
        if cand.exists():
            p = cand
            break
    if p is None:
        return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()

    # 2) intentos de parseo
    tries = [
        dict(sep=",",  encoding="utf-8"),
        dict(sep=None, engine="python", encoding="utf-8"),      # autodetect
        dict(sep=None, engine="python", encoding="utf-8-sig"),  # BOM
        dict(sep=";",  engine="python", encoding="utf-8"),
        dict(sep=";",  engine="python", encoding="utf-8-sig"),
        dict(sep=";",  engine="python", encoding="latin1"),
        dict(sep=",",  engine="python", encoding="latin1"),
    ]

    for kw in tries:
        try:
            df = pd.read_csv(p, on_bad_lines="skip", **kw)
        except TypeError:
            # pandas < 1.3
            df = pd.read_csv(p, error_bad_lines=False, **kw)
        except Exception:
            continue

        if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
            df = df.copy()
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            return df

    # último recurso
    try:
        df = pd.read_table(p, engine="python")
        if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            return df
    except Exception:
        pass

    return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()

def s_factor_vec(wd, we, D, F):
    D = np.array(D, dtype=float)
    F = np.array(F, dtype=float)
    H = np.maximum(D - F, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (wd*H + we*F) / np.where(D>0, D, 1.0)
    out[~np.isfinite(out)] = 1.0
    return out.tolist()

def rampa_geom(R1, R12, n=12):
    if R1 <= 0 or R12 <= 0: return [1.0]*n
    r = (R12/R1)**(1/(n-1))
    return [R1*(r**i) for i in range(n)]

# Unidades y densidades para conversiones (insumos)
_UNIT_ALIASES = {
    "g":"g","gr":"g","gramo":"g","gramos":"g",
    "kg":"kg","kilo":"kg","kilogramo":"kg","kilogramos":"kg",
    "ml":"ml","mililitro":"ml","mililitros":"ml",
    "l":"l","lt":"l","litro":"l","litros":"l",
    "u":"u","unid":"u","unidad":"u","unidades":"u"
}
def _u(u_raw): return _UNIT_ALIASES.get(_norm_txt(u_raw), _norm_txt(u_raw))

# Densidades g/ml
DENS = {"agua":1.00,"leche":1.03,"leche entera":1.03}

def convert_amount(cant, u_from, u_to, insumo_norm=""):
    """Convierte cantidades entre g/kg/ml/l/u. Cruces masa-volumen con densidad si aplica."""
    if pd.isna(cant): return np.nan
    u_from = _u(u_from); u_to = _u(u_to)

    if u_from == u_to: return cant
    # masa
    if u_from == "g"  and u_to == "kg": return cant/1000.0
    if u_from == "kg" and u_to == "g":  return cant*1000.0
    # volumen
    if u_from == "ml" and u_to == "l":  return cant/1000.0
    if u_from == "l"  and u_to == "ml": return cant*1000.0
    # unidad
    if u_from == "u" and u_to == "u":   return cant

    # cruces masa-volumen
    d = None
    for k,v in DENS.items():
        if k in insumo_norm:
            d = v; break
    if d is not None:
        if u_from == "g"  and u_to == "ml": return cant / d
        if u_from == "ml" and u_to == "g":  return cant * d
        if u_from == "kg" and u_to == "l":  return cant / d
        if u_from == "l"  and u_to == "kg": return cant * d

    return np.nan  # no convertible

def s2df(s: pd.Series, value_name: str, index_name: str = "Mes") -> pd.DataFrame:
    return s.rename_axis(index_name).reset_index(name=value_name)

# ============================ 02c – helpers PRO ==============================
PRO_SKUS = [
    {"Linea": "Pan (kg)", "SKU": "Pan Masa Madre 1kg",     "precio_sugerido": 3500, "base_dia": 40},
    {"Linea": "Pan (kg)", "SKU": "Paneton Masa Madre 1kg", "precio_sugerido": 5000, "base_dia": 15},
]

sku_pro_df = pd.DataFrame()
consumo_concentrado_df = pd.DataFrame()

def _pro_agg_from_skus(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"Línea": "PRO", "unidades": 0, "ingreso": 0.0, "cogs": 0.0, "margen": 0.0}
    out = {
        "Línea": "PRO",
        "unidades": int(pd.to_numeric(df.get("unidades", df.get("Unidades", 0)), errors="coerce").fillna(0).sum()),
        "ingreso": float(pd.to_numeric(df.get("ingreso", df.get("Venta_mes", 0.0)), errors="coerce").fillna(0.0).sum()),
    }
    for col in ["cogs", "margen"]:
        if col in df.columns:
            out[col] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    return out

_NESTED_INS = _norm_txt("Masa madre (concentrado)")
_NESTED_SKU_ALIASES = ["Concentrado Masa Madre 1kg", "Concentrado Masa Madre 1 kg"]

def costeo_sku_pro(sku_name: str, fabricar_concentrado: bool = True):
    det, c = costeo_insumos_por_sku(BOM, INS_PREP, sku_name, merma_global_pct=0.0)
    if det.empty:
        return det, float(c)
    if fabricar_concentrado and "__ins_norm__" in det.columns and det["__ins_norm__"].notna().any():
        mask = det["__ins_norm__"].astype(str).str.contains(_NESTED_INS, case=False, na=False)
        if mask.any():
            det_nested, c_nested = pd.DataFrame(), 0.0
            for _alt in _NESTED_SKU_ALIASES:
                dn, cn = costeo_insumos_por_sku(BOM, INS_PREP, _alt, merma_global_pct=0.0)
                if not dn.empty:
                    det_nested, c_nested = dn, cn
                    break
            det = det.copy()
            det.loc[mask, "Precio_base"] = float(c_nested)
            det["Consumo_base"] = pd.to_numeric(det.get("Consumo_base"), errors="coerce")
            det["Precio_base"]  = pd.to_numeric(det.get("Precio_base"),  errors="coerce")
            det["Subtotal_CLP"] = (det["Consumo_base"] * det["Precio_base"]).fillna(0.0)
            c = float(det["Subtotal_CLP"].sum())
    return det, float(c)

# ====================== LECTURA DATOS BASE (CSV) ==============================
BOM         = read_first("01_BOM.csv")
INS         = read_first("02_Precios_Insumos.csv")
PRICES_SKU  = read_first("07_Precios_SKU.csv")
PERUNIT     = read_first("09_PerUnit_Linea.csv", default=pd.DataFrame({
                "Linea":["Pan","Bolleria","Pasteleria","Cafe"],
                "Energia_gas_CLP_por_ud":[0,0,0,0],
                "Logistica_CLP_por_ud":[0,0,0,0],
              }))
OPEX_BASE   = read_first("06_OPEX_OFICIAL_sin_harinas.csv")
IVA_CFG     = read_first("11_IVA_Config.csv", default=pd.DataFrame({
                "Linea":["Pan","Bolleria","Pasteleria","Cafe","Complementarios"],
                "Afecta_IVA":[False,True,True,True,True],
                "Tasa_IVA_pct":[19,19,19,19,19],
              }))
CAPITAL     = read_first("10_Estructura_Capital.csv", default=pd.DataFrame({
                "Concepto":["Aporte_duenos_CLP","Capital_trabajo_inicial_CLP","Permisos_tramites_CLP"],
                "Valor":[20000000,2000000,500000]
              }))
MATERIAL    = read_first("04_Materiales.csv")
CAPEX_TAB   = read_first("06_consolidado.csv")
if CAPEX_TAB.empty:
    CAPEX_TAB = read_first("03_Capex.csv")

# NUEVOS: planillas OFICIALES ya unificadas
COSTOS_UNIF = read_first("COSTOS_LINEA_UNIFICADO_APP.csv")

# Complementarios: si no existe el CSV listo, lo generamos desde
# 07b_Complementarios_Detalle.csv + 02_Precios_Insumos.csv
COSTO_COMP = read_first("COSTO_COMPLEMENTARIOS_SKU.csv")
if COSTO_COMP.empty:
    DET_RAW = read_first("07b_Complementarios_Detalle.csv")
    INS_RAW = read_first("02_Precios_Insumos.csv")

    if not DET_RAW.empty and not INS_RAW.empty:
        # helpers locales (evitar colisiones con funciones globales)
        def _find_col_local(df, keys):
            keys = [k.lower() for k in keys]
            for c in df.columns:
                if any(k in str(c).lower() for k in keys):
                    return c
            return None

        def _prep_ins_local(df):
            """De 02_Precios_Insumos arma: Insumo, Unidad_base, Precio_base, __ins_norm__"""
            df = df.copy()
            col_ins = _find_col_local(df, ["insumo","ingrediente","producto","nombre"])
            col_un  = _find_col_local(df, ["unidad_base","unidad_costo_base","unidad"])
            col_pb  = _find_col_local(df, ["precio_base","precio_por_unidad","clp_base"])
            col_pp  = _find_col_local(df, ["precio_pack","precio_pack_clp","precio_total"])
            col_cp  = _find_col_local(df, ["contenido_pack","contenido"])

            if col_ins is None:
                df["Insumo"] = df.index.astype(str); col_ins = "Insumo"
            if col_un is None:
                df["Unidad_base"] = "kg"; col_un = "Unidad_base"

            df["__ins_norm__"] = df[col_ins].map(_norm_txt)

            if col_pb:
                out = df[[col_ins, col_un, col_pb, "__ins_norm__"]].rename(
                    columns={col_ins:"Insumo", col_un:"Unidad_base", col_pb:"Precio_base"}
                )
            elif col_pp and col_cp:
                tmp = df[[col_ins, col_un, col_pp, col_cp, "__ins_norm__"]].rename(
                    columns={col_ins:"Insumo", col_un:"Unidad_base",
                             col_pp:"Precio_pack_CLP", col_cp:"Contenido_pack"}
                )
                tmp["Precio_base"] = tmp["Precio_pack_CLP"].map(_num) / tmp["Contenido_pack"].map(_num).replace(0, np.nan)
                out = tmp[["Insumo","Unidad_base","Precio_base","__ins_norm__"]]
            else:
                cand = [c for c in df.columns if "precio" in str(c).lower()]
                if cand:
                    df["Precio_base"] = df[cand[0]].map(_num)
                    out = df[[col_ins, col_un, "Precio_base", "__ins_norm__"]].rename(
                        columns={col_ins:"Insumo", col_un:"Unidad_base"}
                    )
                else:
                    out = pd.DataFrame(columns=["Insumo","Unidad_base","Precio_base","__ins_norm__"])

            out["Unidad_base"] = out["Unidad_base"].map(_u)
            return out

        INS_PREP_MIN = _prep_ins_local(INS_RAW)

        c_prod = _find_col_local(DET_RAW, ["producto","sku","nombre"])
        c_ins  = _find_col_local(DET_RAW, ["insumo","ingrediente"])
        c_qty  = _find_col_local(DET_RAW, ["cantidad_por_ud","cantidad","consumo","qty"])
        c_un   = _find_col_local(DET_RAW, ["unidad","u"])
        c_mer  = _find_col_local(DET_RAW, ["merma","merma_pct","merma %"])

        if all([c_prod, c_ins, c_qty, c_un]):
            det = DET_RAW.rename(columns={
                c_prod:"Producto", c_ins:"Insumo", c_qty:"Cantidad_por_ud", c_un:"Unidad"
            })
            det["Merma_pct"] = (DET_RAW[c_mer].map(_num) if c_mer else 0.0)
            det["Merma_pct"] = det["Merma_pct"].fillna(0.0)
            det["__ins_norm__"] = det["Insumo"].map(_norm_txt)

            merged = det.merge(INS_PREP_MIN, on="__ins_norm__", how="left", suffixes=("","_INS"))

            def _consumo_base_local(row):
                cant = _num(row["Cantidad_por_ud"]) * (1.0 + float(row["Merma_pct"])/100.0)
                return convert_amount(cant, row["Unidad"], row["Unidad_base"], row["__ins_norm__"])

            merged["Consumo_base"]   = merged.apply(_consumo_base_local, axis=1)
            merged["Precio_base"]    = merged["Precio_base"].map(_num)
            merged["Costo_insumo_CLP"] = merged["Consumo_base"].astype(float) * merged["Precio_base"].astype(float)

            keep = ["Producto","Insumo","Cantidad_por_ud","Unidad","Merma_pct",
                    "Unidad_base","Consumo_base","Precio_base","Costo_insumo_CLP"]
            for c in keep:
                if c not in merged.columns:
                    merged[c] = np.nan
            COSTO_COMP = merged[keep].copy()

            # Guardamos
            try:
                COSTO_COMP.to_csv(DATA_DIR / "COSTO_COMPLEMENTARIOS_SKU.csv",
                                  index=False, encoding="utf-8-sig")
            except Exception:
                pass

# === Overrides de precios de VENTA para complementarios ======================
PRICE_COMP_OVERRIDE_RAW = {
    "Complementarios Huevo (1 unidad)": 420.0,
    "Complementarios Queso (1 kg)": 12990.0,
    "Complementarios Mermelada (1 kg)": 9490.0,
}
PRICE_COMP_OVERRIDE = {_norm_txt(k): float(v) for k, v in PRICE_COMP_OVERRIDE_RAW.items()}

# ============================================================================ #
# ============================ SIDEBAR (ESCENARIO) =============================

st.sidebar.header("Escenario de Demanda")

# 0) Días / estacionalidad
modo_dias = st.sidebar.radio(
    "Días abiertos",
    ["Todos abiertos", "Cerrar 1 día/semana", "Cerrado todos los días"],
    horizontal=True,
    key="sd_dias"
)
if modo_dias == "Todos abiertos":
    D = [30,28,31,30,31,30,31,31,30,31,30,31]
    F = [8,8,9,8,9,8,9,9,8,9,9,9]
elif modo_dias == "Cerrar 1 día/semana":
    D = [26,24,27,26,27,26,27,27,26,27,26,27]
    F = [8,8,9,8,9,8,9,9,8,9,9,9]
else:  # Cerrado todos los días
    D = [0]*12
    F = [0]*12

E_scalar = st.sidebar.slider(
    "Estacionalidad base (aplica a todas las líneas)", 0.5, 1.5, 1.0, 0.01, key="sd_Escalar"
)
E = [E_scalar]*12

# 1) Rampa
RAMPA = {
    "Peor":[0.75,0.75,0.80,0.80,0.85,0.85,0.90,0.90,0.90,0.95,0.95,1.00],
    "Medio":[0.85,0.90,0.95,1.00,1.05,1.15,1.20,1.25,1.35,1.45,1.50,1.60],
    "Mejor":[1.00,1.10,1.15,1.25,1.35,1.50,1.60,1.75,1.90,2.05,2.20,2.40]
}
esc = st.sidebar.selectbox(
    "Preset rampa",
    list(RAMPA.keys()) + ["Geométrica"],
    index=1,  # ← default: "Medio"
    key="sd_rampa"
)
if esc == "Geométrica":
    c1, c2 = st.sidebar.columns(2)
    R1 = c1.number_input("R1", 0.10, 5.0, 0.85, 0.05, key="sd_R1")
    R12 = c2.number_input("R12", 0.10, 10.0, 1.60, 0.05, key="sd_R12")
    R = rampa_geom(R1, R12, 12)
else:
    R = RAMPA[esc]

# 2) Weekday / Weekend por línea
st.sidebar.subheader("Mix weekday / weekend por línea")
c = st.sidebar.columns(4)
wd_pan = c[0].number_input("Pan wd",        0.1, 5.0, 1.00, 0.05, key="wd_pan")
wd_bol = c[1].number_input("Bollería wd",   0.1, 5.0, 1.00, 0.05, key="wd_bol")
wd_pas = c[2].number_input("Pastelería wd", 0.1, 5.0, 1.00, 0.05, key="wd_pas")
wd_caf = c[3].number_input("Café wd",       0.1, 5.0, 1.10, 0.05, key="wd_caf")

c2 = st.sidebar.columns(4)
we_pan = c2[0].number_input("Pan we",        0.1, 5.0, 1.30, 0.05, key="we_pan")
we_bol = c2[1].number_input("Bollería we",   0.1, 5.0, 1.30, 0.05, key="we_bol")
we_pas = c2[2].number_input("Pastelería we", 0.1, 5.0, 1.30, 0.05, key="we_pas")
we_caf = c2[3].number_input("Café we",       0.1, 5.0, 1.10, 0.05, key="we_caf")

S_pan = s_factor_vec(wd_pan, we_pan, D, F)
S_bol = s_factor_vec(wd_bol, we_bol, D, F)
S_pas = s_factor_vec(wd_pas, we_pas, D, F)
S_caf = s_factor_vec(wd_caf, we_caf, D, F)

# 3) Variación manual (placeholder = 1.0)
def _vec1(): return [1.0]*12
V_pan_vec = _vec1(); V_bol_vec=_vec1(); V_pas_vec=_vec1(); V_caf_vec=_vec1()

# 4) Uplifts / Canales
st.sidebar.subheader("Uplifts / Canales")
delivery_ini = st.sidebar.number_input("Inicio Delivery (mes, 0=nunca)", 0, 12, 0, 1, key="sd_delivery_ini")
delivery_upl = st.sidebar.number_input("Uplift Delivery", 1.00, 2.50, 1.00, 0.01, key="sd_delivery_upl")  # ← 1.00
phi_cafe     = st.sidebar.number_input("Imán Café (φ)", 1.00, 1.50, 1.00, 0.01, key="sd_phi_cafe")

def uplift_vec(start, upl):
    if start <= 0: return [1.0]*12
    return [1.0]*(start-1) + [upl]*(12-start+1)

C_pan = [1.0]*12
C_bol = uplift_vec(delivery_ini, delivery_upl)
C_pas = uplift_vec(delivery_ini, delivery_upl)
C_caf = [1.0]*12
if phi_cafe > 1.0:
    C_bol = [x*phi_cafe for x in C_bol]
    C_pas = [x*phi_cafe for x in C_pas]

# 5) Bases diarias por línea (día neutro)
st.sidebar.subheader("Bases diarias por línea (día neutro)")
c = st.sidebar.columns(4)
base_pan = c[0].number_input("Pan (piezas/día)",     0, 10000, 450, 10, key="sd_base_pan")  # ← 450
base_bol = c[1].number_input("Bollería (ud/día)",    0, 10000,  75,  5, key="sd_base_bol")
base_pas = c[2].number_input("Pastelería (ud/día)",  0, 10000,  15,  1, key="sd_base_pas")
base_caf = c[3].number_input("Café (tazas/día)",     0, 10000,  35,  5, key="sd_base_caf")

# Opciones de comportamiento
redondear_pan_kg = st.sidebar.checkbox("Redondear Pan (kg) a entero", True, key="sd_round_pan_kg")
comp_seasonal    = st.sidebar.checkbox("Complementarios siguen estacionalidad/mix promedio", False, key="sd_comp_seasonal")

# === SKUs por producto (PRO) en el modelo global ===
st.sidebar.header("SKUs por producto (PRO)")
use_pro_global = st.sidebar.checkbox(
    "Incluir SKUs PRO en TODO el modelo", True, key="pro_use_global"
)
fabricar_conc = st.sidebar.checkbox(
    "Fabricar Concentrado in-house", True, key="pro_fabricar_conc"
)
st.sidebar.caption("Los precios y bases diarias de cada SKU se pueden ajustar en la pestaña PRO o aquí si se agregan inputs.")

with st.sidebar.expander("Ajustes rápidos PRO (precio y base/día)", expanded=False):

    # Función de sincronización canónica (tab <-> sidebar)
    def _sync_from_sidebar(i: int):
        p_key = f"sb_pro_p_{i}"
        b_key = f"sb_pro_b_{i}"
        if p_key in st.session_state:
            st.session_state[f"pro_p_{i}"] = int(st.session_state[p_key])
        if b_key in st.session_state:
            st.session_state[f"pro_b_{i}"] = int(st.session_state[b_key])

    for i, item in enumerate(PRO_SKUS):
        sku = item["SKU"]
        canon_p = int(st.session_state.get(f"pro_p_{i}", item["precio_sugerido"]))
        canon_b = int(st.session_state.get(f"pro_b_{i}", item["base_dia"]))

        sb_p_key = f"sb_pro_p_{i}"
        sb_b_key = f"sb_pro_b_{i}"

        if sb_p_key not in st.session_state or int(st.session_state[sb_p_key]) != canon_p:
            st.session_state[sb_p_key] = canon_p
        if sb_b_key not in st.session_state or int(st.session_state[sb_b_key]) != canon_b:
            st.session_state[sb_b_key] = canon_b

        st.caption(f"**{sku}**")

        st.number_input(
            f"Precio – {sku}",
            min_value=0, max_value=50_000, step=100,
            key=sb_p_key,
            on_change=lambda i=i: _sync_from_sidebar(i),
        )
        st.number_input(
            f"Base/día – {sku}",
            min_value=0, max_value=10_000, step=1,
            key=sb_b_key,
            on_change=lambda i=i: _sync_from_sidebar(i),
        )

        # Garantiza sincronización también en cada rerun (por si no saltó on_change)
        _sync_from_sidebar(i)

# === Complementarios (simple: suma directa por SKU) ==========================
st.sidebar.header("Complementarios")

def _find_col(df, keys):
    keys = [k.lower() for k in keys]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None

# Defaults de unidades/día para 3 SKUs frecuentes
DEFAULT_COMP_UNITS = {
    _norm_txt("Complementarios Huevo (1 unidad)"): 16,
    _norm_txt("Complementarios Mermelada (1 kg)"): 2,
    _norm_txt("Complementarios Queso (1 kg)"): 1,
}

st.session_state.setdefault("comp_ventas_dia_override", 0.0)
st.session_state.setdefault("comp_cogs_dia_override", 0.0)
st.session_state.setdefault("comp_margen_dia_override", 0.0)

if COSTO_COMP.empty:
    st.sidebar.info("No se encontró 'data/COSTO_COMPLEMENTARIOS_SKU.csv'.")
    st.session_state["comp_ventas_dia_override"] = 0.0
    st.session_state["comp_cogs_dia_override"]   = 0.0
    st.session_state["comp_margen_dia_override"] = 0.0
else:
    col_prod  = _find_col(COSTO_COMP, ["producto","sku","nombre"])
    col_cost  = _find_col(COSTO_COMP, ["costo_insumo","costo_unit","costo"])
    col_price = _find_col(COSTO_COMP, ["precio_venta","precio","pvp"])

    # Costo unitario por SKU (suma de insumos)
    costo_sum = (
        COSTO_COMP.groupby(col_prod, as_index=True)[col_cost]
                  .sum().map(_num).fillna(0.0)
    )

    # Precio por SKU (con overrides)
    price_map = {}
    if col_price:
        price_map = (COSTO_COMP[[col_prod, col_price]]
                     .dropna(subset=[col_price])
                     .drop_duplicates(subset=[col_prod])
                     .set_index(col_prod)[col_price]
                     .map(_num).to_dict())
    if len(price_map) == 0 and not PRICES_SKU.empty:
        c_prod2 = _find_col(PRICES_SKU, ["producto","sku","nombre"])
        c_p2    = _find_col(PRICES_SKU, ["precio","pvp"])
        if c_prod2 and c_p2:
            price_map = (PRICES_SKU[[c_prod2, c_p2]]
                         .dropna(subset=[c_p2])
                         .drop_duplicates(subset=[c_prod2])
                         .set_index(c_prod2)[c_p2]
                         .map(_num).to_dict())
    if len(price_map) == 0:
        price_map = {k: float(v)*1.5 for k, v in costo_sum.items()}  # fallback

    price_map_norm = { _norm_txt(k): float(v) for k, v in price_map.items() }
    price_map_norm.update(PRICE_COMP_OVERRIDE)
    price_map = price_map_norm

    skus_all = list(costo_sum.index)
    sel = st.sidebar.multiselect(
        "SKUs complementarios",
        options=skus_all,
        default=skus_all[:min(3, len(skus_all))],
        key="y_sel"
    )

    y_ventas = 0.0
    y_cogs   = 0.0
    for sku in sel:
        uds_default = int(DEFAULT_COMP_UNITS.get(_norm_txt(sku), 0))
        uds = st.sidebar.number_input(
            f"{sku} (ud/día)", min_value=0, max_value=5000, value=uds_default, step=1, key=f"y_ud_{sku}"
        )
        p = float(price_map.get(_norm_txt(sku), 0.0))
        c = float(costo_sum.get(sku, 0.0))
        y_ventas += uds * p
        y_cogs   += uds * c

    st.sidebar.caption(
        f"Unidades líneas (base día): Pan {base_pan} • Bollería {base_bol} • Pastelería {base_pas} • Café {base_caf}"
    )

    st.session_state["comp_ventas_dia_override"] = float(y_ventas)
    st.session_state["comp_cogs_dia_override"]   = float(y_cogs)
    st.session_state["comp_margen_dia_override"] = float(y_ventas - y_cogs)

# 6) Costos variables / Per-unit & %  (título eliminado a pedido)
tpv_pct = st.sidebar.number_input("% TPV sobre ventas brutas", 0.0, 5.0, 0.0, 0.1, key="sd_tpv") / 100.0  # ← 0.00
mkt_pct = st.sidebar.number_input("% Marketing sobre ventas brutas", 0.0, 10.0, 0.0, 0.1, key="sd_mkt") / 100.0

costo_cafe_unit = 1000  # fallback solo si no hay BOM ni planilla oficial

# ============================ PREPARACIÓN COSTOS ==============================
def preparar_insumos(INS: pd.DataFrame) -> pd.DataFrame:
    """Devuelve columnas limpias: Insumo, Unidad_base, Precio_base, __ins_norm__"""
    if INS.empty:
        return pd.DataFrame(columns=["Insumo","Unidad_base","Precio_base","__ins_norm__"])
    df = INS.copy()

    col_ins = next((c for c in ["Insumo","insumo","Nombre","Producto"] if c in df.columns), None)
    col_un  = next((c for c in ["Unidad_base","Unidad_costo_base","unidad","Unidad"] if c in df.columns), None)
    col_pb  = next((c for c in ["Precio_base","precio_base","Precio_por_unidad","CLP_base"] if c in df.columns), None)
    col_pp  = next((c for c in ["Precio_pack_CLP","precio_pack_clp","Precio_total","Precio_pack"] if c in df.columns), None)
    col_cp  = next((c for c in ["Contenido_pack","contenido_pack","Contenido"] if c in df.columns), None)

    if col_ins is None:
        df["Insumo"] = df.index.astype(str); col_ins = "Insumo"
    if col_un is None:
        df["Unidad_base"] = "kg"; col_un = "Unidad_base"

    df["__ins_norm__"] = df[col_ins].map(_norm_txt)

    if col_pb is not None:
        out = df[[col_ins, col_un, col_pb, "__ins_norm__"]].rename(
            columns={col_ins:"Insumo", col_un:"Unidad_base", col_pb:"Precio_base"}
        )
    elif (col_pp is not None) and (col_cp is not None):
        tmp = df[[col_ins, col_un, col_pp, col_cp, "__ins_norm__"]].rename(
            columns={col_ins:"Insumo", col_un:"Unidad_base", col_pp:"Precio_pack_CLP", col_cp:"Contenido_pack"}
        )
        tmp["Precio_base"] = tmp["Precio_pack_CLP"].map(_num) / tmp["Contenido_pack"].map(_num).replace(0, np.nan)
        out = tmp[["Insumo","Unidad_base","Precio_base","__ins_norm__"]]
    else:
        cand = [c for c in df.columns if "precio" in c.lower()]
        if cand:
            df["Precio_base"] = df[cand[0]].map(_num)
            out = df[[col_ins, col_un, "Precio_base", "__ins_norm__"]].rename(
                columns={col_ins:"Insumo", col_un:"Unidad_base"}
            )
        else:
            out = pd.DataFrame(columns=["Insumo","Unidad_base","Precio_base","__ins_norm__"])

    out = out.copy()
    out["Unidad_base"] = out["Unidad_base"].map(_u)
    return out

INS_PREP = preparar_insumos(INS)

# ============================ COSTEO UNITARIO POR SKU =========================
SKU_BOM = {
    "Pan":        "Marraqueta 100g (econ)",       # se convertirá a costo por kg (×10)
    "Bolleria":   "Croissant 70g (econ)",
    "Pasteleria": "Trozo Pasteleria 100g (econ)",
    "Cafe":       "Cafe 20g (econ)",              # si no existe, usamos costo_cafe_unit
}

def costeo_insumos_por_sku(bom_df, ins_df, sku_sel, merma_global_pct=0.0):
    """Retorna: detalle (DataFrame) y costo_unit_insumos (float) usando BOM+02_Precios."""
    if bom_df.empty or ins_df.empty:
        return pd.DataFrame(), 0.0

    col_sku  = next((c for c in ["SKU","sku","Producto"] if c in bom_df.columns), None)
    col_ins  = next((c for c in ["Insumo","insumo","Ingrediente"] if c in bom_df.columns), None)
    col_qty  = next((c for c in ["Consumo_BOM","Cantidad","cantidad","Qty","Cantidad_por_ud","Cantidad_por_unidad","Cantidad_por_ud"] if c in bom_df.columns), None)
    col_un   = next((c for c in ["Unidad_BOM","Unidad","unidad"] if c in bom_df.columns), None)
    if col_sku is None or col_ins is None or col_qty is None or col_un is None:
        return pd.DataFrame(), 0.0

    df = bom_df[bom_df[col_sku]==sku_sel].copy()
    if df.empty:
        return pd.DataFrame(), 0.0

    df.rename(columns={col_ins:"Insumo", col_qty:"Consumo_BOM", col_un:"Unidad_BOM"}, inplace=True)
    df["__ins_norm__"] = df["Insumo"].map(_norm_txt)

    merged = df.merge(ins_df, on="__ins_norm__", how="left", suffixes=("_BOM","_COST"))
    if "Insumo_BOM" in merged.columns:
        merged["Insumo"] = merged["Insumo_BOM"]

    merged["Consumo_ajustado"] = merged["Consumo_BOM"].map(_num) * (1 + float(merma_global_pct)/100.0)

    consumo_base = []
    subtot = []
    for _, r in merged.iterrows():
        u_from = r.get("Unidad_BOM")
        u_to = r.get("Unidad_base")
        cant = r.get("Consumo_ajustado")
        ins_norm = r.get("__ins_norm__", "")
        conv = convert_amount(cant, u_from, u_to, ins_norm)
        consumo_base.append(conv)
        pb = r.get("Precio_base", np.nan)
        subtot.append(float(conv)*float(pb) if pd.notna(conv) and pd.notna(pb) else np.nan)

    merged["Consumo_base"] = consumo_base
    merged["Subtotal_CLP"] = pd.Series(subtot, dtype="float64").fillna(0.0)

    cols_show = ["Insumo","Consumo_BOM","Unidad_BOM","Consumo_ajustado","Unidad_base","Consumo_base","Precio_base","Subtotal_CLP"]
    for c in cols_show:
        if c not in merged.columns: merged[c]=np.nan
    detalle = merged[cols_show].copy()

    costo_unit = float(pd.to_numeric(merged["Subtotal_CLP"], errors="coerce").fillna(0.0).sum())
    return detalle, costo_unit

# ============================ MODELO CENTRAL ==================================
LINE_ALIASES = {"pan":"Pan","panaderia":"Pan","bolleria":"Bolleria","pasteleria":"Pasteleria","cafe":"Cafe"}
def map_line_name(x):
    return LINE_ALIASES.get(_norm_txt(x), x)

# Producto representante por línea (planilla oficial)
REP_OFICIAL = {
    "Pan (kg)":  "PAN — Marraqueta 100g (econ)",   # 100 g → ×10 para 1 kg
    "Bolleria":  "CROISSANT — 70g (econ)",
    "Pasteleria":"TROZO — Mil Hojas 1/8 (A PDF)",
    "Cafe":      "CAFÉ — Taza estándar",
}

def cunit_insumos_oficial(linea: str) -> float:
    if COSTOS_UNIF.empty or "Producto" not in COSTOS_UNIF.columns:
        return np.nan
    prod = REP_OFICIAL.get(linea, None)
    if not prod:
        return np.nan
    det = COSTOS_UNIF.loc[COSTOS_UNIF["Producto"] == prod].copy()
    if det.empty or "Costo_insumo_CLP" not in det.columns:
        return np.nan
    costo_ud = float(pd.to_numeric(det["Costo_insumo_CLP"], errors="coerce").fillna(0.0).sum())
    if linea == "Pan (kg)":
        costo_ud *= PAN_PIEZAS_POR_KG  # 10 piezas de 100 g = 1 kg
    return costo_ud

def compute_model():
    global sku_pro_df, consumo_concentrado_df
    sku_pro_df = pd.DataFrame()
    consumo_concentrado_df = pd.DataFrame()
    Cap = [np.inf]*12
    def vec_units(b, Dv, Rv, Ev, Sv, Vv, Cv, Capv):
        arr = np.array(b)*np.array(Dv)*np.array(Rv)*np.array(Ev)*np.array(Sv)*np.array(Vv)*np.array(Cv)
        return np.minimum(arr, np.array(Capv)).round(0).astype(int).tolist()

    U_pan_piezas = vec_units(base_pan, D, R, E, S_pan, V_pan_vec, C_pan, Cap)
    U_bol        = vec_units(base_bol, D, R, E, S_bol, V_bol_vec, C_bol, Cap)
    U_pas        = vec_units(base_pas, D, R, E, S_pas, V_pas_vec, C_pas, Cap)
    U_caf        = vec_units(base_caf, D, R, E, S_caf, V_caf_vec, C_caf, Cap)

    # Conversión Pan a kilos
    _pan_kg_arr = np.array(U_pan_piezas, dtype=float) / PAN_PIEZAS_POR_KG
    if redondear_pan_kg:
        U_pan = _pan_kg_arr.round(0).astype(int).tolist()
    else:
        U_pan = _pan_kg_arr.tolist()

    U_df = pd.DataFrame({
        "Mes":MESES,
        "Pan (kg)": U_pan,
        "Bolleria": U_bol,
        "Pasteleria": U_pas,
        "Cafe": U_caf
    })
    U_df["Total"] = U_df[["Pan (kg)","Bolleria","Pasteleria","Cafe"]].sum(axis=1)
    pro_units = pd.DataFrame()

    # --- PRO SKUs GLOBAL (opcional) -----------------------------------------
    sku_pro_df = pd.DataFrame()
    if st.session_state.get("pro_use_global", True):
        cfg_rows = []
        for i, item in enumerate(PRO_SKUS):
            p = float(st.session_state.get(f"pro_p_{i}", item["precio_sugerido"]))
            b = int(st.session_state.get(f"pro_b_{i}", item["base_dia"]))
            cfg_rows.append({"Linea": item["Linea"], "SKU": item["SKU"], "Precio": p, "Base_dia": b})
        pro_cfg = pd.DataFrame(cfg_rows)

        def _S_for(linea):
            return {"Pan (kg)": S_pan, "Bolleria": S_bol, "Pasteleria": S_pas, "Cafe": S_caf}.get(linea, [1.0]*12)

        rows = []
        for _, r in pro_cfg.iterrows():
            Sv = _S_for(r["Linea"])
            U_mes = np.array([r["Base_dia"]]*12, dtype=float) * np.array(D)*np.array(R)*np.array(E)*np.array(Sv)
            for j, mes in enumerate(MESES):
                uds = float(np.round(U_mes[j], 0))
                rows.append({
                    "Mes": mes, "Linea": r["Linea"], "SKU": r["SKU"],
                    "Unidades": uds, "Precio": r["Precio"], "Venta_mes": uds * r["Precio"]
                })
        sku_pro_df = pd.DataFrame(rows)

        if not sku_pro_df.empty:
            add_units = (
                sku_pro_df.groupby(["Mes","Linea"])["Unidades"].sum().unstack(fill_value=0.0)
            )
            index_order = U_df["Mes"].tolist()
            add_units = add_units.reindex(index_order).fillna(0.0)
            pro_units = add_units.copy()
            for col in add_units.columns:
                if col in U_df.columns:
                    aligned = add_units[col].values
                    U_df[col] = (U_df[col].astype(float) + aligned).round(0).astype(int)
            U_df["Total"] = U_df[["Pan (kg)","Bolleria","Pasteleria","Cafe"]].sum(axis=1)

    # 2) Costos unitarios de insumos (preferencia: planilla OFICIAL; fallback: BOM)
    detalles_costeo = {}
    cunit_ins = {}

    def _costo_linea(lin, sku_bom_key=None):
        c_of = cunit_insumos_oficial(lin)
        if not np.isnan(c_of):
            prod = REP_OFICIAL.get(lin)
            det = pd.DataFrame()
            if prod and not COSTOS_UNIF.empty:
                det = COSTOS_UNIF.loc[COSTOS_UNIF["Producto"]==prod].copy()
                if lin=="Pan (kg)":
                    det = det.copy()
                    det["__nota__"] = "Producto base 100 g – total por kg mostrado (×10)"
            return float(c_of), det

        if sku_bom_key is not None:
            det_bom, c_bom = costeo_insumos_por_sku(BOM, INS_PREP, SKU_BOM[sku_bom_key], 0.0)
            if lin=="Pan (kg)":
                c_bom *= PAN_PIEZAS_POR_KG
            return float(c_bom), det_bom

        if lin=="Cafe":
            return float(costo_cafe_unit), pd.DataFrame()

        return 0.0, pd.DataFrame()

    c_pan, det_pan = _costo_linea("Pan (kg)", "Pan")
    c_bol, det_bol = _costo_linea("Bolleria", "Bolleria")
    c_pas, det_pas = _costo_linea("Pasteleria", "Pasteleria")
    c_caf, det_caf = _costo_linea("Cafe", "Cafe")

    cunit_ins["Pan (kg)"]  = c_pan
    cunit_ins["Bolleria"]  = c_bol
    cunit_ins["Pasteleria"]= c_pas
    cunit_ins["Cafe"]      = c_caf

    detalles_costeo["Pan (kg)"]  = det_pan
    detalles_costeo["Bolleria"]  = det_bol
    detalles_costeo["Pasteleria"]= det_pas
    detalles_costeo["Cafe"]      = det_caf

    # 3) Per-unit (energía/logística) por línea
    per_unit_map = {}
    if not PERUNIT.empty and "Linea" in PERUNIT.columns:
        tmp = PERUNIT.copy()
        tmp["Linea"] = tmp["Linea"].map(map_line_name)
        for c in ["Energia_gas_CLP_por_ud","Logistica_CLP_por_ud"]:
            if c in tmp.columns: tmp[c] = tmp[c].map(_num).fillna(0.0)
            else: tmp[c]=0.0
        tmp["per_unit_total"] = tmp["Energia_gas_CLP_por_ud"] + tmp["Logistica_CLP_por_ud"]
        per_unit_map = tmp.set_index("Linea")["per_unit_total"].to_dict()
    else:
        per_unit_map = {"Pan":0.0,"Bolleria":0.0,"Pasteleria":0.0,"Cafe":0.0}

    if not sku_pro_df.empty:
        sku_pro_df = sku_pro_df.copy()
        sku_pro_df["unidades"] = pd.to_numeric(
            sku_pro_df.get("Unidades", sku_pro_df.get("unidades", 0)),
            errors="coerce"
        ).fillna(0.0)
        sku_pro_df["ingreso"] = pd.to_numeric(
            sku_pro_df.get("Venta_mes", sku_pro_df.get("ingreso", 0.0)),
            errors="coerce"
        ).fillna(0.0)

        def _pro_unit_cost(linea: str) -> float:
            lin_key_per = str(linea).replace(" (kg)", "")
            return float(cunit_ins.get(linea, 0.0)) + float(per_unit_map.get(lin_key_per, 0.0))

        sku_pro_df["cogs"] = sku_pro_df["unidades"] * sku_pro_df["Linea"].map(_pro_unit_cost).fillna(0.0)
        sku_pro_df["margen"] = sku_pro_df["ingreso"] - sku_pro_df["cogs"]

        if st.session_state.get("pro_fabricar_conc", False):
            consumo_rows = []
            for sku_name in sku_pro_df["SKU"].dropna().unique():
                det_sku, _ = costeo_sku_pro(sku_name, fabricar_concentrado=True)
                if det_sku.empty or "__ins_norm__" not in det_sku.columns:
                    continue
                mask = det_sku["__ins_norm__"].astype(str).str.contains(_NESTED_INS, case=False, na=False)
                if not mask.any():
                    continue
                consumo_unit = pd.to_numeric(det_sku.loc[mask, "Consumo_base"], errors="coerce").fillna(0.0).sum()
                subtotal = pd.to_numeric(det_sku.loc[mask, "Subtotal_CLP"], errors="coerce").fillna(0.0).sum()
                precio_unit = (subtotal / consumo_unit) if consumo_unit else 0.0
                for _, row in sku_pro_df.loc[sku_pro_df["SKU"] == sku_name].iterrows():
                    unidades = float(row.get("Unidades", row.get("unidades", 0.0)))
                    consumo_total = consumo_unit * unidades
                    costo_total = consumo_total * precio_unit
                    consumo_rows.append({
                        "Mes": row.get("Mes"),
                        "SKU": sku_name,
                        "Consumo_concentrado": consumo_total,
                        "Costo_concentrado": costo_total,
                    })
            consumo_concentrado_df = pd.DataFrame(consumo_rows)
        else:
            consumo_concentrado_df = pd.DataFrame()

    # 4) Precios de venta (1 representante por línea)
    PRICE = {"Pan (kg)":2200.0, "Bolleria":1200.0, "Pasteleria":3500.0, "Cafe":2500.0}

    # 5) Ventas por línea (representantes)
    rows = []
    for i, mes in enumerate(MESES):
        def _u_rep(col: str) -> float:
            base_val = float(U_df.loc[i, col])
            if (
                isinstance(pro_units, pd.DataFrame)
                and not pro_units.empty
                and col in pro_units.columns
                and mes in pro_units.index
            ):
                return max(base_val - float(pro_units.loc[mes, col]), 0.0)
            return base_val

        pan_rep = _u_rep("Pan (kg)")
        bol_rep = _u_rep("Bolleria")
        pas_rep = _u_rep("Pasteleria")
        caf_rep = _u_rep("Cafe")

        rows.append({
            "Mes": mes,
            "Linea": "Pan (kg)",
            "SKU": "Pan (kg)",
            "Unidades": pan_rep,
            "Precio": PRICE["Pan (kg)"]
        })
        rows.append({
            "Mes": mes,
            "Linea": "Bolleria",
            "SKU": "Croissant",
            "Unidades": bol_rep,
            "Precio": PRICE["Bolleria"]
        })
        rows.append({
            "Mes": mes,
            "Linea": "Pasteleria",
            "SKU": "Trozo Pastel",
            "Unidades": pas_rep,
            "Precio": PRICE["Pasteleria"]
        })
        rows.append({
            "Mes": mes,
            "Linea": "Cafe",
            "SKU": "Cafe",
            "Unidades": caf_rep,
            "Precio": PRICE["Cafe"]
        })

    sku_df = pd.DataFrame(rows)

    if not sku_pro_df.empty:
        sku_df = pd.concat([sku_df, sku_pro_df], ignore_index=True)

    if not sku_df.empty:
        sku_df["Venta_mes"] = sku_df["Unidades"] * sku_df["Precio"]

    # Complementarios: ventas y COGS por mes desde overrides (con opción estacional)
    ov_v = st.session_state.get("comp_ventas_dia_override", None)
    ov_c = st.session_state.get("comp_cogs_dia_override", None)

    base_factor = np.array(D, dtype=float) * np.array(R, dtype=float)
    if comp_seasonal:
        S_prom = (np.array(S_bol, dtype=float) + np.array(S_pas, dtype=float) + np.array(S_caf, dtype=float)) / 3.0
        base_factor = base_factor * np.array(E, dtype=float) * S_prom

    if ov_v is not None:
        ventas_compl_mes = pd.Series(base_factor * float(ov_v), index=MESES, dtype=float)
        cogs_compl_mes   = pd.Series(base_factor * float(ov_c or 0.0), index=MESES, dtype=float)
    else:
        ventas_compl_mes = pd.Series(base_factor * 0.0, index=MESES, dtype=float)
        cogs_compl_mes   = pd.Series(base_factor * 0.0, index=MESES, dtype=float)

    # 6) COGS por mes (líneas + complementarios)
    det_rows=[]
    for (mes, linea), g in sku_df.groupby(["Mes","Linea"]):
        lin_key_per = linea.replace(" (kg)","")
        cu = float(cunit_ins.get(linea, 0.0)) + float(per_unit_map.get(lin_key_per, 0.0))
        uds = float(g["Unidades"].sum())
        det_rows.append({"Linea": linea, "Mes": mes, "Unidades": uds, "Costo_unit_CLP": cu, "COGS": cu * uds})
    cogs_detalle = pd.DataFrame(det_rows)

    cogs_mes_sin_compl = cogs_detalle.groupby("Mes")["COGS"].sum().reindex(MESES).fillna(0.0) if not cogs_detalle.empty else pd.Series(0.0, index=MESES)
    cogs_mes_total = cogs_mes_sin_compl + cogs_compl_mes

    compl_rows = [
        {
            "Linea": "Complementarios",
            "Mes": mes,
            "Unidades": np.nan,
            "Costo_unit_CLP": np.nan,
            "COGS": float(cogs_compl_mes.loc[mes])
        }
        for mes in MESES
    ]
    cogs_detalle_full = pd.concat([cogs_detalle, pd.DataFrame(compl_rows)], ignore_index=True)

    # 7) Ventas brutas
    ventas_brutas_lineas = sku_df.groupby("Mes")["Venta_mes"].sum().reindex(MESES).fillna(0.0) if not sku_df.empty else pd.Series(0.0, index=MESES)
    ventas_brutas = ventas_brutas_lineas + ventas_compl_mes

    # 8) OPEX (target + variables)
    base_ref = OPEX_BASE["CLP_mes"].map(_num).sum() if ("CLP_mes" in OPEX_BASE.columns) else 0.0
    target = st.session_state.get("__opex_target__", int(base_ref) if base_ref>0 else 0)
    target = st.sidebar.number_input("OPEX M2→ base target", 0, 30_000_000, int(target), 50_000, key="sd_opex_target")
    st.session_state["__opex_target__"] = target
    extra_m1 = st.sidebar.number_input("Extra apertura M1", 0, 20_000_000, 2_500_000, 50_000, key="sd_opex_extra_m1")

    opex_fijo_vec = [target]*12
    if len(opex_fijo_vec)>0:
        opex_fijo_vec[0] = opex_fijo_vec[0] + extra_m1
    opex_fijo = pd.Series(opex_fijo_vec, index=MESES, dtype=float)

    opex_tpv = ventas_brutas * tpv_pct
    opex_mkt = ventas_brutas * mkt_pct
    OPEX_mes = opex_fijo + opex_tpv + opex_mkt

    # 9) IVA (débito por líneas afectas; complementarios según IVA_CFG)
    # ---- NORMALIZACIÓN ROBUSTA DE IVA_CFG ----
    def _to_bool_any(x):
        s = str(x).strip().lower()
        return s in ("true","1","si","sí","yes","y","afecto","afecta","v","t")

    def _to_float_any(x, default=None):
        v = _num(x)
        return float(v) if pd.notna(v) and np.isfinite(v) else (float(default) if default is not None else np.nan)

    lineas_cfg = list(IVA_CFG.get("Linea", []))
    afecta_series = pd.Series(list(IVA_CFG.get("Afecta_IVA", []))).map(_to_bool_any) if "Afecta_IVA" in IVA_CFG.columns else pd.Series([False]*len(lineas_cfg))
    tasa_series   = pd.Series(list(IVA_CFG.get("Tasa_IVA_pct", []))).map(lambda x: _to_float_any(x, 19.0)) if "Tasa_IVA_pct" in IVA_CFG.columns else pd.Series([19.0]*len(lineas_cfg))

    afecta_map = dict(zip(lineas_cfg, afecta_series))
    tasa_map   = dict(zip(lineas_cfg, tasa_series))

    ventas_linea = sku_df.groupby(["Mes","Linea"])["Venta_mes"].sum().unstack(fill_value=0.0).reindex(MESES).fillna(0.0) if not sku_df.empty else pd.DataFrame(0.0, index=MESES, columns=["Pan (kg)","Bolleria","Pasteleria","Cafe"])
    ventas_linea["Complementarios"] = ventas_compl_mes

    iva_debito = pd.Series(0.0, index=MESES)
    ventas_netas = pd.Series(0.0, index=MESES)
    for lin in ventas_linea.columns:
        v = ventas_linea[lin]
        lin_map = "Pan" if lin.startswith("Pan") else lin
        if bool(afecta_map.get(lin_map, False)):
            tasa = float(tasa_map.get(lin_map, 19.0)) / 100.0
            net = v / (1.0 + tasa) if (1.0 + tasa) != 0 else v
            iva_debito += (v - net)
            ventas_netas += net
        else:
            ventas_netas += v

    # Crédito de IVA (compras)
    pct_opex_afecto = st.sidebar.slider("% OPEX afecto a IVA crédito", 0, 100, 30, 1, key="sd_pct_iva_opex")/100.0
    pct_cogs_afecto = st.sidebar.slider("% COGS afecto a IVA crédito", 0, 100, 0, 1, key="sd_pct_iva_cogs")/100.0
    tasa_general = st.sidebar.number_input("Tasa IVA general (%)", 0, 30, 19, 1, key="sd_tasa_iva")/100.0

    iva_credito_opex = (OPEX_mes * pct_opex_afecto) * (tasa_general/(1+tasa_general))
    iva_credito_cogs = (cogs_mes_total * pct_cogs_afecto) * (tasa_general/(1+tasa_general))
    iva_credito_total = iva_credito_opex + iva_credito_cogs

    # OPEX / COGS netos de IVA
    OPEX_neto = OPEX_mes - iva_credito_opex
    COGS_neto = cogs_mes_total - iva_credito_cogs
    iva_pagar = iva_debito - iva_credito_total

    # EBITDA (bruto)
    EBITDA = ventas_brutas - cogs_mes_total - OPEX_mes

    # 11) Capex & Materiales
    if not CAPEX_TAB.empty and {"Cantidad","Precio_unit_CLP"}.issubset(CAPEX_TAB.columns):
        capex_total = (CAPEX_TAB["Cantidad"].map(_num)*CAPEX_TAB["Precio_unit_CLP"].map(_num)).sum()
    else:
        capex_total = 0.0

    mat_total = 0.0
    if not MATERIAL.empty:
        if "CLP_total" in MATERIAL.columns:
            mat_total = MATERIAL["CLP_total"].map(_num).sum()
        elif "CLP_mes" in MATERIAL.columns:
            mat_total = MATERIAL["CLP_mes"].map(_num).sum() * 12.0

    return dict(
        U_df=U_df,
        detalles_costeo=detalles_costeo,
        cost_unit_ins=cunit_ins,
        per_unit_map=per_unit_map,
        prices=PRICE,
        sku_df=sku_df,
        ventas_compl_mes=ventas_compl_mes,
        ventas_brutas=ventas_brutas,
        ventas_linea=ventas_linea,

        COGS_detalle=cogs_detalle,
        COGS_detalle_full=cogs_detalle_full,
        COGS_mes_sin_compl=cogs_mes_sin_compl,
        COGS_compl_mes=cogs_compl_mes,
        COGS_mes_total=cogs_mes_total,
        COGS_neto=COGS_neto,

        OPEX_mes=OPEX_mes,
        OPEX_neto=OPEX_neto,

        IVA_debito=iva_debito,
        IVA_credito=iva_credito_total,
        IVA_credito_OPEX=iva_credito_opex,
        IVA_credito_COGS=iva_credito_cogs,
        IVA_pagar=iva_pagar,

        ventas_netas=ventas_netas,
        EBITDA=EBITDA,
        capex_total=float(capex_total),
        materiales_total=float(mat_total),
        sku_pro_df=sku_pro_df,
        consumo_concentrado_df=consumo_concentrado_df,
    )



    
# ============================ EJECUTAR MODELO =================================
MODEL = compute_model()

# ============================ AUDITORÍA RÁPIDA (sidebar) ======================
with st.sidebar.expander("🔎 Auditoría numérica", expanded=True):
    tol = 1e-6
    v_check = (MODEL["ventas_brutas"] - (MODEL["ventas_netas"] + MODEL["IVA_debito"])).abs().max()
    st.write(f"- Ventas_brutas = Ventas_netas + IVA_débito: **{'OK' if v_check < tol else f'FAIL (max {v_check:,.2f})'}**")

    i_check = (MODEL["IVA_pagar"] - (MODEL["IVA_debito"] - MODEL["IVA_credito"])).abs().max()
    st.write(f"- IVA_pagar = IVA_débito − IVA_crédito_TOTAL: **{'OK' if i_check < tol else f'FAIL (max {i_check:,.2f})'}**")

    o_check = (MODEL["OPEX_neto"] - (MODEL["OPEX_mes"] - MODEL["IVA_credito_OPEX"])).abs().max()
    st.write(f"- OPEX_neto = OPEX_bruto − IVA_crédito_OPEX: **{'OK' if o_check < tol else f'FAIL (max {o_check:,.2f})'}**")

    c_check = (MODEL["COGS_neto"] - (MODEL["COGS_mes_total"] - MODEL["IVA_credito_COGS"])).abs().max()
    st.write(f"- COGS_neto = COGS_bruto − IVA_crédito_COGS: **{'OK' if c_check < tol else f'FAIL (max {c_check:,.2f})'}**")

tabs = st.tabs([
    "00 – Supuestos",
    "01 – CapEx & Materiales",
    "02 – Costeo SKU",
    "02c – Planeación PRO (por SKU)",   # ← esta
    "03 – Unidades & Ventas",
    "04 – COGS & OPEX",
    "04b – IVA",
    "05 – EBITDA",
    "06 – Capital & Deuda",
    "07 – Caja & DSCR",
    "08 – Descargas",
    "09 – Márgenes & Competencia",
    "11 – VAN & TIR",
])

# === AJUSTE DE BASE DIARIA (solo en pestaña 00) ===============================
def render_ajuste_diario():
    import math

    st.markdown("---")
    st.subheader("Ajustar base diaria para alcanzar una meta **de ventas** ($/día)")

    c1, c2, c3 = st.columns(3)
    meta_total_dia = c1.number_input(
        "Meta de **ventas** $/día", min_value=0, max_value=5_000_000,
        value=350_000, step=10_000, key="adj_meta_total_dia"
    )
    mes_obj = c3.selectbox("Mes objetivo", options=list(range(1,13)), index=0, key="adj_mes_obj")
    i = int(mes_obj) - 1

    st.session_state["adj_modo"] = "Ventas"

    # Multiplicadores efectivos por línea (mes i)
    adj_pan = float(R[i]) * float(E[i]) * float(S_pan[i])
    adj_bol = float(R[i]) * float(E[i]) * float(S_bol[i])
    adj_pas = float(R[i]) * float(E[i]) * float(S_pas[i])
    adj_caf = float(R[i]) * float(E[i]) * float(S_caf[i])

    # Precios representativos
    PRECIO_PAN_KG = 2200.0
    PRECIO_BOL    = 1200.0
    PRECIO_PAS    = 3500.0
    PRECIO_CAF    = 2500.0

    # Aporte complementarios (día)
    comp_sales_day  = float(st.session_state.get("comp_ventas_dia_override") or 0.0)

    # Aporte SKUs PRO (día)
    pro_items = []
    aporte_pro_dia = 0.0
    if st.session_state.get("pro_use_global", True):
        adj_map = {
            "Pan (kg)": adj_pan,
            "Bolleria": adj_bol,
            "Bollería": adj_bol,
            "Pasteleria": adj_pas,
            "Pastelería": adj_pas,
            "Cafe": adj_caf,
            "Café": adj_caf,
        }
        for i, item in enumerate(PRO_SKUS):
            precio = float(st.session_state.get(f"pro_p_{i}", item["precio_sugerido"]))
            base_val = float(st.session_state.get(f"pro_b_{i}", item["base_dia"]))
            linea_key = item["Linea"]
            adj_linea = adj_map.get(linea_key, adj_map.get(linea_key.replace("Cafe", "Café"), 1.0))
            aporte_pro_dia += base_val * precio * adj_linea
            pro_items.append({
                "SKU": item["SKU"],
                "base": base_val,
                "precio": precio,
                "adj": adj_linea,
            })

    # Totales con la combinación actual
    ingreso_lineas_actual = (
        (base_pan / PAN_PIEZAS_POR_KG) * PRECIO_PAN_KG * adj_pan
        + base_bol * PRECIO_BOL * adj_bol
        + base_pas * PRECIO_PAS * adj_pas
        + base_caf * PRECIO_CAF * adj_caf
        + aporte_pro_dia
    )
    total_con_comp = ingreso_lineas_actual + float(comp_sales_day)

    mA, mB, mC = st.columns(3)
    mA.metric("Total SOLO líneas (base actual)", clp(ingreso_lineas_actual))
    mB.metric("Total líneas + Complementarios (base actual)", clp(total_con_comp))
    mC.metric("Faltante vs meta (con Complementarios)", clp(max(float(meta_total_dia) - total_con_comp, 0.0)))

    contrib_lin = (ingreso_lineas_actual/meta_total_dia*100) if meta_total_dia else 0
    contrib_comp = (float(comp_sales_day)/meta_total_dia*100) if meta_total_dia else 0
    st.caption(f"Contribución estimada a la meta • Líneas {min(contrib_lin,100):.1f}% · Complementarios {min(contrib_comp,100):.1f}%.")

    objetivo_lineas = max(float(meta_total_dia) - float(comp_sales_day), 0.0)
    k = (objetivo_lineas / ingreso_lineas_actual) if ingreso_lineas_actual > 0 else 1.0

    # Bases sugeridas manteniendo mix
    sug_pan_piezas = int(round(base_pan * k))
    sug_pan_kg     = float(sug_pan_piezas) / PAN_PIEZAS_POR_KG
    sug_bol        = int(round(base_bol * k))
    sug_pas        = int(round(base_pas * k))
    sug_caf        = int(round(base_caf * k))

    tabla_bases = pd.DataFrame({
        "Línea": ["Pan (piezas)", "Pan (kg)", "Bollería", "Pastelería", "Café"],
        "Base actual": [
            base_pan,
            round(base_pan / PAN_PIEZAS_POR_KG, 2),
            base_bol,
            base_pas,
            base_caf
        ],
        "Sugerida para meta": [
            sug_pan_piezas,
            round(sug_pan_kg, 2),
            sug_bol,
            sug_pas,
            sug_caf
        ]
    })
    if st.session_state.get("pro_use_global", True) and pro_items:
        base_pro_total = int(round(sum(info.get("base", 0.0) for info in pro_items)))
        sug_pro_total = int(round(sum(float(info.get("base", 0.0)) * k for info in pro_items)))
        tabla_bases = pd.concat([
            tabla_bases,
            pd.DataFrame([
                {
                    "Línea": "PRO",
                    "Base actual": base_pro_total,
                    "Sugerida para meta": sug_pro_total,
                }
            ])
        ], ignore_index=True)
    st.dataframe(tabla_bases, use_container_width=True)

    aporte_pro_sugerido = 0.0
    pro_sug_rows = []
    if pro_items:
        for info in pro_items:
            base_sug = int(round(info["base"] * k))
            aporte_pro_sugerido += base_sug * info["precio"] * info["adj"]
            pro_sug_rows.append({
                "SKU": info["SKU"],
                "Base actual": int(info["base"]),
                "Sugerida para meta": base_sug,
            })

    ingreso_lineas_sugerido = (
        (sug_pan_piezas / PAN_PIEZAS_POR_KG) * PRECIO_PAN_KG * adj_pan
        + sug_bol * PRECIO_BOL * adj_bol
        + sug_pas * PRECIO_PAS * adj_pas
        + sug_caf * PRECIO_CAF * adj_caf
        + aporte_pro_sugerido
    )
    total_dia_prev = ingreso_lineas_sugerido + float(comp_sales_day)

    if pro_sug_rows:
        st.markdown("**SKUs PRO**")
        st.dataframe(pd.DataFrame(pro_sug_rows), use_container_width=True)

    c1b, c2b, c3b = st.columns(3)
    c1b.metric("Objetivo de líneas (meta − Complementarios)", clp(objetivo_lineas))
    c2b.metric("Factor de escala k", f"{k:.3f}")
    c3b.metric("Total $/día estimado (con Complementarios)", clp(total_dia_prev))

# --- 00 Supuestos -------------------------------------------------------------
with tabs[0]:
    st.subheader("Días abiertos, rampa y factores")
    df_sup = pd.DataFrame({
        "Mes":MESES,
        "Días abiertos":D,
        "Fines de semana":F,
        "Estacionalidad":E,
        "Rampa":R,
        "S_pan": np.array(S_pan),
        "S_bol": np.array(S_bol),
        "S_pas": np.array(S_pas),
        "S_caf": np.array(S_caf),
    })
    st.dataframe(df_sup, use_container_width=True)

    st.subheader("Unidades por línea (con estos supuestos)")
    st.dataframe(MODEL["U_df"], use_container_width=True)
    st.markdown("---")
    st.subheader("Precios de venta (visibles en Supuestos)")

    df_precios_lineas = pd.DataFrame([
        {"Línea": "Pan (kg)",   "Precio venta": MODEL["prices"]["Pan (kg)"]},
        {"Línea": "Bolleria",   "Precio venta": MODEL["prices"]["Bolleria"]},
        {"Línea": "Pasteleria", "Precio venta": MODEL["prices"]["Pasteleria"]},
        {"Línea": "Cafe",       "Precio venta": MODEL["prices"]["Cafe"]},
    ])
    st.markdown("**Líneas principales**")
    st.dataframe(
        df_precios_lineas.assign(**{"Precio venta": lambda d: d["Precio venta"].map(clp)}),
        use_container_width=True
    )

    st.markdown("**Complementarios (overrides)**")
    df_precios_comp = (
        pd.Series(PRICE_COMP_OVERRIDE_RAW, name="Precio venta")
          .rename_axis("SKU (Complementario)")
          .reset_index()
    )
    st.dataframe(
        df_precios_comp.assign(**{"Precio venta": lambda d: d["Precio venta"].map(clp)}),
        use_container_width=True
    )
    st.caption("Nota: la mermelada se fija por kg; no usamos pote.")

    st.subheader("SKUs PRO (Masa Madre & Panetón)")
    with st.expander("Ajustes por SKU (precio y base/día)", expanded=False):
        def _sync_from_tab(i: int, kind: str):
            src = f"pro_{kind}_tab_{i}"
            dst = f"pro_{kind}_{i}"
            st.session_state[dst] = st.session_state.get(src, st.session_state.get(dst, 0))

        for i, item in enumerate(PRO_SKUS):
            sku = item["SKU"]
            c1, c2 = st.columns(2)

            with c1:
                p_val = int(st.session_state.get(f"pro_p_{i}", item["precio_sugerido"]))
                p_key = f"pro_p_tab_{i}"
                if st.session_state.get(p_key) != p_val:
                    st.session_state[p_key] = p_val
                st.number_input(
                    f"Precio – {sku}",
                    min_value=0,
                    max_value=50_000,
                    value=p_val,
                    key=p_key,
                    on_change=_sync_from_tab,
                    args=(i, "p"),
                    help="Precio sugerido/actual para este SKU PRO",
                )

            with c2:
                b_val = int(st.session_state.get(f"pro_b_{i}", item.get("base_dia", 0)))
                b_key = f"pro_b_tab_{i}"
                if st.session_state.get(b_key) != b_val:
                    st.session_state[b_key] = b_val
                st.number_input(
                    f"Base/día – {sku}",
                    min_value=0,
                    max_value=10_000,
                    value=b_val,
                    key=b_key,
                    on_change=_sync_from_tab,
                    args=(i, "b"),
                    help="Cantidad base diaria para este SKU PRO",
                )

            st.caption("Los cambios aquí se sincronizan con el sidebar y la pestaña PRO.")
            st.divider()

    with st.expander("Fórmula de unidades (por línea)"):
        st.markdown("""
**Unidades_mes = base_día × Días_mes × Rampa_mes × Estacionalidad_mes × S_mes × Uplifts_mes**
- *Pan se muestra en kg: piezas ÷ 10 = kg.*
        """)

    render_ajuste_diario()
# --- FIN 00 -------------------------------------------------------------------

# --- 01 CapEx & Materiales ---------------------------------------------------
with tabs[1]:
    st.subheader("CapEx")
    st.dataframe(CAPEX_TAB, use_container_width=True)
    st.metric("CapEx total", clp(MODEL['capex_total']))

    st.subheader("Materiales (one-off)")
    if not MATERIAL.empty:
        st.dataframe(MATERIAL, use_container_width=True)
        st.metric("Materiales – compra inicial", clp(MODEL['materiales_total']))
        st.metric("CapEx + Materiales (compra inicial)", clp(MODEL['capex_total'] + MODEL['materiales_total']))

    st.subheader("Indirectos (one-off)")
    st.dataframe(read_first("05_Indirectos.csv"), use_container_width=True)

# --- 02 Costeo SKU -----------------------------------------------------------
with tabs[2]:
    st.subheader("Costeo por SKU (oficial o desde BOM)")

    # Fuentes disponibles
    oficial = COSTOS_UNIF if not COSTOS_UNIF.empty else pd.DataFrame()
    bom = BOM if not BOM.empty else pd.DataFrame()

    # Columnas de referencia
    col_prod_of = "Producto" if "Producto" in oficial.columns else None
    col_sku_bom = next((c for c in ["SKU","sku","Producto"] if c in bom.columns), None)

    # Opciones: unión de oficial + BOM (sin duplicados)
    opciones = []
    if col_prod_of:
        opciones += list(oficial[col_prod_of].dropna().unique())
    if col_sku_bom:
        opciones += list(bom[col_sku_bom].dropna().unique())
    # Orden sugerido, dejando lo demás al final
    orden_pref = {
        "PAN — Marraqueta 100g (econ)": 0,
        "CROISSANT — 70g (econ)": 1,
        "TROZO — Mil Hojas 1/8 (A PDF)": 2,
        "CAFÉ — Taza estándar": 3,
        "Concentrado Masa Madre 1 kg": 4,
        "Pan Masa Madre 1kg": 5,
        "Paneton Masa Madre 1kg": 6,
    }
    opciones = sorted(pd.unique(opciones), key=lambda x: orden_pref.get(x, 999))

    if len(opciones) == 0:
        st.warning("No hay productos ni en COSTOS_LINEA_UNIFICADO_APP.csv ni en 01_BOM.csv.")
        st.stop()

    producto_sel = st.selectbox("Producto", opciones, index=0, key="t2_prod_any")

    # Preferimos detalle oficial si existe; si no, calculamos desde BOM+02_Precios
    det = pd.DataFrame()
    total = np.nan

    if col_prod_of and producto_sel in set(oficial[col_prod_of].unique()):
        det = oficial.loc[oficial[col_prod_of] == producto_sel].copy()
        if "Costo_insumo_CLP" in det.columns:
            total = float(pd.to_numeric(det["Costo_insumo_CLP"], errors="coerce").sum())
    else:
        det, total = costeo_insumos_por_sku(BOM, INS_PREP, producto_sel, merma_global_pct=0.0)

    metric_clp("Costo ingredientes por unidad", total if pd.notna(total) else 0)
    st.dataframe(det, use_container_width=True)

    with st.expander("Resumen por insumo (ordenado por impacto)"):
        if not det.empty:
            col_cost = next((c for c in ["Costo_insumo_CLP","Subtotal_CLP","Costo_unit_CLP"] if c in det.columns), None)
            col_ins  = next((c for c in ["Insumo","insumo"] if c in det.columns), None)
            if col_cost and col_ins:
                resumen = (
                    det.groupby(col_ins, as_index=False)[col_cost]
                      .sum()
                      .sort_values(col_cost, ascending=False)
                )
                show_df_money(resumen, [col_cost], use_container_width=True)
            else:
                st.info("No se pudieron identificar las columnas de insumo/costo para agrupar.")


# --- 03 Unidades & Ventas ----------------------------------------------------
with tabs[4]:
    st.subheader("01 – Demanda (resumen)")
    incluir_pro = st.session_state.get("pro_use_global", True)
    pro_df_global = MODEL.get("sku_pro_df", pd.DataFrame())
    sku_df_model = MODEL["sku_df"].copy()
    pro_skus = set()
    if incluir_pro and isinstance(pro_df_global, pd.DataFrame) and not pro_df_global.empty:
        pro_skus = set(pro_df_global["SKU"].dropna().unique())
        sku_df_base = sku_df_model.loc[~sku_df_model["SKU"].isin(pro_skus)].copy()
    else:
        sku_df_base = sku_df_model

    if not sku_df_base.empty:
        df_lineas = (
            sku_df_base.groupby("Linea", as_index=False)
            .agg(unidades=("Unidades", "sum"), ingreso=("Venta_mes", "sum"))
        )
    else:
        df_lineas = pd.DataFrame(columns=["Linea", "unidades", "ingreso"])

    cogs_detalle = MODEL.get("COGS_detalle", pd.DataFrame())
    if not df_lineas.empty and not cogs_detalle.empty:
        cogs_linea = cogs_detalle.groupby("Linea", as_index=False)["COGS"].sum()
        if incluir_pro and pro_skus and isinstance(pro_df_global, pd.DataFrame) and not pro_df_global.empty and "cogs" in pro_df_global.columns:
            pro_cogs_linea = pro_df_global.groupby("Linea", as_index=False)["cogs"].sum()
            cogs_linea = cogs_linea.merge(pro_cogs_linea, on="Linea", how="left", suffixes=("", "_pro"))
            if "cogs_pro" in cogs_linea.columns:
                cogs_linea["COGS"] = cogs_linea["COGS"] - cogs_linea["cogs_pro"].fillna(0.0)
                cogs_linea = cogs_linea.drop(columns="cogs_pro")
        df_lineas = df_lineas.merge(cogs_linea.rename(columns={"COGS": "cogs"}), on="Linea", how="left")
    # --- Sanitizar df_lineas antes de calcular margen ---
    # 1) Quitar columnas duplicadas y normalizar índice
    df_lineas = df_lineas.loc[:, ~df_lineas.columns.duplicated()].copy()
    df_lineas = df_lineas.reset_index(drop=True)

    # 2) Helper: tomar una columna aun si vino duplicada y pasar a numérico
    def _solid_num_col(df, name):
        col = df[name] if name in df.columns else 0
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return pd.to_numeric(col, errors="coerce").fillna(0.0)

    # 3) Asegurar existencia de 'cogs' y tipos numéricos
    if "cogs" not in df_lineas.columns:
        df_lineas["cogs"] = 0.0

    ingreso_s = _solid_num_col(df_lineas, "ingreso").to_numpy()
    cogs_s    = _solid_num_col(df_lineas, "cogs").to_numpy()

    # 4) Evitar alineación por índice: operar por valores (arrays)
    df_lineas["margen"] = ingreso_s - cogs_s
    df_lineas["unidades"] = pd.to_numeric(df_lineas.get("unidades", df_lineas.get("Unidades", 0)), errors="coerce").fillna(0.0)

    df_lineas = df_lineas.rename(columns={"Linea": "Línea"})
    df_lineas = df_lineas[["Línea", "unidades", "ingreso", "cogs", "margen"]]

    if incluir_pro and isinstance(pro_df_global, pd.DataFrame) and not pro_df_global.empty:
        pro_summary = _pro_agg_from_skus(pro_df_global)
        df_lineas = pd.concat([df_lineas, pd.DataFrame([pro_summary])], ignore_index=True)

    show_df_money(
        df_lineas.assign(
            **{"unidades": lambda d: d["unidades"].map(lambda x: f"{int(round(float(x))):,}".replace(",", ".") if pd.notna(x) else "—")}
        ),
        money_cols=["ingreso", "cogs", "margen"],
        use_container_width=True,
    )

    if incluir_pro and isinstance(pro_df_global, pd.DataFrame) and not pro_df_global.empty:
        u = int(pd.to_numeric(pro_df_global.get("unidades", pro_df_global.get("Unidades", 0)), errors="coerce").fillna(0).sum())
        rev = float(pd.to_numeric(pro_df_global.get("ingreso", pro_df_global.get("Venta_mes", 0.0)), errors="coerce").fillna(0.0).sum())
        total_rev = float(pd.to_numeric(df_lineas["ingreso"], errors="coerce").fillna(0.0).sum())
        mix = 100.0 * rev / total_rev if total_rev > 0 else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Unidades PRO (día)", f"{u:,}".replace(",", "."))
        c2.metric("Ingreso PRO (día)", clp(rev))
        c3.metric("Mix PRO", f"{mix:.1f}%")

    st.markdown("---")
    st.subheader("Unidades por línea (escenario activo)")
    st.dataframe(MODEL["U_df"], use_container_width=True)

    st.subheader("Ventas por línea (representantes + complementarios)")
    ventas_tbl = MODEL["sku_df"][["Mes","Linea","SKU","Unidades","Precio","Venta_mes"]].copy()
    show_df_money(ventas_tbl, ["Precio","Venta_mes"], use_container_width=True)


    # Totales por mes (líneas + complementarios)
    ventas_lineas_mes = (
        ventas_tbl.groupby("Mes", as_index=False)["Venta_mes"]
                  .sum()
                  .rename(columns={"Venta_mes": "Ventas líneas"})
    )
    ventas_lineas_mes = (
        ventas_lineas_mes.set_index("Mes")
                         .reindex(MESES)
                         .fillna(0.0)
                         .rename_axis("Mes")
                         .reset_index()
    )
    comp_mes = s2df(MODEL["ventas_compl_mes"], "Complementarios")

    totales_mes = ventas_lineas_mes.merge(comp_mes, on="Mes", how="left")
    totales_mes["Complementarios"] = totales_mes["Complementarios"].fillna(0.0)
    totales_mes["Total"] = totales_mes["Ventas líneas"] + totales_mes["Complementarios"]

    st.markdown("**Totales por mes (líneas + complementarios)**")
    show_df_money(
    totales_mes.rename(columns={
        "Ventas líneas": "Ventas líneas (c/IVA)",
        "Complementarios": "Complementarios (c/IVA)",
        "Total": "Total (c/IVA)"
    }),
    use_container_width=True
)

    # Resumen anual por producto y totales
    st.markdown("---")
    st.subheader("Resumen anual por producto y totales")

    def _fmt_int(x):
        try:
            return f"{int(round(float(x))):,}".replace(",", ".")
        except Exception:
            return str(x)

    ventas_sku_anual = (
        MODEL["sku_df"]
        .groupby(["Linea","SKU"], as_index=False)
        .agg(
            Unidades_año=("Unidades","sum"),
            Precio_ref=("Precio","first"),
            Ventas_año=("Venta_mes","sum"),
        )
    )

    comp_df = pd.DataFrame(columns=["SKU","Uds/día","Precio","Uds/año","Ventas_año"])
    if not COSTO_COMP.empty:
        def _find_col(df, keys):
            keys = [k.lower() for k in keys]
            for c in df.columns:
                if any(k in c.lower() for k in keys):
                    return c
            return None

        col_prod  = _find_col(COSTO_COMP, ["producto","sku","nombre"])
        col_cost  = _find_col(COSTO_COMP, ["costo_insumo","costo_unit","costo"])
        col_price = _find_col(COSTO_COMP, ["precio_venta","precio","pvp"])

        costo_sum = (COSTO_COMP.groupby(col_prod, as_index=True)[col_cost]
                              .sum().map(_num).fillna(0.0))

        price_map = {}
        if col_price:
            price_map = (
                COSTO_COMP[[col_prod, col_price]]
                .dropna(subset=[col_price])
                .drop_duplicates(subset=[col_prod])
                .set_index(col_prod)[col_price]
                .map(_num).to_dict()
            )
        if len(price_map) == 0 and not PRICES_SKU.empty:
            c_prod2 = _find_col(PRICES_SKU, ["producto","sku","nombre"])
            c_p2    = _find_col(PRICES_SKU, ["precio","pvp"])
            if c_prod2 and c_p2:
                price_map = (
                    PRICES_SKU[[c_prod2, c_p2]]
                    .dropna(subset=[c_p2])
                    .drop_duplicates(subset=[c_prod2])
                    .set_index(c_prod2)[c_p2]
                    .map(_num).to_dict()
                )
        if len(price_map) == 0:
            price_map = {k: float(v)*1.5 for k, v in costo_sum.items()}

        price_map_norm = { _norm_txt(k): float(v) for k, v in price_map.items() }
        price_map_norm.update(PRICE_COMP_OVERRIDE)

        sel = st.session_state.get("y_sel", [])
        dias_equiv_anio = float(np.sum(np.array(D, dtype=float) * np.array(R, dtype=float)))

        rows = []
        for sku_name in sel:
            uds_dia = float(st.session_state.get(f"y_ud_{sku_name}", 0.0))
            p = float(price_map_norm.get(_norm_txt(sku_name), 0.0))
            uds_anio = uds_dia * dias_equiv_anio
            ventas_anio = uds_anio * p
            rows.append({
                "SKU": sku_name,
                "Uds/día": uds_dia,
                "Precio": p,
                "Uds/año": uds_anio,
                "Ventas_año": ventas_anio
            })
        if rows:
            comp_df = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Líneas principales – ventas por producto (año)**")
        st.dataframe(
            ventas_sku_anual.assign(
                **{"Unidades_año": lambda d: d["Unidades_año"].map(_fmt_int)},
                Precio_ref=lambda d: d["Precio_ref"].map(clp),
                **{"Ventas_año":   lambda d: d["Ventas_año"].map(clp)},
            ),
            use_container_width=True
        )

        tot_linea = (
            ventas_sku_anual.groupby("Linea", as_index=False)
                            .agg(Unidades_año=("Unidades_año","sum"),
                                 Ventas_año=("Ventas_año","sum"))
        )
        st.markdown("**Totales por línea (año)**")
        st.dataframe(
            tot_linea.assign(
                **{"Unidades_año": lambda d: d["Unidades_año"].map(_fmt_int)},
                **{"Ventas_año":   lambda d: d["Ventas_año"].map(clp)},
            ),
            use_container_width=True
        )

    with c2:
        st.markdown("**Complementarios – ventas por SKU (año)**")
        if comp_df.empty:
            st.info("Sin SKUs de complementarios seleccionados en el sidebar.")
        else:
            st.dataframe(
                comp_df.assign(
                    **{"Uds/día":   lambda d: d["Uds/día"].map(_fmt_int)},
                    Precio=lambda d: d["Precio"].map(clp),
                    **{"Uds/año":   lambda d: d["Uds/año"].map(_fmt_int)},
                    **{"Ventas_año":lambda d: d["Ventas_año"].map(clp)},
                ),
                use_container_width=True
            )

    total_lineas = float(ventas_sku_anual["Ventas_año"].sum())
    total_comp   = float(comp_df["Ventas_año"].sum()) if not comp_df.empty else float(MODEL["ventas_compl_mes"].sum())
    total_general = total_lineas + total_comp

    m1, m2, m3 = st.columns(3)
    metric_clp("Total líneas (año)", total_lineas)
    metric_clp("Total complementarios (año)", total_comp)
    metric_clp("Total general (año)", total_general)
# Mapa de factores S por línea (reusa los tuyos)
_S_FACTORS = {
    "Pan (kg)": lambda: S_pan,
    "Bolleria": lambda: S_bol,
    "Pasteleria": lambda: S_pas,
    "Cafe": lambda: S_caf,
}
def _factor_linea(linea):
    f = _S_FACTORS.get(linea, lambda: [1.0]*12)
    return f()

def _unidades_mensuales_desde_base(base_dia, linea, Dv, Rv, Ev):
    Sv = _factor_linea(linea)
    arr = np.array([base_dia]*12, dtype=float) * np.array(Dv)*np.array(Rv)*np.array(Ev)*np.array(Sv)
    return arr.round(0)

# --------------------------- Pestaña PRO -------------------------------------
with tabs[3]:
    st.subheader("Planeación y Márgenes por SKU (PRO)")

    st.markdown("**Factores de escala**: usa tus D, R, E y S_linea actuales.")
    fabricar_conc = st.checkbox(
        "Fabricar Concentrado de Masa Madre in-house (BOM anidado)",
        value=st.session_state.get("pro_fabricar_conc", True),
        key="pro_fabricar_conc"
    )

    st.markdown("### Catálogo de SKUs (precio y base diaria)")
    cfg_rows = []
    for i, item in enumerate(PRO_SKUS):
        c0, c1, c2, c3 = st.columns([1.2, 3, 1.4, 1.4])
        c0.write(item["Linea"])
        c1.write(item["SKU"])
        p = c2.number_input("Precio", min_value=0, max_value=50_000, value=int(item["precio_sugerido"]),
                            step=100, key=f"pro_p_{i}")
        b = c3.number_input("Base día", min_value=0, max_value=10_000, value=int(item["base_dia"]),
                            step=1, key=f"pro_b_{i}")
        cfg_rows.append({"Linea": item["Linea"], "SKU": item["SKU"], "Precio": float(p), "Base_dia": int(b)})
    cfg_df = pd.DataFrame(cfg_rows)

    # Costos unitarios con/ sin anidado
    costos, detalles = [], {}
    for sku in cfg_df["SKU"]:
        det_sku, cunit = costeo_sku_pro(sku, fabricar_concentrado=fabricar_conc)
        detalles[sku] = det_sku
        costos.append({"SKU": sku, "Costo_unit": float(cunit)})
    costos_df = pd.DataFrame(costos)

    # Unidades, ventas y consumo de insumos por mes
    ventas_rows, consumo_ins_rows = [], []
    for _, r in cfg_df.iterrows():
        linea, sku, precio, base_dia = r["Linea"], r["SKU"], r["Precio"], r["Base_dia"]
        U_mes = _unidades_mensuales_desde_base(base_dia, linea, D, R, E)
        for i, mes in enumerate(MESES):
            uds = float(U_mes[i])
            ventas_rows.append({"Mes": mes, "Linea": linea, "SKU": sku,
                                "Unidades": uds, "Precio": precio, "Venta_mes": uds * precio})
            det = detalles.get(sku, pd.DataFrame())
            if not det.empty and {"Insumo", "Consumo_base", "Unidad_base"}.issubset(det.columns):
                tmp = det[["Insumo", "Unidad_base", "Consumo_base"]].copy()
                tmp["Mes"] = mes; tmp["SKU"] = sku
                tmp["Consumo_total_mes"] = pd.to_numeric(tmp["Consumo_base"], errors="coerce").fillna(0.0) * uds
                consumo_ins_rows.append(tmp)

    ventas_sku_mes = pd.DataFrame(ventas_rows) if ventas_rows else pd.DataFrame(
        columns=["Mes","Linea","SKU","Unidades","Precio","Venta_mes"])
    consumo_ins_mes = (pd.concat(consumo_ins_rows, ignore_index=True)
                       if consumo_ins_rows else
                       pd.DataFrame(columns=["Mes","SKU","Insumo","Unidad_base","Consumo_total_mes"]))

        # COGS y márgenes
    df_costos = ventas_sku_mes.merge(costos_df, on="SKU", how="left")
    df_costos["COGS_mes"] = df_costos["Unidades"] * df_costos["Costo_unit"]
    df_costos["Margen_mes"] = df_costos["Venta_mes"] - df_costos["COGS_mes"]
    df_costos["Margen_%"] = np.where(
        df_costos["Venta_mes"] > 0,
        df_costos["Margen_mes"] / df_costos["Venta_mes"] * 100.0,
        np.nan
    )

    st.markdown("### Márgenes por SKU (mensual)")
    show_df_money(
        df_costos.assign(
            **{"Costo_unit": lambda d: d["Costo_unit"].map(clp)},
            **{"COGS_mes":   lambda d: d["COGS_mes"].map(clp)},
            **{"Venta_mes":  lambda d: d["Venta_mes"].map(clp)},
            **{"Margen_mes": lambda d: d["Margen_mes"].map(clp)},
            **{"Margen_%":   lambda d: d["Margen_%"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")},
        ),
        use_container_width=True
    )

    st.markdown("#### Resumen anual por SKU")
    anual = (
        df_costos.groupby(["Linea","SKU"], as_index=False)
                 .agg(Unidades_año=("Unidades","sum"),
                      Ventas_año=("Venta_mes","sum"),
                      COGS_año=("COGS_mes","sum"))
    )
    anual["Margen_año"] = anual["Ventas_año"] - anual["COGS_año"]
    show_df_money(
        anual.assign(
            **{"Ventas_año": lambda d: d["Ventas_año"].map(clp)},
            **{"COGS_año":   lambda d: d["COGS_año"].map(clp)},
            **{"Margen_año": lambda d: d["Margen_año"].map(clp)},
        ),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas (año, SKUs PRO)", clp(float(anual["Ventas_año"].sum()) if not anual.empty else 0))
    c2.metric("COGS (año, SKUs PRO)",    clp(float(anual["COGS_año"].sum())  if not anual.empty else 0))
    c3.metric("Margen (año, SKUs PRO)",  clp(float(anual["Margen_año"].sum()) if not anual.empty else 0))

    st.markdown("### Consumo de insumos (plan de compras)")
    if consumo_ins_mes.empty:
        st.info("Sin consumo calculado (revisa el catálogo y las bases diarias).")
    else:
        consumo_mes_ins = (
            consumo_ins_mes.groupby(["Mes","Insumo","Unidad_base"], as_index=False)
                           .agg(Cantidad=("Consumo_total_mes","sum"))
        )
        st.dataframe(consumo_mes_ins, use_container_width=True)

        st.markdown("#### Total anual por insumo")
        consumo_anual_ins = (
            consumo_mes_ins.groupby(["Insumo","Unidad_base"], as_index=False)
                           .agg(Cantidad=("Cantidad","sum"))
                           .sort_values("Cantidad", ascending=False)
        )
        st.dataframe(consumo_anual_ins, use_container_width=True)

# ========================== FIN 02c – Planeación PRO ==========================

# --- 04 COGS & OPEX ----------------------------------------------------------
with tabs[5]:
    st.subheader("Costo unitario y margen por representante")
    rows=[]
    for lin in ["Pan (kg)","Bolleria","Pasteleria","Cafe"]:
        price = MODEL["prices"][lin]
        cu = float(MODEL["cost_unit_ins"].get(lin,0.0)) + float(MODEL["per_unit_map"].get(lin.replace(" (kg)",""),0.0))
        margin = price - cu
        m_pct = (margin/price*100.0) if price>0 else 0.0
        rows.append({"Línea":lin,"Precio":clp(price),"Costo unit.":clp(cu),"Margen unit.":clp(margin),"Margen %":f"{m_pct:.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("#### Auditoría unitario (insumos + per-unit)")
    audit_rows = []
    for lin in ["Pan (kg)","Bolleria","Pasteleria","Cafe"]:
        ins = float(MODEL["cost_unit_ins"].get(lin,0.0))
        per = float(MODEL["per_unit_map"].get(lin.replace(" (kg)",""),0.0))
        audit_rows.append({"Línea": lin, "Insumos_ud": clp(ins), "PerUnit_energ/log": clp(per), "Unit_total": clp(ins+per)})
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

    st.subheader("COGS por mes (líneas + complementarios)")
    df_cogs_show = MODEL.get("COGS_detalle_full", MODEL["COGS_detalle"])
    df_cogs_show = df_cogs_show.copy()
    df_cogs_show["__Mes_ord__"] = pd.Categorical(df_cogs_show["Mes"], categories=MESES, ordered=True)
    df_cogs_show = df_cogs_show.sort_values(["__Mes_ord__", "Linea"]).drop(columns="__Mes_ord__", errors="ignore")
    st.dataframe(df_cogs_show, use_container_width=True)


    c1,c2,c3 = st.columns(3)
    c1.metric("COGS total (sin complementarios)", clp(MODEL["COGS_mes_sin_compl"].sum()))
    c2.metric("COGS complementarios (año)", clp(MODEL["COGS_compl_mes"].sum()))
    c3.metric("COGS total (incl. complementarios)", clp(MODEL["COGS_mes_total"].sum()))

    st.subheader("OPEX por mes (bruto y neto)")
    df_opex = pd.DataFrame({
        "Mes": MESES,
        "OPEX_bruto": MODEL["OPEX_mes"].values,
        "OPEX_neto": MODEL["OPEX_neto"].values
    })
    st.dataframe(df_opex, use_container_width=True)
    b1,b2 = st.columns(2)
    b1.metric("OPEX bruto (año)", clp(MODEL["OPEX_mes"].sum()))
    b2.metric("OPEX neto (año)", clp(MODEL["OPEX_neto"].sum()))

    if st.session_state.get("pro_fabricar_conc", False):
        st.subheader("Consumo por SKUs PRO (Concentrado in-house)")
        consumo_df = MODEL.get("consumo_concentrado_df", pd.DataFrame())
        if isinstance(consumo_df, pd.DataFrame) and not consumo_df.empty:
            show_df_money(
                consumo_df.assign(
                    **{"Consumo_concentrado": lambda d: d["Consumo_concentrado"].map(lambda x: f"{float(x):,.2f}".replace(",", ".") if pd.notna(x) else "—")},
                    **{"Costo_concentrado": lambda d: d["Costo_concentrado"].map(clp)},
                ),
                money_cols=["Costo_concentrado"],
                use_container_width=True,
            )
        else:
            st.info("No hay consumo de concentrado disponible para los SKUs PRO.")

# --- 04b IVA ------------------------------------------------------------------
with tabs[6]:
    st.subheader("IVA – Débito y Crédito")
    iva_df = pd.DataFrame({
        "Mes":MESES,
        "Ventas_brutas":MODEL["ventas_brutas"].values,
        "IVA_débito":MODEL["IVA_debito"].values,
        "IVA_crédito_TOTAL":MODEL["IVA_credito"].values,
        "IVA_crédito_OPEX":MODEL["IVA_credito_OPEX"].values,
        "IVA_crédito_COGS":MODEL["IVA_credito_COGS"].values,
        "IVA_pagar":MODEL["IVA_pagar"].values,
        "Ventas_netas":MODEL["ventas_netas"].values
    })
    st.dataframe(iva_df, use_container_width=True)

# --- 05 EBITDA ---------------------------------------------------------------
with tabs[7]:
    st.subheader("EBITDA")

    # Solo métricas BRUTAS para esta vista
    fin = pd.DataFrame({
        "Mes": MESES,
        "Ventas_brutas": MODEL["ventas_brutas"].values,
        "COGS_bruto":    MODEL["COGS_mes_total"].values,
        "OPEX_bruto":    MODEL["OPEX_mes"].values,
    })

    # EBITDA bruto: Ventas_brutas - COGS_bruto - OPEX_bruto
    fin["EBITDA"] = fin["Ventas_brutas"] - fin["COGS_bruto"] - fin["OPEX_bruto"]

    # Tabla (formateada en CLP)
    show_df_money(fin, use_container_width=True)

    # Gráfico: orden fijo Ene→Dic y EBITDA dibujado ENCIMA
    fin["Mes"] = pd.Categorical(fin["Mes"], categories=MESES, ordered=True)
    fin = fin.sort_values("Mes")

    # Etiqueta "01 Ene", "02 Feb", ... (según los meses presentes en 'fin')
    fin_plot = fin.copy()
    fin_plot["Mes_lbl"] = [f"{MESES.index(str(m))+1:02d} {m}" for m in fin_plot["Mes"].astype(str)]

    cols_plot = ["COGS_bruto", "OPEX_bruto", "Ventas_brutas", "EBITDA"]  # EBITDA al final = arriba
    st.area_chart(fin_plot.set_index("Mes_lbl")[cols_plot])

    st.caption("EBITDA = Ventas_brutas − COGS_bruto − OPEX_bruto.")

# # --- 06 Capital & Deuda ------------------------------------------------------
with tabs[8]:
    st.subheader("Capital & Deuda")
    try:
        cap_map = dict(CAPITAL[["Concepto","Valor"]].values)
    except Exception:
        cap_map = {}

    # Helper robusto: castea a int con default si viene vacío o inválido
    def _int_safe(x, default=0):
        v = _num(x)
        return int(v) if pd.notna(v) and np.isfinite(v) else int(default)

    # === Inputs en el cuerpo (no sidebar) ====================================
    c1, c2, c3 = st.columns(3)
    with c1:
        aporte_base = st.number_input(
            "Aporte dueños", 0, 500_000_000,
            _int_safe(cap_map.get("Aporte_duenos_CLP"), 20_000_000),
            500_000, key="cd_aporte_duenos"
        )
        ct_ini = st.number_input(
            "CT inicial", 0, 200_000_000,
            _int_safe(cap_map.get("Capital_trabajo_inicial_CLP"), 2_000_000),
            100_000, key="cd_ct_inicial"
        )
    with c2:
        permisos = st.number_input(
            "Permisos", 0, 50_000_000,
            _int_safe(cap_map.get("Permisos_tramites_CLP"), 500_000),
            50_000, key="cd_permisos"
        )
        garantia = st.number_input(
            "Garantía arriendo", 0, 200_000_000,
            _int_safe(cap_map.get("Garantia_arriendo_CLP"), 5_000_000),
            100_000, key="cd_garantia_arriendo"
        )
    with c3:
        remodel = st.number_input(
            "Remodelación", 0, 300_000_000,
            _int_safe(cap_map.get("Remodelacion_CLP"), 5_000_000),
            100_000, key="cd_remodelacion"
        )

    # Cálculo de inversión base (compra inicial)
    inversion_total_base = float(MODEL["capex_total"] + MODEL["materiales_total"] + ct_ini + permisos + garantia + remodel)
    deuda_req_base = max(inversion_total_base - aporte_base, 0.0)

    # Parámetros de deuda (francés)
    st.subheader("Parámetros de deuda (francés)")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        tna = st.number_input("TNA nominal (%)", 0.0, 100.0, 18.0, 0.1, key="cd_tna")
    with d2:
        plazo = st.number_input("Plazo (meses)", 1, 120, 36, 1, key="cd_plazo")
    with d3:
        gracia = st.number_input("Gracia (meses, interés-solo)", 0, 24, 3, 1, key="cd_gracia")
    with d4:
        intereses_solo = st.checkbox("Gracia intereses-solo", True, key="cd_intsolo")

    # Helper: tabla de deuda francés
    def tabla_deuda_frances(monto, tna, plazo, gracia, intereses_solo):
        monto = float(monto)
        if monto <= 0 or plazo <= 0:
            return pd.DataFrame(columns=["Mes","Cuota","Interes","Amortizacion","Saldo"])
        i_m = tna/12/100.0
        saldo = float(monto)
        n = plazo - (gracia if intereses_solo else 0)
        cuota_const = (saldo*i_m*((1+i_m)**n))/(((1+i_m)**n)-1) if i_m>0 and n>0 else (saldo/n if n>0 else 0)
        rows=[]
        for mm in range(1, plazo+1):
            interes = saldo*i_m
            if intereses_solo and mm<=gracia:
                cuota = interes; amort = 0.0
            else:
                cuota = cuota_const; amort = max(cuota - interes, 0.0)
            saldo = max(saldo - amort, 0.0)
            rows.append({"Mes":mm,"Cuota":cuota,"Interes":interes,"Amortizacion":amort,"Saldo":saldo})
        return pd.DataFrame(rows)

    # === Mezclar financiamiento del faltante 12m ==============================
    st.subheader("Liquidez 12m → mezclar deuda/aporte")
    use_mix = st.checkbox("Incluir faltante de caja (12m) en la deuda/aporte", value=False, key="cd_mix_on")
    mix_pct = st.slider("Financiar con deuda (%) del faltante 12m", 0, 100, 0, 5, key="cd_mix_pct")/100.0

    # Iteración corta para converger
    deuda = float(deuda_req_base)
    aporte_equity = float(aporte_base)
    aportes_mes = np.zeros(12, dtype=float)
    deuda_tabla = pd.DataFrame()

    for _ in range(6):  # 6 iteraciones suele bastar
        deuda_tabla = tabla_deuda_frances(deuda, tna, plazo, gracia, intereses_solo)
        cuotas_12 = deuda_tabla["Cuota"].iloc[:12].values if not deuda_tabla.empty else np.zeros(12)

        # Flujo operativo a 12m (sin financiamiento): EBITDA - IVA - servicio de deuda
        ebitda_12 = MODEL["EBITDA"].values[:12].astype(float)
        iva_p_12  = MODEL["IVA_pagar"].values[:12].astype(float)
        flujo_mes = ebitda_12 - iva_p_12 - cuotas_12

        # Aportes para mantener caja ≥ 0 con caja inicial = CT inicial
        aportes_new = []
        saldo = float(ct_ini)
        for f in flujo_mes:
            caja_pre = saldo + f
            a = -caja_pre if caja_pre < 0 else 0.0
            saldo = caja_pre + a
            aportes_new.append(a)
        aportes_new = np.array(aportes_new, dtype=float)

        if not use_mix:
            aportes_mes = aportes_new
            break

        faltante_total = float(aportes_new.sum())
        deuda_nueva = deuda_req_base + mix_pct * faltante_total
        aporte_nuevo = aporte_base + (1.0 - mix_pct) * faltante_total

        if abs(deuda_nueva - deuda) < 1.0:  # convergió
            deuda, aporte_equity, aportes_mes = deuda_nueva, aporte_nuevo, aportes_new
            break
        deuda, aporte_equity, aportes_mes = deuda_nueva, aporte_nuevo, aportes_new

    # Métricas
    inversion_total = float(inversion_total_base)
    deuda_req_total = float(deuda)
    m1, m2, m3 = st.columns(3)
    m1.metric("Inversión total (one-off)", clp(inversion_total))
    m2.metric("Deuda requerida TOTAL", clp(deuda_req_total))
    apal = (deuda_req_total/inversion_total*100.0) if inversion_total>0 else 0.0
    m3.metric("Apalancamiento (Deuda/Inv)", f"{apal:.1f}%")

    inv_rows = [
        {"Componente":"CapEx","CLP": int(MODEL["capex_total"])},
        {"Componente":"Materiales (compra inicial)","CLP": int(MODEL["materiales_total"])},
        {"Componente":"Capital de trabajo inicial","CLP": int(ct_ini)},
        {"Componente":"Permisos","CLP": int(permisos)},
        {"Componente":"Garantía arriendo","CLP": int(garantia)},
        {"Componente":"Remodelación","CLP": int(remodel)},
    ]
    st.caption("Inversión total = CapEx + Materiales + CT inicial + Permisos + Garantía + Remodelación")
    st.dataframe(pd.DataFrame(inv_rows), use_container_width=True)

    # Mostrar tabla de deuda
    st.subheader("Calendario de deuda")
    st.dataframe(deuda_tabla, use_container_width=True)

    # Aportes sugeridos (ya recalculados arriba)
    st.subheader("Aportes sugeridos mes a mes (mantener caja ≥ 0)")
    iva_p_12  = MODEL["IVA_pagar"].values[:12]
    ebitda_12 = MODEL["EBITDA"].values[:12]
    cuotas_12 = (deuda_tabla["Cuota"].iloc[:12].values if not deuda_tabla.empty else np.zeros(12))
    flujo_mes = ebitda_12 - iva_p_12 - cuotas_12

    # Evolución de caja con aportes
    caja_fin = []
    saldo = float(ct_ini)
    for f, a in zip(flujo_mes, aportes_mes):
        saldo = saldo + f + a
        caja_fin.append(saldo)

    df_aportes = pd.DataFrame({
        "Mes": MESES[:12],
        "EBITDA": ebitda_12,
        "IVA_pagar": iva_p_12,
        "Servicio_deuda": cuotas_12,
        "Flujo_mes": flujo_mes,
        "Aporte_sugerido": aportes_mes,
        "Caja_fin": np.array(caja_fin),
    })
    st.dataframe(
        df_aportes.assign(
            EBITDA=lambda d: d["EBITDA"].map(clp),
            IVA_pagar=lambda d: d["IVA_pagar"].map(clp),
            Servicio_deuda=lambda d: d["Servicio_deuda"].map(clp),
            Flujo_mes=lambda d: d["Flujo_mes"].map(clp),
            Aporte_sugerido=lambda d: d["Aporte_sugerido"].map(clp),
            Caja_fin=lambda d: d["Caja_fin"].map(clp),
        ),
        use_container_width=True
    )
    st.metric("Aporte total (12 meses)", clp(float(aportes_mes.sum())))

    # Guardamos para VAN/TIR
    st.session_state["deuda_tabla"] = deuda_tabla
    st.session_state["aportes_mes_12"] = aportes_mes
    st.session_state["aporte_inicial_equity"] = aporte_equity
    st.session_state["ct_inicial_equity"] = float(ct_ini)


# --- 07 Caja & DSCR ----------------------------------------------------------
with tabs[9]:
    st.subheader("Caja & DSCR (12 meses)")

    ebitda12 = MODEL["EBITDA"].values[:12].astype(float)

    deuda_tab = st.session_state.get("deuda_tabla", pd.DataFrame())
    if deuda_tab is None or (isinstance(deuda_tab, pd.DataFrame) and deuda_tab.empty):
        deuda_tab = MODEL.get("deuda_tabla", pd.DataFrame())

    if deuda_tab is None or deuda_tab.empty:
        servicio12 = np.zeros(12, dtype=float)
        st.info("No hay servicio de deuda configurado para los próximos 12 meses.")
    else:
        servicio12 = deuda_tab["Cuota"].iloc[:12].astype(float).values

    dscr12 = np.divide(ebitda12, np.where(servicio12 == 0, np.nan, servicio12))

    dscr_df = pd.DataFrame({
        "Mes": list(range(1, 13)),
        "EBITDA": ebitda12,
        "Servicio_deuda": servicio12,
        "DSCR": dscr12
    })
    st.dataframe(
        dscr_df.assign(
            EBITDA=lambda d: d["EBITDA"].map(clp),
            Servicio_deuda=lambda d: d["Servicio_deuda"].map(clp),
            DSCR=lambda d: d["DSCR"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        ),
        use_container_width=True
    )
    st.bar_chart(pd.DataFrame({"Mes": dscr_df["Mes"], "DSCR": dscr12}).set_index("Mes"))
    if np.all(np.isnan(dscr12)):
        st.caption("Sin servicio de deuda en los primeros 12 meses (DSCR indefinido).")
        st.metric("DSCR mínimo 12m", "—")
    else:
        st.metric("DSCR mínimo 12m", f"{np.nanmin(dscr12):.2f}")

# --- 08 Descargas ------------------------------------------------------------
with tabs[10]:  # si no aplicaste el ajuste #1 aún, usa el índice donde lo tengas
    st.subheader("Descargar salidas")

    dfs = {
        "Unidades": MODEL["U_df"],
        "Ventas_lineas": MODEL["sku_df"],
        "Ventas_complementarios": s2df(MODEL["ventas_compl_mes"], "CLP"),
        "COGS_mes_sin_compl": s2df(MODEL["COGS_mes_sin_compl"], "CLP"),
        "COGS_complementarios": s2df(MODEL["COGS_compl_mes"], "CLP"),
        "COGS_mes_total": s2df(MODEL["COGS_mes_total"], "CLP"),
        "COGS_mes_neto": s2df(MODEL["COGS_neto"], "CLP"),
        "OPEX_mes_bruto": s2df(MODEL["OPEX_mes"], "CLP"),
        "OPEX_mes_neto": s2df(MODEL["OPEX_neto"], "CLP"),
        "IVA": pd.DataFrame({
            "Mes":MESES,
            "Ventas_brutas":MODEL["ventas_brutas"].values,
            "IVA_debito":MODEL["IVA_debito"].values,
            "IVA_credito_TOTAL":MODEL["IVA_credito"].values,
            "IVA_credito_OPEX":MODEL["IVA_credito_OPEX"].values,
            "IVA_credito_COGS":MODEL["IVA_credito_COGS"].values,
            "IVA_pagar":MODEL["IVA_pagar"].values,
            "Ventas_netas":MODEL["ventas_netas"].values
        }),
        "EBITDA": pd.DataFrame({"Mes":MESES,"EBITDA":MODEL["EBITDA"].values})
    }
    if not st.session_state.get("deuda_tabla", pd.DataFrame()).empty:
        dfs["Amortizacion"] = st.session_state["deuda_tabla"]

    # --- helper con fallback de engine ---
    def _excel_bytes(dfs: dict) -> bytes:
        from io import BytesIO
        bio = BytesIO()
        try:
            # intento 1: xlsxwriter
            with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
                for k, df in dfs.items():
                    df.to_excel(w, sheet_name=str(k)[:31], index=False)
        except Exception:
            # intento 2: openpyxl
            bio = BytesIO()  # reset del buffer
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                for k, df in dfs.items():
                    df.to_excel(w, sheet_name=str(k)[:31], index=False)
        bio.seek(0)
        return bio.getvalue()

    xls_bytes = _excel_bytes(dfs)
    st.download_button("Descargar Excel", data=xls_bytes, file_name="masabroso_modelo.xlsx")

# --- 09 Márgenes & Competencia ----------------------------------------------
with tabs[11]:
    st.subheader("Márgenes unitarios y markup por línea")

    lineas = ["Pan (kg)", "Bolleria", "Pasteleria", "Cafe"]
    rows = []
    for lin in lineas:
        price = float(MODEL["prices"].get(lin, 0.0))
        lin_key = lin.replace(" (kg)", "")
        c_unit = float(MODEL["cost_unit_ins"].get(lin, 0.0)) + float(MODEL["per_unit_map"].get(lin_key, 0.0))
        margen = price - c_unit
        margen_pct = (margen / price * 100.0) if price > 0 else np.nan
        markup_pct = (margen / c_unit * 100.0) if c_unit > 0 else np.nan
        rows.append({
            "Línea": lin,
            "Precio": price,
            "Costo unit.": c_unit,
            "Margen unit.": margen,
            "Margen %": margen_pct,
            "Markup % (sobre costo)": markup_pct
        })

    incluir_pro = st.session_state.get("pro_use_global", True)
    pro_df_global = MODEL.get("sku_pro_df", pd.DataFrame())
    if incluir_pro and isinstance(pro_df_global, pd.DataFrame) and not pro_df_global.empty:
        unidades_pro = float(pd.to_numeric(pro_df_global.get("unidades", pro_df_global.get("Unidades", 0)), errors="coerce").fillna(0.0).sum())
        ingreso_pro = float(pd.to_numeric(pro_df_global.get("ingreso", pro_df_global.get("Venta_mes", 0.0)), errors="coerce").fillna(0.0).sum())
        cogs_pro = float(pd.to_numeric(pro_df_global.get("cogs", 0.0), errors="coerce").fillna(0.0).sum()) if "cogs" in pro_df_global.columns else np.nan
        precio_prom = (ingreso_pro / unidades_pro) if unidades_pro > 0 else 0.0
        costo_prom = (cogs_pro / unidades_pro) if unidades_pro > 0 and not np.isnan(cogs_pro) else np.nan
        margen_unit = (precio_prom - costo_prom) if not np.isnan(costo_prom) else np.nan
        margen_pct = ((margen_unit / precio_prom) * 100.0) if precio_prom > 0 and not np.isnan(margen_unit) else np.nan
        markup_pct = ((margen_unit / costo_prom) * 100.0) if not np.isnan(costo_prom) and costo_prom > 0 and not np.isnan(margen_unit) else np.nan
        rows.append({
            "Línea": "PRO",
            "Precio": precio_prom,
            "Costo unit.": costo_prom,
            "Margen unit.": margen_unit,
            "Margen %": margen_pct,
            "Markup % (sobre costo)": markup_pct,
        })

    df_marg = pd.DataFrame(rows)
    st.dataframe(
        df_marg.assign(
            Precio=lambda d: d["Precio"].map(clp),
            **{"Costo unit.": lambda d: d["Costo unit."].map(clp)},
            **{"Margen unit.": lambda d: d["Margen unit."].map(clp)},
            **{"Margen %": lambda d: d["Margen %"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")},
            **{"Markup % (sobre costo)": lambda d: d["Markup % (sobre costo)"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")},
        ),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Competencia – precio por kg de pan (Barrio Italia)")
    st.caption("Rangos de referencia armados con precios observados en panaderías del sector Barrio Italia.")

    precio_pan_kg = float(MODEL["prices"].get("Pan (kg)", 0.0))
    st.metric("Nuestro precio Pan (kg)", clp(precio_pan_kg))

    segmentos = [
        ("Tradicional de barrio",          2000, 2500),
        ("Artesanal 'precio barrio'",      2500, 3000),
        ("Francesa a 'precio justo'",      2500, 3200),
        ("Artesanal premium / masa madre", 5000, 7000),
    ]

    comp_rows = []
    match_seg = None
    for seg, lo, hi in segmentos:
        en_rango = (lo <= precio_pan_kg <= hi)
        if en_rango:
            match_seg = seg
        comp_rows.append({
            "Segmento": seg,
            "Precio mín": lo,
            "Precio máx": hi,
            "Punto medio": (lo + hi) / 2.0,
            "¿En rango?": "Sí" if en_rango else "No",
        })

    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(
        df_comp.assign(
            **{"Precio mín": lambda d: d["Precio mín"].map(clp)},
            **{"Precio máx": lambda d: d["Precio máx"].map(clp)},
            **{"Punto medio": lambda d: d["Punto medio"].map(clp)},
        ),
        use_container_width=True
    )

    if match_seg:
        st.success(f"Tu precio de Pan (kg) está **en línea** con el segmento: **{match_seg}**.")
    else:
        lo_min = min(s[1] for s in segmentos)
        hi_max = max(s[2] for s in segmentos)
        if precio_pan_kg < lo_min:
            st.warning("Tu precio de Pan (kg) está **por debajo** del rango tradicional del sector.")
        elif precio_pan_kg > hi_max:
            st.warning("Tu precio de Pan (kg) está **por encima** del rango premium del sector.")
        else:
            st.info("Tu precio cae en un intermedio entre segmentos (no coincide exactamente con un rango definido).")

    with st.expander("Notas"):
        st.markdown(
            "- **Margen %** = (Precio − Costo) / Precio\n"
            "- **Markup %** = (Precio − Costo) / Costo\n"
            "- Los rangos de mercado son de referencia local para Barrio Italia."
        )
# --- 11 – VAN & TIR ----------------------------------------------------------
with tabs[12]:
    st.subheader("VAN & TIR (flujo a equity, mensual)")

    # ======= Parámetros =========
    cA, cB = st.columns(2)
    ke_anual = cA.number_input("Tasa de descuento anual (Ke, %)", 0.0, 200.0, 20.0, 0.1, key="van_ke_anual")
    horizonte = cB.number_input("Horizonte (meses)", 1, 60, 12, 1, key="van_horizonte")

    c1, c2, c3, c4 = st.columns([1,1,1,1.6])
    dso = c1.number_input("DSO (días de cobro)", 0, 120, 0, 1, key="van_dso")
    dio = c2.number_input("DIO (días de inventario)", 0, 120, 7, 1, key="van_dio")
    dpo = c3.number_input("DPO (días de pago a prov.)", 0, 120, 30, 1, key="van_dpo")
    usar_ct = c4.checkbox("Modelar capital de trabajo (DSO/DIO/DPO)", True, key="van_usar_ct")

    # ======= Series base (primeros 'h' meses) =========
    h = int(horizonte)
    km = (1.0 + ke_anual/100.0) ** (1.0/12.0) - 1.0  # tasa mensual

    ebitda_m = MODEL["EBITDA"].values[:h].astype(float)
    iva_p_m  = MODEL["IVA_pagar"].values[:h].astype(float)

    deuda_tab = st.session_state.get("deuda_tabla", pd.DataFrame())
    if deuda_tab is None or deuda_tab.empty:
        servicio_m = np.zeros(h, dtype=float)
    else:
        servicio_m = deuda_tab["Cuota"].iloc[:h].astype(float).values

    # Flujo operativo a equity SIN CT
    base_cf_m = ebitda_m - iva_p_m - servicio_m

    # ======= Ajuste por Capital de Trabajo (ΔWC) =======
    if usar_ct:
        ventas = MODEL["ventas_netas"].values[:h].astype(float)
        cogs_neto = (
            MODEL["COGS_neto"].values[:h].astype(float)
            if "COGS_neto" in MODEL
            else MODEL["COGS_mes_total"].values[:h].astype(float)
        )

        # Saldos estimados de AR, Inventario y AP (mes a mes)
        AR  = ventas    * (dso/30.0)
        INV = cogs_neto * (dio/30.0)
        AP  = cogs_neto * (dpo/30.0)
        WC  = AR + INV - AP

        # ΔWC_t = WC_t − WC_{t−1}, con WC_{0}=0
        dWC = np.diff(np.concatenate([[0.0], WC]))
    else:
        dWC = np.zeros(h, dtype=float)

    # Aportes/retiradas de equity (si configuraste en la pestaña de deuda)
    aportes_mes = st.session_state.get("aportes_mes_12", np.zeros(12, dtype=float))[:h]
    aporte_inicial_equity = float(st.session_state.get("aporte_inicial_equity", 0.0))
    ct_inicial_equity     = float(st.session_state.get("ct_inicial_equity", 0.0))

    # Flujo a equity mensual (positivo = entra a la caja del equity)
    cf_m = base_cf_m - dWC - aportes_mes  # los aportes son dinero puesto por el equity → restan al flujo

    # Serie completa de flujos incluyendo el t=0
    flujos = np.concatenate([[-(aporte_inicial_equity + ct_inicial_equity)], cf_m.astype(float)])

    # VAN (VPN) con tasa mensual km
    descuentos = np.array([(1.0 + km) ** t for t in range(len(flujos))], dtype=float)
    VAN = float(np.sum(flujos / descuentos))

    # TIR (mensual) + anualizada
    def _irr(cashflows: np.ndarray) -> float:
        try:
            import numpy_financial as npf  # si está disponible, úsalo
            return float(npf.irr(cashflows))
        except Exception:
            # Newton-Raphson simple como fallback
            r = 0.1
            for _ in range(100):
                denom = np.array([(1+r)**t for t in range(len(cashflows))], dtype=float)
                f  = float(np.sum(cashflows/denom))
                df = 0.0
                for t in range(1, len(cashflows)):
                    df += -t * cashflows[t] / ((1+r) ** (t+1))
                if abs(df) < 1e-12:
                    break
                r2 = r - f/df
                if not np.isfinite(r2):
                    break
                if abs(r2 - r) < 1e-8:
                    r = r2
                    break
                r = r2
            return float(r)

    TIR_m = _irr(flujos)
    TIR_a = (1.0 + TIR_m) ** 12 - 1.0 if np.isfinite(TIR_m) else np.nan

    # Tabla de soporte
    df_van = pd.DataFrame({
        "Mes": ["t=0"] + list(range(1, h+1)),
        "Flujo_equity": flujos,
        "Factor_descuento": descuentos,
        "Flujo_descuento": flujos / descuentos
    })

    st.markdown("### Flujos a equity (mensual)")
    show_df_money(
        df_van.assign(
            Flujo_equity=lambda d: d["Flujo_equity"].map(clp),
            Flujo_descuento=lambda d: d["Flujo_descuento"].map(clp),
        ),
        money_cols=["Flujo_equity", "Flujo_descuento"],
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("VAN (mes 0…h)", clp(VAN))
    c2.metric("TIR mensual", f"{(TIR_m*100.0):.2f}%" if np.isfinite(TIR_m) else "—")
    c3.metric("TIR anualizada", f"{(TIR_a*100.0):.2f}%" if np.isfinite(TIR_a) else "—")

    with st.expander("Detalle de supuestos usados"):
        st.write(f"- Horizonte: **{h}** meses")
        st.write(f"- Ke anual: **{ke_anual:.2f}%** → tasa mensual **{km*100:.3f}%**")
        st.write(f"- DSO/DIO/DPO: **{dso}/{dio}/{dpo}** días (usar CT: **{usar_ct}**)")

