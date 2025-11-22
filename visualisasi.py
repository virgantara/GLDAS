# ============================================================
# GRACE vs GLDAS EWH Time Series (2005–2025)
# Full Script from Reading Data to Plotting (Interval 2 Tahun)
# ============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 1. LOAD DATA GRACE (0.5°)
# ============================================================
BASE_DIR = 'data'
grace = pd.read_csv(os.path.join(BASE_DIR, "Input_R_GRACE-GLDAS_2005-2025_0_5.csv"),
                    delimiter=',')

grace['time'] = pd.to_datetime(grace['time'])

# GRACE scaled_lwe_cm = TWSA = EWH (cm)
# (SUDAH DALAM CM → tidak perlu dikonversi)
grace['GRACE_EWH_cm'] = grace['scaled_lwe_cm']


# ============================================================
# 2. DEFINISI WARNA (optional)
# ============================================================

colors_rdylbu = {
    'JPL': '#74ADD1',
    'CSR': '#A50026',
    'GSFC': '#FDAE61'
}

# ============================================================
# 2. LOAD DATA GLDAS (0.25° / 0.5° after resample)
# ============================================================

gldas = pd.read_csv(os.path.join(BASE_DIR, "Input_R_GLDAS_2005-2025_0_25.csv"),
                    delimiter=',')

gldas['time'] = pd.to_datetime(gldas['time'])

# Convert GLDAS kg/m2 → cm (1 kg/m² = 1 mm air = 0.1 cm)
gldas['SMSA_cm'] = gldas['SMSA'] * 0.1
gldas['PCSA_cm'] = gldas['PCSA'] * 0.1

# Represent GLDAS TWSA (total water storage anomaly) in cm
gldas['GLDAS_TWSA_cm'] = gldas['SMSA_cm'] + gldas['PCSA_cm']


# ============================================================
# 3. HITUNG RATA-RATA GRID PER BULAN
# ============================================================

grace_ts = (
    grace.groupby("time")['GRACE_EWH_cm']
    .mean()
    .reset_index()
)

gldas_ts = (
    gldas.groupby("time")['GLDAS_TWSA_cm']
    .mean()
    .reset_index()
)

# ============================================================
# 4. MERGE DATA BERDASARKAN TIME
# ============================================================

combined = pd.merge(grace_ts, gldas_ts, on="time", how="inner")

# Filter 2005–2025
combined = combined[
    (combined["time"] >= "2005-01-01") &
    (combined["time"] <= "2025-06-30")
]


# ============================================================
# 5. PLOTTING GRACE vs GLDAS (EWH cm) — INTERVAL 2 TAHUN
# ============================================================

plt.figure(figsize=(16,6))

# plt.plot(combined['time'], combined['GRACE_EWH_cm'],
#          label='GRACE JPL',
#          color=colors_rdylbu['CSR'],
#          linewidth=2,
#          marker='o',
#          markersize=4,
#          markerfacecolor=colors_rdylbu['CSR'],
#          markeredgecolor=colors_rdylbu['CSR'],
#          linestyle='-')

plt.plot(combined['time'], combined['GLDAS_TWSA_cm'],
         label='GLDAS',
         color=colors_rdylbu['GSFC'],
         linewidth=2,
         marker='o',
         markersize=4,
         markerfacecolor=colors_rdylbu['GSFC'],
         markeredgecolor=colors_rdylbu['GSFC'],
         linestyle='-')

plt.xlabel("Time (Year)")
plt.ylabel("Equivalent Water Height (cm)")
plt.title("Comparison of GRACE JPL and GLDAS in East Java Province (2005–2025)")

# === INTERVAL X-AXIS SETIAP 2 TAHUN ===
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(base=2))   # interval 2 tahun
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # format tahun
plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Plot GRACE vs GLDAS (EWH cm) 2005–2025 berhasil dibuat.")
