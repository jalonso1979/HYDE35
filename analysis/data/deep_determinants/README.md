# Deep determinants — paper 5 substrate data

Five substrate-layer parquets feed the horserace panel:

- `predicted_het_pw_adjusted.parquet` — Ashraf-Galor (2013, AER) predicted heterozygosity, ancestry-adjusted via Putterman-Weil (2010) migration weights. iso3 + H_pred + H_pred_pwadj + provenance.
- `ancestral_crop_yield.parquet` — Galor-Özak (2016, AER) gridded prehistoric crop-yield potential aggregated to ISO3 weighted by HYDE 1500 cropland mask. iso3 + ancestral_yield + provenance.
- `state_history_pw.parquet` — Putterman-Weil (2010, QJE) state-history index 0-1500 CE, ancestry-adjusted. iso3 + state_hist + state_hist_pwadj + provenance.
- `pandemic_intensity_pre1500.parquet` — pre-1500 pandemic exposure intensity from Brecke + AntiquityPandemics + Justinianic + Antonine + Cyprian reconstructions. iso3 + pandemic_intensity + provenance.
- `modern_outcomes.parquet` — modern demographic outcomes 1950-2025 (pop growth, urbanisation change, log GDPpc 2015, demographic-transition timing). iso3 + 4 outcomes + sources.

The master joined panel lives at `analysis/data/deep_determinants_horserace.parquet`.
See `docs/superpowers/specs/2026-05-18-deep-determinants-horserace-design.md` for substrate definitions.
