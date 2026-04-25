# permABC: Setup và hướng dẫn chạy my-reproduces

[![arXiv](https://img.shields.io/badge/arXiv-2507.06037-b31b1b.svg)](https://www.arxiv.org/abs/2507.06037)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tài liệu này tập trung vào cách setup và chạy các script trong thư mục `my-reproduces/` để tái lập các thí nghiệm simulation và real-world weather.

## 1) Yêu cầu

- Python 3.9+ (khuyến nghị 3.10/3.11)
- pip mới


## 2) Setup môi trường

Từ thư mục gốc repository:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Kiểm tra import nhanh:

```bash
python3 -c "import permabc; print(permabc.get_version())"
```

## 3) Cấu trúc thư mục my-reproduces

- `my-reproduces/data/`: dữ liệu đầu vào (vd: `df_weather_clean.csv`)
- `my-reproduces/results/`: nơi lưu kết quả CSV/JSON/figure
- `my-reproduces/scripts_simulation/`: benchmark simulation
- `my-reproduces/scripts_real_world/`: inference trên dữ liệu thời tiết thực

## 4) Chạy benchmark simulation

### 4.1 Chạy benchmark chính

Script:
- `my-reproduces/scripts_simulation/run_performance_comparison_simulation.py`

Ví dụ full benchmark:

```bash
python3 my-reproduces/scripts_simulation/run_performance_comparison_simulation.py \
  --methods all \
  --K 20 \
  --K-outliers 4 \
  --seed 42 \
  --prefix perf_compare_like_experiments_K20
```

Ví dụ smoke test nhanh:

```bash
python3 my-reproduces/scripts_simulation/run_performance_comparison_simulation.py \
  --methods all \
  --K 8 \
  --K-outliers 2 \
  --seed 42 \
  --n-points 10000 \
  --n-particles 200 \
  --n-sim-budget 100000 \
  --n-epsilon 1000 \
  --m0-values 12 16 24 \
  --l0-values 2 4 6 8 \
  --prefix smoke_perf_compare_all_fast
```

Tùy chọn quan trọng:
- `--methods`: `all|vanilla|smc|pmc|osum`
- `--output-dir` (mặc định: `my-reproduces/results/simulation`)
- `--plot-after-run` / `--no-plot-after-run`

### 4.2 Vẽ lại hình từ CSV đã có

Script:
- `my-reproduces/scripts_simulation/plot_performance_comparison_simulation.py`

Ví dụ:

```bash
python3 my-reproduces/scripts_simulation/plot_performance_comparison_simulation.py \
  --prefix perf_compare_like_experiments_K20
```

Hoặc chỉ định CSV cụ thể:

```bash
python3 my-reproduces/scripts_simulation/plot_performance_comparison_simulation.py \
  --csv my-reproduces/results/simulation/perf_compare_like_experiments_K20.csv
```

## 5) Chạy real-world weather inference

### 5.1 Kiểm tra model Bernoulli-logit trước khi chạy dài

Script:
- `my-reproduces/scripts_real_world/validate_bernoulli_model.py`

```bash
python3 my-reproduces/scripts_real_world/validate_bernoulli_model.py
```

### 5.2 Chạy inference trên dữ liệu thời tiết

Script:
- `my-reproduces/scripts_real_world/run_weather_rain_probability.py`

Ví dụ đầy đủ:

```bash
python3 my-reproduces/scripts_real_world/run_weather_rain_probability.py \
  --seed 42 \
  --scales national regional provincial \
  --n_particles 1000 \
  --final_iteration 100 \
  --num_gibbs_blocks 3 \
  --gibbs_T 1000 \
  --gibbs_M_loc 50 \
  --gibbs_M_glob 100 \
  --max_days 0
```

Tùy chọn quan trọng:
- `--data_csv` (mặc định: `my-reproduces/data/df_weather_clean.csv`)
- `--results_dir` (mặc định: `my-reproduces/results/weather_rain_inference`)
- `--methods` để chọn subset phương pháp

### 5.3 Vẽ hình từ kết quả đã lưu

Script:
- `my-reproduces/scripts_real_world/plot_weather_rain_probability.py`

Ví dụ:

```bash
python3 my-reproduces/scripts_real_world/plot_weather_rain_probability.py \
  --seed 42 \
  --scales national regional provincial
```

Thêm diagnostic figure:

```bash
python3 my-reproduces/scripts_real_world/plot_weather_rain_probability.py \
  --seed 42 \
  --scale regional \
  --include_extra
```

## 6) Thư mục kết quả

Sau khi chạy, kết quả thường nằm ở:
- `my-reproduces/results/simulation/`
  - `*.csv`, `*_summary.json`, `figures_performance/*.png`
- `my-reproduces/results/weather_rain_inference/`
  - file kết quả theo method/scale/seed
  - `comparison_weather_rain_seed_*.csv`
  - `figures/*.png`

## 7) Lỗi thường gặp

- Không import được `permabc`:
  - đảm bảo đã chạy `pip install -e .`
- Lỗi khi vẽ hình trên môi trường headless:
  - các script đã set backend phù hợp (`Agg`), thử chạy lại trong shell sạch
- Chạy chậm hoặc memory cao:
  - giảm `--n-particles`, `--n-points`, `--n-epsilon`

## Citing permABC

If you use `permABC` in your research, please cite our paper:

```bibtex
@misc{luciano2025permutationsaccelerateapproximatebayesian,
      title={Permutations accelerate Approximate Bayesian Computation}, 
      author={Antoine Luciano and Charly Andral and Christian P. Robert and Robin J. Ryder},
      year={2025},
      eprint={2507.06037},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2507.06037}, 
}
```
