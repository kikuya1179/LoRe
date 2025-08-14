import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np


def set_japanese_font() -> None:
	# Try common JP fonts on Windows/macOS/Linux
	candidates = [
		"Yu Gothic",
		"Yu Gothic UI",
		"Meiryo",
		"MS Gothic",
		"MS Mincho",
		"Noto Sans CJK JP",
		"Noto Serif CJK JP",
		"Hiragino Sans",
		"Hiragino Kaku Gothic ProN",
	]
	available = {f.name for f in font_manager.fontManager.ttflist}
	for name in candidates:
		if name in available:
			rcParams["font.family"] = name
			rcParams["axes.unicode_minus"] = False
			return
	# Fallback: keep default but avoid minus glyph issue
	rcParams["axes.unicode_minus"] = False


def parse_success_rates(log_path: str) -> Tuple[List[int], List[float], List[int], List[float]]:
	"""Parse success_rate time series for RL-only and LoRe (LLM) segments.

	We split the log into two segments by detecting the first occurrence of
	"[LLM] Controller initialized". Lines of interest look like:
	  step=250 episode=1 return=0.00 success_rate=0.000
	"""
	steps_rl: List[int] = []
	rates_rl: List[float] = []
	steps_lore: List[int] = []
	rates_lore: List[float] = []

	llm_mode = False
	step_re = re.compile(r"step=(\d+).*?success_rate=([0-9.]+)")

	with open(log_path, "r", encoding="utf-8") as f:
		for line in f:
			if (not llm_mode) and ("[LLM] Controller initialized" in line):
				llm_mode = True
				continue
			m = step_re.search(line)
			if not m:
				continue
			step = int(m.group(1))
			rate = float(m.group(2))
			if llm_mode:
				steps_lore.append(step)
				rates_lore.append(rate)
			else:
				steps_rl.append(step)
				rates_rl.append(rate)

	return steps_rl, rates_rl, steps_lore, rates_lore


def cutoff_series(steps: List[int], rates: List[float], max_step: int) -> Tuple[List[int], List[float]]:
	cut_steps: List[int] = []
	cut_rates: List[float] = []
	for s, r in zip(steps, rates):
		if s <= max_step:
			cut_steps.append(s)
			cut_rates.append(r)
	return cut_steps, cut_rates


def rolling_median_se(x: List[int], y: List[float], window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Rolling median with standard error band (median ± SE) over a +/- window/2.

	SE is computed as std(vals)/sqrt(n) within the window.
	"""
	if not x:
		return np.array([]), np.array([]), np.array([])
	x_arr = np.array(x)
	y_arr = np.array(y)
	grid = np.array(sorted(set(x_arr.tolist())))
	half = window // 2
	med = np.zeros_like(grid, dtype=float)
	low = np.zeros_like(grid, dtype=float)
	high = np.zeros_like(grid, dtype=float)
	for i, gx in enumerate(grid):
		lo = gx - half
		hi = gx + half
		mask = (x_arr >= lo) & (x_arr <= hi)
		vals = y_arr[mask]
		if vals.size == 0:
			med[i] = np.nan
			low[i] = np.nan
			high[i] = np.nan
			continue
		med[i] = np.median(vals)
		se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
		low[i] = med[i] - se
		high[i] = med[i] + se
	return grid, med, np.vstack([low, high])


def main() -> None:
	set_japanese_font()

	here = os.path.dirname(__file__)
	log_path = os.path.join(here, "result.txt")
	out_path = os.path.join(here, "sr_compare.png")

	steps_rl, rates_rl, steps_lore, rates_lore = parse_success_rates(log_path)

	max_step = 10000
	steps_rl, rates_rl = cutoff_series(steps_rl, rates_rl, max_step)
	steps_lore, rates_lore = cutoff_series(steps_lore, rates_lore, max_step)

	# Rolling statistics
	window = 1000
	gx_rl, med_rl, band_rl = rolling_median_se(steps_rl, rates_rl, window)
	gx_lo, med_lo, band_lo = rolling_median_se(steps_lore, rates_lore, window)

	plt.figure(figsize=(8.0, 4.5), dpi=160)
	ax = plt.gca()
	# Highlight evaluation window 10k±250
	eval_c = 10000
	eval_w = 250
	ax.axvspan(eval_c - eval_w, eval_c + eval_w, color="#cccccc", alpha=0.25, label="評価窓 10k±250")

	# Raw curves
	if steps_rl:
		plt.plot(steps_rl, rates_rl, label="RLのみ (raw)", color="#1f77b4", linewidth=1.1, alpha=0.6)
	if steps_lore:
		plt.plot(steps_lore, rates_lore, label="LoRe (raw)", color="#d62728", linewidth=1.1, alpha=0.6)
	# Medians and SE bands
	if gx_rl.size:
		plt.fill_between(gx_rl, band_rl[0], band_rl[1], color="#1f77b4", alpha=0.18, linewidth=0)
		plt.plot(gx_rl, med_rl, color="#1f77b4", linewidth=2.2, label="RLのみ 中央値±SE (1k)")
	if gx_lo.size:
		plt.fill_between(gx_lo, band_lo[0], band_lo[1], color="#d62728", alpha=0.18, linewidth=0)
		plt.plot(gx_lo, med_lo, color="#d62728", linewidth=2.2, label="LoRe 中央値±SE (1k)")

	plt.xlabel("Steps")
	plt.ylabel("Success Rate (SR)")
	plt.title("DoorKey-5×5: 成功率 (seed=42)")
	plt.grid(True, linestyle=":", alpha=0.6)
	plt.xlim(0, max_step)
	plt.ylim(0, max(0.21, max(rates_rl + rates_lore) if (rates_rl or rates_lore) else 0.2))
	plt.legend(loc="upper right", frameon=False, fontsize=9)
	plt.tight_layout()
	plt.savefig(out_path)
	print(f"Saved figure to {out_path}")


if __name__ == "__main__":
	main()
