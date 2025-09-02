--
# anet-lite

壊れる瞬間を、言葉にするノイズ指紋監視キット  
**Noise is trust. Accident is the only log that never lies.**

---

## 概要
AIや機械学習モデルが“壊れる瞬間”を数値と言葉に落とすためのミニマル監視ツール。  
ノイズ指紋・一致度ϑ・総合スコアF_Total・二秒前駆アラートで「事故の前触れ」を検知・記録・説明できる。  
1行追加で既存ループに差し込める。“むずかしい理屈”より、“すぐ動く”が正義。

---

## 主な特徴

- **一致度ϑ**: 正常運転時ノイズと現在ノイズの“指紋一致度”で安定/崩壊の度合いを測る
- **F_Total**: エントロピー変動とプロセス効率の合成スコア。実運用の決定指標
- **二秒前駆アラート**: 事故直前の“ゆらぎ”をzスコアでアラート
- **墓標ログ**: 忘却/再学習時の説明責任を自動記録（JSON出力）

---

## インストール

```bash
pip install anet-lite
````

---

## クイックスタート

```python
from anet_lite import theta_consistency, entropy_rate, processivity, f_total, two_second_precursor

loss_hist     = [...]  # 損失や温度の履歴
grad_norms    = [...]  # 勾配ノルム履歴
step_norms    = [...]  # パラメタ更新ノルム
noise_series  = [...]  # 勾配ノイズ時系列
ref_series    = [...]  # 正常運転の基準系列

phi = theta_consistency(noise_series, ref_series)     # 0..1（1=正常）
F   = f_total(entropy_rate(loss_hist), processivity(grad_norms, step_norms), w_s=0.7, w_p=0.3)
alert, z = two_second_precursor(loss_hist, rate_hz=5) # 2秒窓の異常度（z>2.5で要注意）
print(phi, F, alert, z)
```

---

## 主要API

* `theta_consistency(noise_seq, ref_seq, beta=3.0)`
  　ノイズ指紋の一致度（0〜1）。1に近いほど正常。
* `entropy_rate(values, window=32)`
  　時系列エントロピー差分。>0で混乱増、<0で収束。
* `processivity(grad_norms, step_norms)`
  　勾配とパラメータ更新の整合度（-1〜1）。
* `f_total(s_dot, p, w_s=1.0, w_p=1.0)`
  　Ṡ（エントロピー変化）×P（プロセス効率）の合成スコア。
* `two_second_precursor(series, rate_hz=10, window_sec=2.0, z=2.5)`
  　2秒窓でzスコア超の異常を返す。

---

## 用途例

* 学習暴走・崩壊の早期減速（学習率制御/LoRA監視）
* GANの“交配”時に危険境界を見抜く
* 忘却/再学習時の説明責任・証跡ログ化
* PyTorch学習ループ、LoRAトレーニング、論文実験の異常検知
* “壊れ方”を美学として記録したい全ての人類

---

## 墓標ログサンプル

```json
{
  "event": "forget",
  "time": "2025-09-02T12:34:56Z",
  "policy": "ttl|compliance|drift",
  "scope": {"layers": ["L12,L13"], "datasets": ["foo"]},
  "evidence": {"theta": 0.42, "f_total": -0.8, "precursor_z": 3.1},
  "operator": "auto",
  "hash": "sha256:... (immutable)"
}
```

---

## ライセンス

MIT License
Questions / Issues: [GitHub Issues](https://github.com/Risa-Sawamura/anet-lite/issues)
X(Twitter): [@risa-sawamura](https://twitter.com/)

---

**“事故を言葉にできる時代、それがAIの夜明け。”**

---

> **an open-source, accident-aware OS for noise-centric trust.**

--
