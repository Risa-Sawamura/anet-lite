# anet-lite

壊れる瞬間を、言葉にするノイズ指紋監視キット  
**Noise is trust. Accident is the only log that never lies.**

[日本語(JA)](#ja-概要) • [English(EN)](#en-overview)

---

## JA: 概要  {#ja-概要}

**anet-lite** は、AI/機械学習モデルが “壊れる瞬間” をノイズの指紋として**検知・記録・説明**するミニマルな監視ツールです。  
以下の3本柱を既存の学習/推論ループに挿すだけで動きます。

- **一致度 ϑ (theta)**：正常運転と現在の**ノイズ指紋**の一致度（0..1）
- **F_Total**：エントロピー変化 **Ṡ** と **Processivity P**（勾配→更新の整合）を合成した運用スコア
- **二秒前駆アラート**：崩壊直前の “ゆらぎ” を z スコアで逐次検出

**ねらい**：事故を「数値→言葉→意思決定」に橋渡しすること。面倒な依存はなく、まず “動く” を優先しています。

---

## EN: Overview  {#en-overview}

**anet-lite** is a minimal monitoring toolkit that **detects, records, and explains** the moment your ML model starts to fail, using **noise fingerprints**.

Core metrics, drop-in:

- **Consistency ϑ (theta):** similarity (0..1) between a reference noise fingerprint and the current one  
- **F_Total:** operational score combining **entropy rate Ṡ** and **processivity P** (how well gradients translate into parameter updates)  
- **Two-second precursor alert:** sequential detection of pre-collapse jitter via z-score

**Goal:** turn accidents into numbers, numbers into words, and words into decisions. Minimal dependencies; prioritizes “works first”.

---

## JA: 主な特徴

- **ノイズ一次市民**：勾配ノイズ/更新系列の周波数指紋で “安定/漂流” を即判定  
- **F_Total ひと目**：Ṡ↑=混乱増、P↑=仕事能増。加重合成で現場の判断を単純化  
- **二秒前駆**：落ちる前の“揺れ”を短窓で拾う（逐次・オンライン）  
- **墓標ログ（任意）**：忘却/ロールバックの**理由・範囲・証拠・ハッシュ**を自動記録  
- **5分導入**：学習ループに2フック、外部DB不要。CSV/JSONで吐けます

## EN: Key features

- **Noise-first:** spectral fingerprints of gradient/update streams for instant stable/drifting judgment  
- **One-look F_Total:** Ṡ up → more chaos, P up → more work done; weighted combination for ops decisions  
- **Two-second precursor:** online, short-window jitter detection before collapse  
- **Tombstone logs (optional):** automatic records of *reason/scope/evidence/hash* for forgetting/rollback  
- **5-minute setup:** two hooks in your training loop; no external DB; CSV/JSON outputs

---

## JA: インストール / EN: Installation

```bash
pip install anet-lite
# or: pip install -e .   # local editable install
````

---

## JA: クイックスタート / EN: Quick start (10 lines)

```python
from anet_lite import theta_consistency, entropy_rate, processivity, f_total, two_second_precursor

loss_hist     = [...]  # JA: 損失/温度の履歴 | EN: loss/temperature history
grad_norms    = [...]  # JA: 勾配ノルム     | EN: gradient norms
step_norms    = [...]  # JA: 更新ノルム     | EN: parameter update norms
noise_series  = [...]  # JA: 勾配ノイズ列   | EN: gradient noise series
ref_series    = [...]  # JA: 正常参照列     | EN: reference (healthy) series

phi = theta_consistency(noise_series, ref_series)     # 0..1 (1=healthy)
F   = f_total(entropy_rate(loss_hist), processivity(grad_norms, step_norms), w_s=0.7, w_p=0.3)
alert, z = two_second_precursor(loss_hist, rate_hz=5) # True if z>2.5
print(phi, F, alert, z)
```

---

## JA: PyTorch連携最小例 / EN: Minimal PyTorch hook

```python
import torch
from anet_lite import snapshot  # returns theta/sdot/proc/F/precursor

class GradNoiseTap:
    def __init__(self, model, ema=0.9):
        self.model, self.ema = model, ema
        self.m = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
    def norm(self):
        acc = 0.0
        for n,p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None: continue
            g = p.grad.detach()
            self.m[n].mul_(self.ema).add_(g, alpha=1-self.ema)
            acc += float((g - self.m[n]).norm().cpu().item())
        return acc

logs = {'loss':[], 'g':[], 's':[]}
model = ...; opt = torch.optim.AdamW(model.parameters(), 3e-4)
tap = GradNoiseTap(model); ref = []

for t, batch in enumerate(loader, 1):
    opt.zero_grad(); loss = model(batch).loss; loss.backward()
    gnorm = tap.norm()
    snorm = sum(float((p.grad**2).sum().sqrt().cpu()) for p in model.parameters() if p.grad is not None)
    opt.step()
    logs['loss'].append(float(loss)); logs['g'].append(gnorm); logs['s'].append(snorm)
    if t < 100: ref.append(gnorm); continue
    snap = snapshot(loss_hist=logs['loss'][-64:], grad_norms=logs['g'][-64:],
                    step_norms=logs['s'][-64:], noise_seq=logs['g'][-64:], ref_seq=ref[-64:])
    if t % 20 == 0:
        print(f"[{t}] θ={snap.theta:.2f} F={snap.f_total:+.2f} pre={snap.precursor} z={snap.zscore:+.2f}")
    if snap.precursor or snap.theta < 0.6:
        for g in opt.param_groups: g['lr'] *= 0.8   # slow down gently
```

---

## JA: API 要点 / EN: Minimal API

* `theta_consistency(noise_seq, ref_seq, beta=3.0) -> float`
  **JA:** 0..1 の一致度（1=一致）| **EN:** similarity 0..1 (1=match)
* `entropy_rate(values, window=32) -> float`
  **JA:** 直近と一つ前の窓のエントロピー差 Ṡ | **EN:** entropy rate between last two windows
* `processivity(grad_norms, step_norms) -> float`
  **JA:** 勾配と更新の整合（-1..1）| **EN:** gradient-to-update alignment
* `f_total(s_dot, p, w_s=1.0, w_p=1.0) -> float`
  **JA/EN:** F\_Total = w\_s·Ṡ + w\_p·P
* `two_second_precursor(series, rate_hz=10, window_sec=2.0, z=2.5) -> (bool, float)`
  **JA:** 2秒窓の z スコア判定 | **EN:** 2-sec z-score alert

---

## JA: 使いどころ / EN: Use cases

* **学習暴走の早期減速 / Early slowdown for runaway training**
* **交配/マージの危険境界可視化 / Safer model merging boundaries**
* **署名保全アップスケール / Signature-preserving super-resolution**
* **忘却の説明責任ログ / Explainable forgetting tombstones**

---

## JA: 墓標ログサンプル / EN: Tombstone log example

```json
{
  "event": "forget",
  "time": "2025-09-02T12:34:56Z",
  "policy": "ttl|compliance|drift",
  "scope": {"layers": ["L12","L13"], "datasets": ["foo"]},
  "evidence": {"theta": 0.42, "f_total": -0.8, "precursor_z": 3.1},
  "operator": "auto",
  "hash": "sha256:... (immutable)"
}
```

---

## JA: 互換性 / EN: Compatibility

* Python 3.9+ / NumPy (PyTorch optional)
* CPUで動作（GPU不要）。Windows/macOS/Linux

---

## JA: よくある質問 / EN: FAQ

**Q(JA): ϑが0.8→0.6へ落ちた。止めるべき？**
**A:** まず学習率/温度/LoRAランクを段階的に下げる。二秒前駆（z>2.5）が同時ならセーフティへ。

**Q(EN): How do I build the reference series?**
**A:** Use a window from a *known-good* period. Refresh slowly to avoid chasing the drift.

**Q(JA): 指標を増やせば強くなる？**
**A:** 相関が高い指標を増やしても実質1つ。直交性を確認してから追加。

---

## JA: 貢献 / EN: Contributing

Issues/Pull Requests 歓迎。**失敗録の共有**は最高の貢献です。
We welcome bug reports, feature requests, and failure diaries. Be kind; bring data.

---

## JA/EN: ライセンス / License

MIT License

---

## JA/EN: 連絡 / Contact

* GitHub Issues: [https://github.com/Risa-Sawamura/anet-lite/issues](https://github.com/Risa-Sawamura/anet-lite/issues)
* X(Twitter)

```
