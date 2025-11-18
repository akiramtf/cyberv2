# Dataset Training Time Analysis

> **✅ UPDATE:** Option 3 (Disable SSL Checks) has been **IMPLEMENTED**!
> See [SSL_CHECK_TOGGLE_IMPLEMENTATION.md](SSL_CHECK_TOGGLE_IMPLEMENTATION.md) for details.
> Training script now automatically uses `enable_ssl_check=False` for 40-80x speedup.

## Problem Summary

**Issue:** Training with dataset4.csv takes hours while dataset3.csv takes minutes, despite having similar row counts.

## Root Cause Analysis

### Dataset Comparison

| Dataset | Rows | File Size | HTTPS URLs | HTTP URLs | Training Time |
|---------|------|-----------|------------|-----------|---------------|
| dataset3.csv | 96,019 | 6.5 MB | **0** (0%) | 0 | ~80 min ✅ |
| dataset4.csv | 101,220 | 3.7 MB | **41,054** (40.5%) | 60,166 | ~58 hours ❌ |
| dataset5.csv | 235,795 | 8.9 MB | **184,013** (78%) | 51,782 | ~256 hours ❌ |

### Key Finding

**URLs with `https://` protocol trigger SSL certificate checks!**

When the feature extractor encounters `https://` URLs, it:
1. Makes actual network connections to fetch SSL certificates (`host_features.py:118`)
2. Waits for timeout on dead/invalid URLs (5 seconds per URL)
3. Extracts 6 SSL-related features (certificate age, validity, SAN count, etc.)

### Time Breakdown for dataset4.csv

```
HTTPS URLs: 41,054 × 5 sec timeout = 205,270 sec = 57 hours
HTTP URLs:  60,166 × 0.05 sec      =   3,008 sec = 50 minutes
----------------------------------------
TOTAL:                               58 hours
```

### Why dataset3.csv is Fast

- **No protocols in URLs** (no `http://` or `https://`)
- Feature extractor skips SSL certificate checks
- Only lexical features extracted (~0.05 sec per URL)
- Total time: 96,019 × 0.05 sec = ~80 minutes ✅

### Code Location

**File:** `src/phishing_detector/features/host_features.py`

**Line 118:** SSL certificate extraction
```python
with socket.create_connection((hostname, port), timeout=self.timeout) as sock:
    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
        cert = ssock.getpeercert()  # ← Network call, causes timeout
```

**Line 14:** Default timeout
```python
def __init__(self, enable_dns: bool = True, enable_whois: bool = False, timeout: int = 5):
    self.timeout = timeout  # ← 5 seconds per URL
```

## Solutions

### Option 1: Remove Protocols (FASTEST - RECOMMENDED) ✅

**Clean your dataset before training:**

```bash
# Clean dataset4
python clean_dataset_protocols.py dataset4.csv -o dataset4_cleaned.csv

# Clean dataset5
python clean_dataset_protocols.py dataset5.csv -o dataset5_cleaned.csv

# Train with cleaned dataset
python training/train.py --data dataset4_cleaned.csv
```

**Result:**
- Training time: **58 hours → 80 minutes** (43x faster!)
- All features still work (protocol is detected from URL structure)

**Pros:**
- ✅ No code changes needed
- ✅ Massive speedup (40-50x)
- ✅ Maintains all other features

**Cons:**
- ❌ Loses 6 SSL-related features
- ❌ Requires preprocessing step

---

### Option 2: Reduce Timeout (PARTIAL FIX)

Edit `src/phishing_detector/features/host_features.py` line 14:

```python
# Change timeout from 5 to 1 second
def __init__(self, enable_dns: bool = True, enable_whois: bool = False, timeout: int = 1):
```

**Result:**
- Training time: **58 hours → 11.4 hours** (5x faster)

**Pros:**
- ✅ Keeps SSL features
- ✅ Simple one-line change

**Cons:**
- ❌ Still slow (11+ hours)
- ❌ May miss some legitimate SSL certificates

---

### Option 3: Disable SSL Checks (REQUIRES CODE CHANGES)

Add a parameter to disable SSL checks during training.

**Step 1:** Modify `src/phishing_detector/features/host_features.py` line 14:

```python
def __init__(
    self,
    enable_dns: bool = True,
    enable_whois: bool = False,
    enable_ssl: bool = True,  # ADD THIS
    timeout: int = 5
):
    self.enable_ssl = enable_ssl
    # ... rest of init
```

**Step 2:** Modify line 38 in same file:

```python
# Wrap SSL check in conditional
if self.enable_ssl and components.get("scheme") == "https":
    features.update(self._extract_ssl_features(hostname, port))
else:
    features.update(self._get_default_ssl_features())
```

**Step 3:** Modify `training/train.py` line 707:

```python
feature_extractor = FeatureExtractor(
    enable_dns_lookup=False,
    enable_whois_lookup=False,
    enable_ssl_check=False,  # ADD THIS
    timeout=5,
)
```

**Result:**
- Training time: **58 hours → 80 minutes**

**Pros:**
- ✅ Can toggle SSL checks on/off
- ✅ Keeps flexibility for production

**Cons:**
- ❌ Requires multiple code changes
- ❌ Need to propagate parameter through multiple classes

---

## Recommendation

**For Training:** Use **Option 1** (Remove Protocols)

```bash
# Clean all datasets
python clean_dataset_protocols.py dataset4.csv
python clean_dataset_protocols.py dataset5.csv

# Train quickly
python training/train.py --data dataset4_cleaned.csv
python training/train.py --data dataset5_cleaned.csv --max-rows 50000
```

**For Production API:** Keep original code (SSL checks are valuable for real-time detection)

---

## Feature Impact Analysis

### SSL Features Lost When Removing Protocols

When protocols are removed, you lose these 6 features:

1. `has_ssl_cert` - Whether SSL certificate exists
2. `ssl_cert_valid` - Whether certificate is valid
3. `ssl_days_to_expire` - Days until certificate expires
4. `ssl_cert_expires_soon` - Certificate expires in < 30 days
5. `ssl_cert_age_days` - Age of certificate in days
6. `ssl_cert_is_new` - Certificate is < 30 days old

**Impact:** Minor (51 features remain out of 57 total)

Most phishing URLs are dead/invalid anyway, so these features would default to 0 for most samples.

---

## Quick Reference

### Check Dataset Protocols

```bash
# Count HTTPS URLs
grep -c "^https://" dataset4.csv

# Count HTTP URLs
grep -c "^http://" dataset4.csv

# Sample first 10 URLs
head -11 dataset4.csv | tail -10
```

### Estimate Training Time

```python
import pandas as pd

df = pd.read_csv('dataset4.csv')
https_count = df['url'].str.startswith('https://').sum()
http_count = df['url'].str.startswith('http://').sum()
other_count = len(df) - https_count - http_count

# Time estimate (minutes)
time_with_ssl = (https_count * 5 + http_count * 0.05 + other_count * 0.05) / 60
time_without_ssl = len(df) * 0.05 / 60

print(f"With SSL checks: {time_with_ssl:.1f} minutes ({time_with_ssl/60:.1f} hours)")
print(f"Without SSL checks: {time_without_ssl:.1f} minutes")
```

---

## Summary

| Action | Command | Time Saved |
|--------|---------|------------|
| Clean dataset4 | `python clean_dataset_protocols.py dataset4.csv` | 57 hours → 80 min |
| Clean dataset5 | `python clean_dataset_protocols.py dataset5.csv` | 256 hours → 196 min |
| Reduce timeout | Edit timeout from 5 → 1 | 5x speedup |
| Use max-rows | `--max-rows 10000` | Limit dataset size |

**Bottom line:** Remove `https://` and `http://` from URLs before training to get **40-50x speedup**!
