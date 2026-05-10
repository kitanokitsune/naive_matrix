# Naive Matrix 演算ライブラリ

正確な有理数演算、複素数、および様々な行列分解をサポートする、軽量で直感的なPython用行列演算ライブラリです。

## 🌟 特徴

- **演算**: 加算 (`+`)、減算 (`-`)、スカラー倍 (`*`)、行列乗算 (`@`)、行列除算 (`/`)、および整数冪 (`**`) をサポートしています。
- **高精度**: Pythonの `fractions.Fraction` および `decimal.Decimal` とシームレスに連携し、浮動小数点誤差を回避した正確な計算が可能です。
- **高度な演算**:
  - 行列式 (`det()`)
  - 階数 (`rank()`)
  - 逆行列 (`inv()`)
  - ムーア・ペンローズ擬似逆行列 (`pinv()`)
  - 転置 (`T()` または `transpose()`)
  - 共役転置 (`adjoint()`)
- **型変換**: 正確な有理数表現 (`Frac()`) と浮動小数点表現 (`Float()`) を簡単に切り替えることができます。
- **整形表示 (Pretty Printing)**: 可読性を高めるため、整形された行列出力機能を備えています。

## 📋 必要要件

- Python 3.x

このライブラリは、ガウス・ジョルダン消去法および複素数演算を行うために、いくつかのモジュールに依存しています。同じディレクトリに以下のファイルがあることを確認してください。

- `ratcomplex.py` (有理数複素数用)
- `gaussjordan.py` (コアとなる行列アルゴリズム用)

## 🚀 クイックスタート

```python
from naive_matrix import NaiveMatrix, NaiveIdentityMatrix

# 1. 行列の作成
A = NaiveMatrix([[1, 2], [3, 4]])
B = NaiveMatrix([[5, 6], [7, 8]])
A.pprint("Matirx A=")
B.pprint("Matirx B=")

# 2. 基本演算
print("スカラ乗算:\n", -1 * A)
print("スカラ除算:\n", A / 2)
print("加算:\n", A + B)
print("行列乗算:\n", A @ B)
print("行列除算:\n", A / B)  # A @ B.inv() と等価

# 3. 正確な有理数演算
# 浮動小数点誤差を避けるために Fraction に変換
C = NaiveMatrix([[0.1, 0.2], [0.3, 0.4]]).Frac(max_denominator=10)
print("正確な行列:")
C.pprint()

# 4. 高度な演算
print("行列式:", C.det())
print("逆行列:\n", C.inv())  # 1/A または A**-1 と等価
print("転置:\n", C.T())

# 5. 特殊な行列
I = NaiveIdentityMatrix(3)
print("単位行列:\n", I)
```

## 🛠 APIリファレンス

### `NaiveMatrix(mat=None)`
コアとなるクラス。`mat` にはリストのリスト、または他の `NaiveMatrix` インスタンスを指定できます。

| メソッド | 説明 |
|:---|:---|
| `.inv()` | 逆行列を返します。（`1/A`, `A**-1` と等価） |
| `.pinv()` | ムーア・ペンローズ擬似逆行列を返します。 |
| `.det()` | 行列式を返します。 |
| `.rank()` | 行列の階数 (rank) を返します。 |
| `.T()` / `.transpose()` | 転置行列を返します。 |
| `.adjoint()` | 共役転置を返します。 |
| `.Frac(max_denominator=None)` | 要素を `Fraction` に変換します。 |
| `.Float()` | 要素を `float` に変換します。 |
| `.pprint(prefix='')` | 整形してコンソールに表示します。 |
| `.spprint(prefix='')` | `.pprint()` の出力を文字列として返します。 |


### ユーティリティ関数
- `NaiveIdentityMatrix(n)`: $n \times n$ の単位行列を返します。
- `NaiveZeroMatrix(m, n=None)`: $m \times n$ の零行列を返します。
- `NaiveOneMatrix(m, n=None)`: 全ての要素が 1 の $m \times n$ 行列を返します。

## ⚠️ 計算量に関する注意
「Naive（素朴な）」という名前が示す通り、このライブラリは**教育目的、または小規模な正確な計算**のために設計されています。NumPyのような、大規模なデータに対する高性能な計算には最適化されていません。

## 📄 ライセンス
[MIT License](LICENSE)
