"""본실험 표본 크기 계산(PREREGISTRATION.md 5절).

    python power.py            # 기본값으로 표 출력
    python power.py --rho .5 --arms 4

공분산분석(사전 점수 공변량)의 검정력을 두 집단 t검정의 비중심 t 분포로 근사한다.
사전 점수와의 상관 rho만큼 오차가 줄어 효과크기가 d / sqrt(1 - rho^2)로 커진다고 본다.
  주효과(H1·H2): 단계적 감소 네 칸을 반씩 나눈 비교 → 집단당 2n
  H3: 한 칸 대 한 칸 → 집단당 n   (n = 칸당 학생 수, 다섯 칸 균등 배정)
"""
import argparse
import numpy as np
from scipy import stats


def power(n1, n2, d, rho, alpha, k=3):
    de = d / np.sqrt(1 - rho ** 2); df = n1 + n2 - k
    nc = de * np.sqrt(n1 * n2 / (n1 + n2)); tc = stats.t.ppf(1 - alpha / 2, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=.4, help="사전·사후 점수 상관(기본 .4)")
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--arms", type=int, default=5, help="5: 계속 되묻기 포함, 4: 2x2만")
    a = ap.parse_args()
    print(f"rho={a.rho}, alpha={a.alpha}, 칸 {a.arms}개 균등 배정\n")
    print("총 학생 | 주효과 d=.3 .4 .5 | H3 d=.3 .4 .5 .8")
    for N in (100, 125, 150, 175, 200, 250, 300):
        n = N / a.arms
        me = [power(2 * n, 2 * n, d, a.rho, a.alpha) for d in (.3, .4, .5)]
        h3 = [power(n, n, d, a.rho, a.alpha) for d in (.3, .4, .5, .8)] if a.arms == 5 else []
        print(f"{N:6d} | " + " ".join(f"{x:.2f}" for x in me) + " | " + " ".join(f"{x:.2f}" for x in h3))


if __name__ == "__main__":
    main()
