## E2. Quy hoạch động: Policy Iteration vs Value Iteration

| config | phương pháp | V*(start) | số vòng lặp | thời gian (s) | policy tối ưu |
|---|---|---|---|---|---|
| map1_classic | Value Iteration | -17.00 | 18 vòng quét | 0.00 | có |
| map1_classic | Policy Iteration | -17.00 | 17 lần cải thiện, 16002 vòng quét đánh giá | 0.81 | khớp VI 100% |
| map2_stochastic | Value Iteration | -26.49 | 99 vòng quét | 0.09 | có |
| map2_stochastic | Policy Iteration | -26.49 | 20 lần cải thiện, 14043 vòng quét đánh giá | 1.74 | khớp VI 100% |
| map3_anchor | Value Iteration | -30.19 | 200 vòng quét | 0.69 | có |
| map3_anchor | Policy Iteration | -30.19 | 27 lần cải thiện, 18111 vòng quét đánh giá | 9.71 | khớp VI 100% |
