## E2. Đối chiếu với Value Iteration (setting nhỏ)

| setting | phương pháp | V*(start) | return greedy | tỉ lệ tới G | khớp policy tối ưu | ghi chú |
|---|---|---|---|---|---|---|
| S1 gió tất định | Value Iteration | -15.00 | -15.00 | 1.00 | 1.00 | 16 vòng quét, 0.00s |
| S1 gió tất định | SARSA | -15.00 | -15.33 | 1.00 | 0.93 | hội tụ ~ episode 258 |
| S1 gió tất định | Q-learning | -15.00 | -15.00 | 1.00 | 1.00 | hội tụ ~ episode 139 |
| S1 gió ngẫu nhiên | Value Iteration | -20.63 | -20.30 | 1.00 | 1.00 | 103 vòng quét, 0.02s |
| S1 gió ngẫu nhiên | SARSA | -20.63 | -681.96 | 0.67 | 0.90 | hội tụ ~ episode 997 |
| S1 gió ngẫu nhiên | Q-learning | -20.63 | -681.05 | 0.67 | 0.94 | hội tụ ~ episode 997 |
