## E3. Đối chiếu policy học được với policy tối ưu (Map 1)

| setting | phương pháp | V*(start) | return greedy | seed luôn tới G | khớp policy tối ưu | ghi chú | số bước (seed tới G) |
|---|---|---|---|---|---|---|---|
| Map 1 gió tất định | Value Iteration | -17.00 | -17.00 | – | 1.00 | 18 vòng quét | – |
| Map 1 gió tất định | Monte Carlo | -17.00 | -812.40 | 6/10 | 0.75 | hội tụ ~ episode 2977 | 20.67 |
| Map 1 gió tất định | SARSA | -17.00 | -216.00 | 9/10 | 0.94 | hội tụ ~ episode 142 | 17.78 |
| Map 1 gió tất định | Q-learning | -17.00 | -17.00 | 10/10 | 1.00 | hội tụ ~ episode 119 | 17.00 |
| Map 1 gió ngẫu nhiên | Value Iteration | -26.30 | -26.80 | – | 1.00 | 183 vòng quét | – |
| Map 1 gió ngẫu nhiên | Monte Carlo | -26.30 | -438.79 | 8/10 | 0.57 | hội tụ ~ episode 2789 | 48.48 |
| Map 1 gió ngẫu nhiên | SARSA | -26.30 | -224.48 | 9/10 | 0.94 | hội tụ ~ episode 2904 | 27.20 |
| Map 1 gió ngẫu nhiên | Q-learning | -26.30 | -224.07 | 9/10 | 0.93 | hội tụ ~ episode 2944 | 26.74 |
