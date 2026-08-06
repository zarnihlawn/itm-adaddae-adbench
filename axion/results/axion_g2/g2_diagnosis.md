# AXION G2 diagnosis

- Unsup: PR=34.05 ROC=81.82 pass=False
- Semi: PR=43.53 ROC=82.90 pass=False
- Classical-8 semi mean: 53.34
- CV/NLP-4 semi mean: 23.90

## Worst-5 semi PR
- Imdb: PR=9.32 ROC=50.33
- cover: PR=14.74 ROC=85.10
- Agnews: PR=14.85 ROC=66.51
- CIFAR10: PR=17.70 ROC=64.98
- glass: PR=24.60 ROC=79.11

## All semi

| Imdb | 9.32 | 50.33 |
| cover | 14.74 | 85.10 |
| Agnews | 14.85 | 66.51 |
| CIFAR10 | 17.70 | 64.98 |
| glass | 24.60 | 79.11 |
| backdoor | 26.13 | 86.48 |
| fraud | 38.91 | 95.12 |
| FashionMNIST | 53.74 | 87.29 |
| cardio | 58.85 | 83.94 |
| thyroid | 68.87 | 97.50 |
| breastw | 97.24 | 98.54 |
| satimage-2 | 97.36 | 99.86 |
