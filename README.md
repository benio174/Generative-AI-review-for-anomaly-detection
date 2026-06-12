# Ewaluacja Generowania Kontrfaktów w Detekcji Anomalii

## Oceniane Algorytmy
Projekt zestawia ze sobą metody rekonstrukcyjne (inpainting/autoenkodery) oraz nowoczesne modele generatywne (GAN/Diffusion). Do wygenerowania obrazów wykorzystano oficjalne implementacje poniższych architektur:

**Modele Generatywne (Pełna dwukierunkowość: A2N & N2A):**
* [AnoStyler](https://github.com/yulimso/AnoStyler)
* [DFMGAN](https://github.com/Ldhlwh/DFMGAN)
* [FAIR](https://github.com/liutongkun/FAIR)

**Modele Rekonstrukcyjne (Zoptymalizowane pod naprawę: A2N):**
* [RIAD](https://github.com/plutoyuxie/Reconstruction-by-inpainting-for-visual-anomaly-detection)
* [InTra](https://github.com/jhy12/inpainting-transformer)
* [AMI-Net](https://github.com/luow23/AMI-Net)

## Metodyka Ewaluacji (Sędziowie)
Autentyczność wygenerowanych kontrfaktów jest oceniana automatycznie za pomocą trzech referencyjnych detektorów z biblioteki [Anomalib](https://github.com/openvinotoolkit/anomalib):
1. **PaDiM**
2. **PatchCore**
3. **DFM**

---

## Struktura Projektu i Instrukcja do Danych (Ważne!)

Aby uruchomić program należy pobrać [Folder](https://drive.google.com/file/d/1UOhOlsbS2Vb0Al0RpBzjgg-2-V9yhkp9/view?usp=sharing), który zawiera wagi sędziów oraz obrazki wygenerowane przez powyższe algorytmy
Ze względu na rozmiar zbiorów danych oraz wag modeli, repozytorium zawiera wyłącznie kod skrypty ewaluacyjne i trenujące. 
Aby uruchomić projekt lokalnie, należy samodzielnie odtworzyć poniższą strukturę folderów:

```text
Anomaly-Evaluation-Project/
│
├── evaluate.py                  # Główny skrypt wyliczający metryki
├── train_padim_judge.py         # Skrypt trenujący detektor PaDiM
├── train_patchcore_judge.py     # Skrypt trenujący detektor PatchCore
├── train_dfm_judge.py           # Skrypt trenujący detektor DFM
│
├── models/                      # TUTAJ umieść wytrenowane wagi sędziów z Anomalib
│   ├── padim_judge_256/
│   │   └── weights/torch/model.pt
│   ├── patchcore_judge_256/
│   │   └── weights/torch/model.pt
│   └── dfm_judge_256/
│       └── weights/torch/model.pt
│
└── data/                        # TUTAJ umieść wszystkie zdjęcia i dataset
    ├── baseline/
    │   ├── train/good/          # Oryginalne zdrowe orzechy (MVTec)
    │   └── test/hole/           # Oryginalne uszkodzone orzechy (MVTec)
    ├── AnoStyler/               # Wygenerowane kontrfakty z AnoStylera
    ├── DFMGAN/                  # Wygenerowane kontrfakty z DFMGAN
    ├── FAIR/                    # Wygenerowane kontrfakty z FAIR
    ├── RIAD/                    # Zrekonstruowane obrazy z RIAD
    ├── InTra/                   # Zrekonstruowane obrazy z InTra
    └── AMI-Net/                 # Zrekonstruowane obrazy z AMI-Net
```
Aby otrzymać finalny raport w terminalu z projektem należy wpisać:

```bash
python evaluate.py
```

# Przykładowy otrzymany wynik z programu:
```text
===================================================================================================================
                      OSTATECZNY ZINTEGROWANY RAPORT PORÓWNAWCZY 
===================================================================================================================
Kategoria / Metoda             | Średni PaDiM    | Średni PatchCore   | Średni DFM     
-------------------------------------------------------------------------------------------------------------------
Baseline: Zdrowe               | 0.0155          | 0.0304             | 0.0685         
Baseline: Wada (Hole)          | 0.6669          | 0.8175             | 0.7641         
-------------------------------------------------------------------------------------------------------------------
AnoStyler: Abnormal (N2A)      | 0.2790          | 0.1564             | 0.0828         
AnoStyler: Normal (A2N)        | 0.3047          | 0.1392             | 0.5191         
-------------------------------------------------------------------------------------------------------------------
FAIR: Abnormal (N2A)           | 0.8388          | 0.9442             | 0.3497         
FAIR: Normal (A2N)             | 0.9879          | 0.2696             | 0.9750         
-------------------------------------------------------------------------------------------------------------------
DFMGAN: Abnormal (N2A)         | 0.7965          | 0.8588             | 1.0000         
DFMGAN: Normal (A2N)           | 0.1667          | 0.0615             | 1.0000         
-------------------------------------------------------------------------------------------------------------------
RIAD: Abnormal (N2A)           | -               | -                  | -              
RIAD: Normal (A2N)             | 0.3653          | 0.3280             | 1.0000         
-------------------------------------------------------------------------------------------------------------------
InTra: Abnormal (N2A)          | -               | -                  | -              
InTra: Normal (A2N)            | 0.6642          | 0.7089             | 1.0000         
-------------------------------------------------------------------------------------------------------------------
AMI-Net: Abnormal (N2A)        | -               | -                  | -              
AMI-Net: Normal (A2N)          | 0.2800          | 0.3322             | 1.0000         
-------------------------------------------------------------------------------------------------------------------
