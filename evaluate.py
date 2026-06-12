import os
import cv2
import torch
import numpy as np
import random
from tqdm import tqdm
from anomalib.deploy import TorchInferencer

os.environ["TRUST_REMOTE_CODE"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY"] = "0"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

PADIM_MODEL_PATH = os.path.join(MODELS_DIR, "padim_judge_256", "weights", "torch", "model.pt")
PATCHCORE_MODEL_PATH = os.path.join(MODELS_DIR, "patchcore_judge_256", "weights", "torch", "model.pt")
DFM_MODEL_PATH = os.path.join(MODELS_DIR, "dfm_judge_256", "weights", "torch", "model.pt")

SAMPLE_SIZE = 20 

PATHS = {
    "Baseline: Zdrowe": os.path.join(DATA_DIR, "baseline", "train", "good"),
    "Baseline: Wada (Hole)": os.path.join(DATA_DIR, "baseline", "test", "hole"),
    
    "AnoStyler: Abnormal (N2A)": os.path.join(DATA_DIR, "AnoStyler", "hazelnut_CF_abnormal_v2"),
    "AnoStyler: Normal (A2N)": os.path.join(DATA_DIR, "AnoStyler", "hazelnut_CF_normal"),
    
    "FAIR: Abnormal (N2A)": os.path.join(DATA_DIR, "FAIR", "synthetic_defects"),
    "FAIR: Normal (A2N)": os.path.join(DATA_DIR, "FAIR", "normal"),
    
    "DFMGAN: Abnormal (N2A)": os.path.join(DATA_DIR, "DFMGAN", "abnormal"),
    "DFMGAN: Normal (A2N)": os.path.join(DATA_DIR, "DFMGAN", "normal"),
    
    "RIAD: Normal (A2N)": os.path.join(DATA_DIR, "RIAD", "hazelnut_normal_1"),
    "InTra: Normal (A2N)": os.path.join(DATA_DIR, "InTra", "results"),
    "AMI-Net: Normal (A2N)": os.path.join(DATA_DIR, "AMI-Net", "img")
}

print("[1/2] Budzenie sędziów (PaDiM, PatchCore, DFM)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

inferencer_padim = TorchInferencer(path=PADIM_MODEL_PATH, device=device)
inferencer_patchcore = TorchInferencer(path=PATCHCORE_MODEL_PATH, device=device)
inferencer_dfm = TorchInferencer(path=DFM_MODEL_PATH, device=device)

def get_scores(image_path):
    image = cv2.imread(image_path)
    if image is None: return None, None, None
    
    if image.shape[0] != 256 or image.shape[1] != 256:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pred_padim = float(inferencer_padim.predict(image=image).pred_score)
    pred_patchcore = float(inferencer_patchcore.predict(image=image).pred_score)
    pred_dfm = float(inferencer_dfm.predict(image=image).pred_score)
    
    return pred_padim, pred_patchcore, pred_dfm

def get_filtered_files(folder_path, key):
    all_found = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            f_lower = f.lower()
            
            if "DFMGAN: Abnormal" in key:
                if "_img" in f_lower: all_found.append(os.path.join(root, f))
            elif "DFMGAN: Normal" in key:
                if "proj" in f_lower: all_found.append(os.path.join(root, f))
            
            elif "RIAD: Normal" in key:
                if f_lower.endswith("_recon.jpg") or f_lower.endswith("_recon.jpeg") or f_lower.endswith("_recon.png"):
                    all_found.append(os.path.join(root, f))
            
            elif "InTra: Normal" in key:
                if f_lower.endswith("_recon.jpg") or f_lower.endswith("_recon.jpeg") or f_lower.endswith("_recon.png"):
                    all_found.append(os.path.join(root, f))

            elif "AMI-Net: Normal" in key:
                if f_lower.endswith("_recon.jpg") or f_lower.endswith("_recon.png"):
                    all_found.append(os.path.join(root, f))
            
            else:
                all_found.append(os.path.join(root, f))
    return all_found

def evaluate_category(name, folder_path):
    if not os.path.exists(folder_path): 
        print(f"Brak folderu: {folder_path} - pomijam...")
        return (0.0, 0.0, 0.0)
        
    files = get_filtered_files(folder_path, name)
    if not files: return (0.0, 0.0, 0.0)
    
    sampled_files = random.sample(files, min(len(files), SAMPLE_SIZE))
    
    padim_list, patch_list, dfm_list = [], [], []
    
    for f in tqdm(sampled_files, desc=f"Ocenianie {name[:18]}", leave=False):
        s_padim, s_patch, s_dfm = get_scores(f)
        if s_padim is not None:
            padim_list.append(s_padim)
            patch_list.append(s_patch)
            dfm_list.append(s_dfm)
            
    return np.mean(padim_list), np.mean(patch_list), np.mean(dfm_list)

def main():
    results = {}
    print("\n[2/2] Rozpoczynam wielką zintegrowaną ewaluację (Może to chwilę potrwać).")
    
    for key in PATHS.keys():
        m_padim, m_patch, m_dfm = evaluate_category(key, PATHS[key])
        results[key] = {
            "padim": m_padim,
            "patchcore": m_patch,
            "dfm": m_dfm
        }

    print("\n" + "="*115)
    print("                      OSTATECZNY ZINTEGROWANY RAPORT PORÓWNAWCZY ")
    print("="*115)
    print(f"{'Kategoria / Metoda':<30} | {'Średni PaDiM':<15} | {'Średni PatchCore':<18} | {'Średni DFM':<15}")
    print("-" * 115)
    
    groups = [
        ["Baseline: Zdrowe", "Baseline: Wada (Hole)"],
        ["AnoStyler: Abnormal (N2A)", "AnoStyler: Normal (A2N)"],
        ["FAIR: Abnormal (N2A)", "FAIR: Normal (A2N)"],
        ["DFMGAN: Abnormal (N2A)", "DFMGAN: Normal (A2N)"],
        ["RIAD: Abnormal (N2A)", "RIAD: Normal (A2N)"],
        ["InTra: Abnormal (N2A)", "InTra: Normal (A2N)"],
        ["AMI-Net: Abnormal (N2A)", "AMI-Net: Normal (A2N)"]
    ]
    
    for group in groups:
        for item in group:
            if item in results:
                res = results[item]
                print(f"{item:<30} | {res['padim']:<15.4f} | {res['patchcore']:<18.4f} | {res['dfm']:<15.4f}")
            else:
                print(f"{item:<30} | {'-':<15} | {'-':<18} | {'-':<15}")
        print("-" * 115)

    print("LEGENDA I INTERPRETACJA METRYK:")
    print("  > Znak '-': Oznacza brak natywnego wsparcia modelu (architektury rekonstrukcyjne) dla generacji N2A.")
    print("  > Dla wierszy Abnormal (N2A): WYŻSZY wynik u sędziego oznacza LEPSZĄ, bardziej realistyczną stymulację wady.")
    print("  > Dla wierszy Normal (A2N):   NIŻSZY wynik u sędziego oznacza LEPSZĄ, czystszą rekonstrukcję/naprawę obrazu.")
    print("="*115)

if __name__ == "__main__":
    main()