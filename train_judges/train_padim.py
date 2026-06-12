import os
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.data import Folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "baseline")
EXPORT_DIR = os.path.join(BASE_DIR, "models", "padim_judge_256")

def main():
    print(f"[1/3] Przygotowanie danych z: {DATA_ROOT} ...")
    datamodule = Folder(
        name="hazelnut",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="test/hole",
        train_batch_size=32
    )
    
    print("[2/3] Inicjalizacja PaDiM (ResNet18)...")
    model = Padim(backbone="resnet18")
    
    engine = Engine() 
    
    print("[*] Trwa nauka! PaDiM modeluje rozkłady Gaussa dla pikseli")
    engine.fit(datamodule=datamodule, model=model)
    
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    print("[3/3] Eksportowanie wag modelu...")
    engine.export(
        model=model, 
        export_type="torch", 
        export_root=EXPORT_DIR
    )
    print(f"\n[*] SUKCES! Sędzia PaDiM gotowy i zapisany w: {EXPORT_DIR}")

if __name__ == "__main__":
    main()