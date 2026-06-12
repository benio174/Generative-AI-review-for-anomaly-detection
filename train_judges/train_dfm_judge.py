import os
import sys
from anomalib.engine import Engine
from anomalib.models import Dfm
from anomalib.data import Folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "baseline")
EXPORT_DIR = os.path.join(BASE_DIR, "models", "dfm_judge_256")

def main():
    train_good_dir = os.path.join(DATA_ROOT, "train", "good")
    
    if not os.path.exists(train_good_dir) or len(os.listdir(train_good_dir)) == 0:
        print(f"\n[!] BŁĄD KRYTYCZNY: Twój folder {train_good_dir} jest pusty lub nie istnieje!")
        sys.exit()

    print(f"[1/3] Przygotowanie danych z: {DATA_ROOT} ...")
    datamodule = Folder(
        name="hazelnut",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="test/hole"
    )
    
    print("[2/3] Inicjalizacja modelu DFM (Sędziego)")
    model = Dfm()
    engine = Engine()
    
    print("Trwa nauka")
    engine.fit(datamodule=datamodule, model=model)
    
    print("\n[3/3] Zapisywanie Sędziego (wag modelu)")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    engine.export(
        model=model, 
        export_type="torch", 
        export_root=EXPORT_DIR
    )
    print(f"[*] Gotowe! Sędzia zapamiętał reguły i czeka w folderze: {EXPORT_DIR}")

if __name__ == "__main__":
    main()