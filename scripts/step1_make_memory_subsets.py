import argparse
import os
import torch
import random
try:
    from romemo.memory.schema import MemoryBank
except ImportError:
    print("Please install romemo first.")
    exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_domain_path", type=str, required=True, help="Path to large in-domain memory .pt")
    parser.add_argument("--ood_path", type=str, required=True, help="Path to large OOD memory .pt")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading in-domain: {args.in_domain_path}")
    mem_in = MemoryBank.load_pt(args.in_domain_path)
    print(f"Loading OOD: {args.ood_path}")
    mem_ood = MemoryBank.load_pt(args.ood_path)

    # Convert to lists for easier slicing
    # Assuming experiences are stored in a list or similar structure
    exps_in = list(mem_in.experiences)
    exps_ood = list(mem_ood.experiences)
    
    random.shuffle(exps_in)
    random.shuffle(exps_ood)

    print(f"Total In-Domain: {len(exps_in)}")
    print(f"Total OOD: {len(exps_ood)}")

    # 1. Scaling Subsets (Pure In-Domain)
    sizes = [10, 50, 100, 500, 1000, 2000]
    for size in sizes:
        if size > len(exps_in):
            print(f"Warning: Not enough in-domain data for size {size}")
            continue
        
        subset = exps_in[:size]
        new_bank = MemoryBank(name=f"mem_{size}")
        for e in subset:
            new_bank.add(e)
        
        out_path = os.path.join(args.out_dir, f"mem_{size}.pt")
        new_bank.save_pt(out_path)
        print(f"Saved {out_path} ({len(new_bank)} items)")

    # 2. OOD Mixture Subsets (Fixed Total Size N=100)
    # Ratios: 100, 80, 60, 40, 20 (Percent In-Domain)
    total_n = 100
    ratios = [100, 80, 60, 40, 20]
    
    for r in ratios:
        n_in = int(total_n * (r / 100.0))
        n_ood = total_n - n_in
        
        if n_in > len(exps_in):
            print(f"Error: Not enough in-domain for ratio {r}")
            continue
        if n_ood > len(exps_ood):
            print(f"Error: Not enough OOD for ratio {r}")
            continue

        # Mix
        subset_in = exps_in[:n_in]
        subset_ood = exps_ood[:n_ood]
        mixed = subset_in + subset_ood
        random.shuffle(mixed) # Shuffle so index doesn't have order bias

        new_bank = MemoryBank(name=f"mem_mix_{r}pct")
        for e in mixed:
            new_bank.add(e)
            
        out_path = os.path.join(args.out_dir, f"mem_mix_{r}pct.pt")
        new_bank.save_pt(out_path)
        print(f"Saved {out_path} (In: {n_in}, OOD: {n_ood})")

if __name__ == "__main__":
    main()