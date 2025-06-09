awk '{
  if (match($0, /val_recon=([0-9.]+)/, R) && match($0, /val_cls=([0-9.]+)/, C)) {
    ratio = R[1] * C[1]
    printf "%.8f %s\n", ratio, $0
  }
}' SiLU_8latent.log | sort -n | head -n 10 > score.dat
