awk '{
  if (match($0, /val_recon=([0-9.]+)/, R) && match($0, /val_cls=([0-9.]+)/, C)) {
    ratio = R[1] * C[1]
    if (min == "" || ratio < min) {
      min = ratio
      recon = R[1]
      cls = C[1]
      line = $0
    }
  }
}
END {
  printf "Minimum ratio: %.4f%%\nval_recon=%s, val_cls=%s\nLine: %s\n", min, recon, cls, line
}' 1000_train_PReLU.log > score.dat