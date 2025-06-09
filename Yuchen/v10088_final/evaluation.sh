awk -f - bettercls_droppath_no_dropout.log <<'EOF' | sort -n | head -n 20 > less_layer_score.dat
{
  if (match($0, /Recon=([0-9.]+)/, R) &&
      match($0, /Acc=([0-9.]+)/, C) &&
      $0 ~ /Val/) {
    ratio = R[1] / C[1]
    printf "%.8f %s\n", ratio, $0
  }
}
EOF
