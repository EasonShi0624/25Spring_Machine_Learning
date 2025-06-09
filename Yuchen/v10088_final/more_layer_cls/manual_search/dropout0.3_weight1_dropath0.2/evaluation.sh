awk -f - dropout0.3_weight1_dropath0.2.log <<'EOF' | sort -n | head -n 100 > less_layer_score.dat
{
  if (match($0, /Recon=([0-9.]+)/, R) &&
      match($0, /Acc=([0-9.]+)/, C) &&
      $0 ~ /Val/) {
    ratio = R[1] / C[1]
    printf "%.8f %s\n", ratio, $0
  }
}
EOF
