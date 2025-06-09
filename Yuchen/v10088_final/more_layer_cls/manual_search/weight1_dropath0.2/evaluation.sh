awk -f - seed3407.log <<'EOF' | sort -n | head -n 20 > acc_score.dat
{
  if (match($0, /Recon=([0-9.]+)/, R) &&
      match($0, /Acc=([0-9.]+)/, C) &&
      $0 ~ /Val/) {
    ratio = R[1] / C[1]
    printf "%.8f %s\n", ratio, $0
  }
}
EOF
