awk -f - ydx_log_less_layer_no_atten_more_res.log <<'EOF' | sort -n | head -n 10 > ydx_log_less_layer_no_atten_more_res_score.dat
{
  if (match($0, /Recon=([0-9.]+)/, R) &&
      match($0, /Acc=([0-9.]+)/, C) &&
      $0 ~ /Val/) {
    ratio = R[1] / C[1]
    printf "%.8f %s\n", ratio, $0
  }
}
EOF
