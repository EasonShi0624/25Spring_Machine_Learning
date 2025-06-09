awk '{
  if (match($0, /recon=([0-9.]+)/, R) && match($0, /acc=([0-9.]+)/, C)) {
    ratio = R[1] / C[1]
    printf "%.8f %s\n", ratio, $0
  }
}' train.log | sort -n | head -n 10 > score.dat
