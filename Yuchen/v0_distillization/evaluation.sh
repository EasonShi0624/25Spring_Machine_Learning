awk '{
  if (match($0, /rec ([0-9.]+)/, R) && match($0, /perc ([0-9.]+)/, C)) {
    ratio = R[1] * C[1]
    printf "%.8f %s\n", ratio, $0
  }
}' train_teacher.log | sort -n | head -n 10 > score.dat
