using Luxor

R = 256
S = 8
r = 2 / (1 + sqrt(5))

jax_light_blue = "#5D99F7"
jax_light_green = "#22A79B"
jax_light_purple = "#EB82FD"

jax_blue = "#3067D7"
jax_green = "#007A6C"
jax_purple = "#9E23B2"

jax_dark_blue = "#2655C8"
jax_dark_green = "#006A5C"
jax_dark_purple = "#6B149C"

Drawing(R * 2, R * 2, "logo.svg")
origin()

setcolor(jax_blue)
circle(Point(0, 0), R, action=:fill)
setline(S)
setcolor("#ffffff")
circle(Point(0, 0), R - S / 2 + 1, action=:stroke)

R1 = R * r
P1 = (R - R1) * sqrt(2) / 2
setcolor(jax_green)
circle(Point(P1, P1), R1, action=:fill)
setline(S)
setcolor("#ffffff")
circle(Point(P1, P1), R1 - S / 2 + 1, action=:stroke)

R2 = R * r^2
P2 = (R - R2) * sqrt(2) / 2
setcolor(jax_purple)
circle(Point(P2, P2), R2, action=:fill)
setline(S)
setcolor("#ffffff")
circle(Point(P2, P2), R2 - S / 2 + 1, action=:stroke)

R3 = R * r^3
P3 = (R - R3) * sqrt(2) / 2
setcolor(Luxor.julia_red)
circle(Point(P3, P3), R3, action=:fill)
setline(S)
setcolor("#ffffff")
circle(Point(P3, P3), R3 - S / 2 + 1, action=:stroke)

finish()
preview()