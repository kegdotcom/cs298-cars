export function distance_to_line(cx, cy, lx0, ly0, lx1, ly1, r = 0) {
  const lineV = [lx1 - lx0, ly1 - ly0];
  const circV = [cx - lx0, cy - ly0];

  const t = (circV[0] * lineV[0] + circV[1] * lineV[1]) / (lineV[0] * lineV[0] + lineV[1] * lineV[1]);
  return t > 0 && t < 1 ?
    (circV[1] * lineV[0] - circV[0] * lineV[1]) / Math.hypot(...lineV) :
    Number.MAX_SAFE_INTEGER;
}

export function distance_to_circle(x, y, cx, cy, r = 0) {
  return Math.hypot(x - cx, y - cy) - (2 * r);
}

export function reflect(vx, vy, nx, ny) {
  const magN = Math.hypot(nx, ny);
  const [nNx, nNy] = [nx / magN, ny / magN];
  const dot = vx * nNx + vy * nNy;
  return [vx - 2 * dot * nNx, vy - 2 * dot * nNy];
}

function is_between(x, a, b) {
  return x instanceof Array ?
    x.every(xi => xi >= a && xi <= b) :
    x >= a && x <= b;
}

function vec_add(x, y, a, b) {
  return x.length == y.length ?
    x.map((xi, i) => a * xi + b * y[i]) :
    [];
}

export function intersect_st(x0 = [0, 0], dx0 = [1, 1], x1 = [0, 0], dx1 = [1, 1]) {
  let d = dx0[0] * dx1[1] - dx1[0] * dx0[1];
  if (d == 0) {
    if ((x0[0] - x1[0]) * dx1[1] == (x0[1] - x1[1]) * dx1[0]) {
      if (dx1[0] != 0) return [0, (x0[0] - x1[0]) / dx1[0]];
      else if (dx1[1] != 0) return [0, (x0[1] - x1[1]) / dx1[1]];
      else return [];
    }
    else return [];
  }
  else {
    let s = dx1[0] * x0[1] - dx1[0] * x1[1] - dx1[1] * x0[0] + dx1[1] * x1[0];
    let t = dx0[0] * x0[1] - dx0[0] * x1[1] - dx0[1] * x0[0] + dx0[1] * x1[0];
    return [s / d, t / d];
  }
}

export function seg_intersect_seg(x0 = [0, 0], e0 = [1, 1], x1 = [0, 0], e1 = [1, 1]) {
  let st = intersect_st(x0, vec_add(e0, x0, 1, -1), x1, vec_add(e1, x1, 1, -1));
  if (st.length && is_between(st, 0, 1)) return vec_add(x0, vec_add(e0, x0, 1, -1), 1, st[0]);
  else return [];
}

export function ray_intersect_seg(x0 = [0, 0], d0 = [1, 1], x1 = [0, 0], e1 = [1, 1]) {
  let st = intersect_st(x0, d0, x1, vec_add(e1, x1, 1, -1));
  if (st.length && st[0] >= 0 && is_between(st[1], 0, 1)) return vec_add(x0, d0, 1, st[0]);
  else return [];
}

export function ray_intersect_circle(x0 = [0, 0], d0 = [1, 1], x1 = [0, 0], r = 1) {
  const thetaRay = Math.atan2(d0[1], d0[0]);
  const thetaCircle = Math.atan2(x1[1] - x0[1], x1[0] - x0[0]);
  const theta = thetaRay - thetaCircle;
  const d = Math.hypot(x1[0] - x0[0], x1[1] - x0[1]);
  const dNorm = d * Math.sin(theta)
  if (dNorm > r) return [];

  let a = d0[0] * d0[0] + d0[1] * d0[1];
  let b = 2 * d0[0] * (x0[0] - x1[0]) + 2 * d0[1] * (x0[1] - x1[1]);
  let c = (x0[0] - x1[0]) * (x0[0] - x1[0]) + (x0[1] - x1[1]) * (x0[1] - x1[1]) - r * r;
  let det = b * b - 4 * a * c;
  if (det < 0) return [];
  else {
    let t = (-b - Math.sqrt(det)) / (2 * a);
    if (t < 0) return [];
    else return vec_add(x0, d0, 1, t);
  }
}