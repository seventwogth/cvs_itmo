from pathlib import Path
import csv
import numpy as np

R, rho, flux = 2.0, 0.9, 1000.0
nz, nphi, rays, seed = 4, 8, 100000, 20261014
N = nz * nphi
area = 4 * np.pi * R**2 / N
E0 = flux / (4 * np.pi * R**2)
truth = E0 / (1 - rho)
rng = np.random.default_rng(seed)
F = np.zeros((N, N))
for i in range(N):
    iz, ip = divmod(i, nphi)
    z = -1 + 2 * (iz + rng.random(rays)) / nz
    phi = 2 * np.pi * (ip + rng.random(rays)) / nphi
    p = R * np.column_stack((np.sqrt(1-z*z)*np.cos(phi),
                             np.sqrt(1-z*z)*np.sin(phi), z))
    normal = -p / R
    ref = np.tile([0., 0., 1.], (rays, 1))
    ref[np.abs(normal[:, 2]) > 0.99] = [1., 0., 0.]
    tangent = np.cross(ref, normal)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    bitangent = np.cross(normal, tangent)
    u, az = rng.random(rays), 2*np.pi*rng.random(rays)
    direction = (np.sqrt(u)[:, None] *
                 (np.cos(az)[:, None]*tangent +
                   np.sin(az)[:, None]*bitangent) +
                 np.sqrt(1-u)[:, None]*normal)
    t = -2 * np.einsum('ij,ij->i', p, direction)
    hit = p + t[:, None]*direction
    jz = np.clip(((hit[:, 2]/R + 1)*nz/2).astype(int), 0, nz-1)
    jp = (np.mod(np.arctan2(hit[:, 1], hit[:, 0]), 2*np.pi)
          * nphi/(2*np.pi)).astype(int)
    F[i] = np.bincount(jz*nphi+jp, minlength=N) / rays

# F[i,j] transports flux from patch i to patch j.
# Areas are equal, hence the irradiance operator is F.T.
exact_F = np.full((N, N), 1/N)
assert np.allclose(F.sum(axis=1), 1)
assert np.min(F) >= 0
E = np.full(N, E0)

rows = []
selected = {0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 65, 80, 100}
for k in range(101):
    analytic = E0*(1-rho**(k+1))/(1-rho)
    assert abs(E.mean()-analytic) < 1e-10
    if k in selected:
        rows.append([k, truth, analytic, E.mean(), E.min(), E.max(),
                      100*rho**(k+1),
                      100*np.sqrt(np.mean((E-truth)**2))/truth])
    E = E0 + rho * (F.T @ E)
steady = np.linalg.solve(np.eye(N)-rho*F.T, np.full(N, E0))
out = Path(__file__).resolve().parent
np.savetxt(out/'form_factors.csv', F, delimiter=',')
with (out/'iterations.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(['k','true_lux','analytic_lux','mc_mean_lux',
                'mc_min_lux','mc_max_lux','analytic_error_pct','mc_rmse_pct'])
    w.writerows(rows)
with (out/'tables.tex').open('w') as f:
    for k,t,a,m,lo,hi,err,rmse in rows:
        f.write(f'{k} & {t:.4f} & {a:.4f} & {lo:.4f} & {hi:.4f} & {err:.4f} & {rmse:.4f} \\\\\n')
print('E0, truth:', E0, truth)
print('F RMS error, max error:', np.sqrt(np.mean((F-exact_F)**2)),
      np.max(np.abs(F-exact_F)))
print('reciprocity max:', np.max(np.abs(F-F.T)))
print('steady min,max,relative RMS %:', steady.min(), steady.max(),
      100*np.sqrt(np.mean((steady-truth)**2))/truth)
print('first row:', F[0,:8])

# Additional experiments change one physical parameter at a time.
cases = [('A', 2., 0.90, 1000.), ('B', 1., 0.90, 1000.),
         ('C', 2., 0.90, 2000.), ('D', 2., 0.50, 1000.),
         ('E', 2., 0.98, 1000.)]
steps = [0, 1, 5, 10, 20, 50, 100, 200, 350, 500]
results, limits, summary = {}, {}, []
all_rows = []
for label, radius, reflectance, lumens in cases:
    assert radius > 0 and lumens > 0 and 0 <= reflectance < 1
    direct = lumens / (4*np.pi*radius**2)
    true = direct / (1-reflectance)
    value = np.full(N, direct)
    results[label] = {}
    for k in range(max(steps)+1):
        analytic = true*(1-reflectance**(k+1))
        assert np.isclose(value.mean(), analytic, rtol=1e-12)
        if k in steps:
            err = 100*reflectance**(k+1)
            rmse = 100*np.sqrt(np.mean((value-true)**2))/true
            row = [label, k, true, analytic, value.mean(),
                   value.min(), value.max(), err, rmse]
            all_rows.append(row)
            results[label][k] = row
        value = direct + reflectance*(F.T @ value)
    limit = np.linalg.solve(np.eye(N)-reflectance*F.T,
                             np.full(N, direct))
    assert np.isclose((1-reflectance)*4*np.pi*radius**2
                       *limit.mean(), lumens, rtol=1e-12)
    limits[label] = [limit.mean(), limit.min(), limit.max(),
                    100*np.sqrt(np.mean((limit-true)**2))/true]
    n1 = max(0, int(np.ceil(np.log(.01)/np.log(reflectance)))-1)
    n01 = max(0, int(np.ceil(np.log(.001)/np.log(reflectance)))-1)
    summary.append([label, radius, reflectance, lumens,
                    direct, true, n1, n01])
with (out/'all_cases.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['case','k','true_lux','analytic_lux','mc_mean_lux',
                'mc_min_lux','mc_max_lux','analytic_error_pct','mc_rmse_pct'])
    w.writerows(all_rows)
with (out/'case_summary.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['case','R_m','rho','flux_lm','direct_lux','true_lux',
                'k_for_1pct','k_for_0.1pct'])
    w.writerows(summary)

def tex_table(headers, rows, digits=4):
    lines = [r'\begin{center}\small', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{'+'r'*len(headers)+'}', r'\toprule',
             ' & '.join(headers)+r' \\', r'\midrule']
    for row in rows:
        cells = [f'{v:.{digits}f}' if isinstance(v, (float, np.floating))
                 else str(v) for v in row]
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    return '\n'.join(lines)+'\n'

blocks = {}
blocks['inputs'] = tex_table(
    ['Набор', '$R$, м', '$\\rho$', '$\\Phi$, лм'], cases, 2)
blocks['summary'] = tex_table(
    ['Набор', '$E_0$, лк', '$E_*$, лк', '$k_{1\\%}$', '$k_{0{,}1\\%}$'],
    [[s[0], s[4], s[5], s[6], s[7]] for s in summary])
blocks['iterations'] = tex_table(
    ['$k$']+[c[0] for c in cases],
    [[k]+[results[c[0]][k][3] for c in cases] for k in steps])
blocks['errors'] = tex_table(
    ['$k$', '$\\delta_D$', '$\\delta_{A,B,C}$', '$\\delta_E$'],
    [[k]+[results[c][k][7] for c in ['D','A','E']] for k in steps])
blocks['mc'] = tex_table(
    ['Набор', '$\\overline E_\\infty$', '$E_{\\min}$', '$E_{\\max}$',
     '$\\delta_{\\rm MC}$, \\%'],
    [[c[0]]+limits[c[0]] for c in cases])
for name, body in blocks.items():
    (out/f'cases_{name}.tex').write_text(body, encoding='utf-8')
print('Additional cases:', summary)
print('Monte Carlo limits:', limits)
