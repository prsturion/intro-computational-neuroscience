
# q2a_ing_network_stable.py
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-9
def clipexp(x): return np.exp(np.clip(x, -60.0, 60.0))
def x_inf(a, b): return a / (a + b + EPS)
def tau_x(a, b): return 1.0 / (a + b + EPS)

class SpikeDetector:
    def __init__(self, n, thr=-20.0, refrac_ms=2.0):
        self.prevV = None; self.last_t = np.full(n, -1e9)
        self.thr = thr; self.refrac = refrac_ms
    def update(self, V, t):
        hits = []
        if self.prevV is None:
            self.prevV = V.copy(); return hits
        for i in range(len(V)):
            if (self.prevV[i] < self.thr) and (V[i] >= self.thr) and ((t - self.last_t[i]) >= self.refrac):
                hits.append(i); self.last_t[i] = t
        self.prevV = V.copy(); return hits

def rates_I(V):
    V = np.asarray(V)
    am = 0.1*(V+35.0) / (1.0 - clipexp(-(V+35.0)/10.0) + EPS)
    bm = 4.0*clipexp(-(V+60.0)/18.0)
    ah = 0.07*clipexp(-(V+58.0)/20.0)
    bh = 1.0/(clipexp(-0.1*(V+28.0)) + 1.0)
    an = 0.01*(V+34.0) / (1.0 - clipexp(-(V+34.0)) + EPS)
    bn = 0.125*clipexp(-(V+44.0)/80.0)
    return dict(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn)

I_params = dict(Cm=1.0, gNa=35.0, gK=9.0, gL=0.1, ENa=55.0, EK=-90.0, EL=-65.0)

def rates_O_mhn(V):
    V = np.asarray(V)
    am = -0.1*(V+38.0) / (1.0 - clipexp(-(V+38.0)/10.0) + EPS)
    bm = 4.0*clipexp(-(V+65.0)/18.0)
    ah = 0.07*clipexp(-(V+63.0)/20.0)
    bh = 1.0/(1.0 + clipexp(-(V+33.0)/10.0))
    an = 0.018*(V-25.0) / (1.0 - clipexp(-(V-25.0)/25.0) + EPS)
    bn = 0.0036*clipexp(-(V-35.0)/12.5)
    return dict(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn)

def O_aux_gates_inf_tau(V):
    V = np.asarray(V)
    a_inf = 1.0/(1.0 + clipexp(-(V+14.0)/16.6)); tau_a = 5.0
    b_inf = 1.0/(1.0 + clipexp((V+71.0)/7.3))
    tau_b = 1.0/( clipexp((V-26.0)/18.5)/100000.0 + 0.2 + clipexp((V+70.0)/11.0)*0.014 + EPS )
    r_inf = 1.0/(1.0 + clipexp((V+84.0)/10.2))
    tau_r = clipexp(-14.59 - 0.086*V) + clipexp(-1.87 + 0.0701*V)
    return a_inf, tau_a, b_inf, tau_b, r_inf, tau_r

O_params = dict(Cm=1.3, gNa=30.0, gK=23.0, gA=16.0, gh=12.0, gL=0.05,
                ENa=90.0, EK=-100.0, EA=-90.0, Eh=-32.9, EL=-70.0)

def dsdt(Vpre, s, tau_rise, tau_decay):
    drive = 0.5*(1.0 + np.tanh(Vpre/4.0))
    return drive*(1.0 - s)/tau_rise - s/tau_decay

GABA_I = dict(tau_rise=0.3, tau_decay=9.0, Erev=-80.0)
GABA_O = dict(tau_rise=0.2, tau_decay=20.0, Erev=-80.0)

class INGNetwork:
    def __init__(self, n_I=5, n_O=5, dt=0.005):
        self.nI = n_I; self.nO = n_O; self.N = n_I + n_O; self.dt = dt
        self.V = np.zeros(self.N)
        self.mI = np.zeros(self.nI); self.hI = np.zeros(self.nI); self.nIgate = np.zeros(self.nI)
        self.mO = np.zeros(self.nO); self.hO = np.zeros(self.nO); self.nOgate = np.zeros(self.nO)
        self.aO = np.zeros(self.nO); self.bO = np.zeros(self.nO); self.rO = np.zeros(self.nO)

        rng = np.random.default_rng(42)
        VI0 = rng.uniform(-70.0, -60.0, size=self.nI)
        rI = rates_I(VI0)
        self.V[:self.nI] = VI0
        self.mI[:] = x_inf(rI['am'], rI['bm']); self.hI[:] = x_inf(rI['ah'], rI['bh']); self.nIgate[:] = x_inf(rI['an'], rI['bn'])

        VO0 = -75.61; self.V[self.nI:] = VO0
        self.mO[:] = 0.0122; self.nOgate[:] = 0.07561; self.hO[:] = 0.9152; self.rO[:] = 0.06123; self.aO[:] = 0.0229; self.bO[:] = 0.2843

        self.s = np.zeros((self.N, self.N))
        self.gbar = np.zeros((self.N, self.N))
        I = np.arange(self.nI); O = np.arange(self.nI, self.N)
        self.gbar[np.ix_(O, I)] = 0.04
        self.gbar[np.ix_(I, O)] = 0.12
        self.gbar[np.ix_(I, I)] = 0.02

        self.J_I = 1.0; self.J_O = -3.0
        self.detector = SpikeDetector(self.N, thr=-20.0, refrac_ms=2.0)

    def external_current(self):
        J = np.zeros(self.N); J[:self.nI] = self.J_I; J[self.nI:] = self.J_O; return J

    def _clamp_state(self, V, mI, hI, nI, mO, hO, nO, aO, bO, rO, s):
        V  = np.clip(V,  -120.0, 50.0)
        mI = np.clip(mI, 0.0, 1.0); hI = np.clip(hI, 0.0, 1.0); nI = np.clip(nI, 0.0, 1.0)
        mO = np.clip(mO, 0.0, 1.0); hO = np.clip(hO, 0.0, 1.0); nO = np.clip(nO, 0.0, 1.0)
        aO = np.clip(aO, 0.0, 1.0); bO = np.clip(bO, 0.0, 1.0); rO = np.clip(rO, 0.0, 1.0)
        s  = np.clip(s,  0.0, 1.0)
        return V, mI, hI, nI, mO, hO, nO, aO, bO, rO, s

    def derive(self, V, mI, hI, nI, mO, hO, nO, aO, bO, rO, s):
        V, mI, hI, nI, mO, hO, nO, aO, bO, rO, s = self._clamp_state(V, mI, hI, nI, mO, hO, nO, aO, bO, rO, s)
        nIcount = self.nI; nOcount = self.nO
        dV = np.zeros(self.N); dmI = np.zeros(nIcount); dhI = np.zeros(nIcount); dnI = np.zeros(nIcount)
        dmO = np.zeros(nOcount); dhO = np.zeros(nOcount); dnO = np.zeros(nOcount); daO = np.zeros(nOcount); dbO = np.zeros(nOcount); drO = np.zeros(nOcount)
        ds = np.zeros_like(s)

        VI = V[:nIcount]; VO = V[nIcount:]

        rI = rates_I(VI)
        INa = np.nan_to_num(I_params['gNa']*(mI**3)*hI*(I_params['ENa'] - VI))
        IK  = np.nan_to_num(I_params['gK']*(nI**4)*(I_params['EK'] - VI))
        IL  = I_params['gL']*(I_params['EL'] - VI)

        IsynI = np.zeros(nIcount)
        s_I_to_I = s[:nIcount, :nIcount]; g_I_to_I = self.gbar[:nIcount, :nIcount]
        IsynI += np.sum(g_I_to_I * s_I_to_I * (-80.0 - VI), axis=0)
        s_O_to_I = s[nIcount:, :nIcount]; g_O_to_I = self.gbar[nIcount:, :nIcount]
        IsynI += np.sum(g_O_to_I * s_O_to_I * (-80.0 - VI), axis=0)

        JI = self.external_current()[:nIcount]
        dV[:nIcount] = (INa + IK + IL + IsynI + JI) / I_params['Cm']

        dmI[:] = (x_inf(rI['am'], rI['bm']) - mI)/tau_x(rI['am'], rI['bm'])
        dhI[:] = (x_inf(rI['ah'], rI['bh']) - hI)/tau_x(rI['ah'], rI['bh'])
        dnI[:] = (x_inf(rI['an'], rI['bn']) - nI)/tau_x(rI['an'], rI['bn'])

        rOm = rates_O_mhn(VO)
        a_inf, tau_a, b_inf, tau_b, r_inf, tau_r = O_aux_gates_inf_tau(VO)

        INa = np.nan_to_num(O_params['gNa']*(mO**3)*hO*(O_params['ENa'] - VO))
        IK  = np.nan_to_num(O_params['gK']*(nO**4)*(O_params['EK'] - VO))
        IA  = np.nan_to_num(O_params['gA']*aO*bO*(O_params['EA'] - VO))
        Ih  = np.nan_to_num(O_params['gh']*rO*(O_params['Eh'] - VO))
        IL  = O_params['gL']*(O_params['EL'] - VO)

        IsynO = np.zeros(nOcount)
        s_I_to_O = s[:nIcount, nIcount:]; g_I_to_O = self.gbar[:nIcount, nIcount:]
        IsynO += np.sum(g_I_to_O * s_I_to_O * (-80.0 - VO), axis=0)

        JO = self.external_current()[nIcount:]
        dV[nIcount:] = (INa + IK + IA + Ih + IL + IsynO + JO) / O_params['Cm']

        dmO[:] = (x_inf(rOm['am'], rOm['bm']) - mO)/tau_x(rOm['am'], rOm['bm'])
        dhO[:] = (x_inf(rOm['ah'], rOm['bh']) - hO)/tau_x(rOm['ah'], rOm['bh'])
        dnO[:] = (x_inf(rOm['an'], rOm['bn']) - nO)/tau_x(rOm['an'], rOm['bn'])
        daO[:] = (a_inf - aO)/(tau_a + EPS); dbO[:] = (b_inf - bO)/(tau_b + EPS); drO[:] = (r_inf - rO)/(tau_r + EPS)

        for j in range(self.N):
            Vpre = V[j]
            tr, td = (0.3, 9.0) if j < self.nI else (0.2, 20.0)
            ds[j, :] = dsdt(Vpre, s[j, :], tr, td)

        return dV, dmI, dhI, dnI, dmO, dhO, dnO, daO, dbO, drO, ds

    def step(self, t):
        V = self.V; mI,hI,nI = self.mI,self.hI,self.nIgate
        mO,hO,nO = self.mO,self.hO,self.nOgate; aO,bO,rO = self.aO,self.bO,self.rO
        s = self.s; dt = self.dt

        k1 = self.derive(V,mI,hI,nI,mO,hO,nO,aO,bO,rO,s)
        k2 = self.derive(V+0.5*dt*k1[0], mI+0.5*dt*k1[1], hI+0.5*dt*k1[2], nI+0.5*dt*k1[3],
                         mO+0.5*dt*k1[4], hO+0.5*dt*k1[5], nO+0.5*dt*k1[6],
                         aO+0.5*dt*k1[7], bO+0.5*dt*k1[8], rO+0.5*dt*k1[9], s+0.5*dt*k1[10])
        k3 = self.derive(V+0.5*dt*k2[0], mI+0.5*dt*k2[1], hI+0.5*dt*k2[2], nI+0.5*dt*k2[3],
                         mO+0.5*dt*k2[4], hO+0.5*dt*k2[5], nO+0.5*dt*k2[6],
                         aO+0.5*dt*k2[7], bO+0.5*dt*k2[8], rO+0.5*dt*k2[9], s+0.5*dt*k2[10])
        k4 = self.derive(V+dt*k3[0], mI+dt*k3[1], hI+dt*k3[2], nI+dt*k3[3],
                         mO+dt*k3[4], hO+dt*k3[5], nO+dt*k3[6],
                         aO+dt*k3[7], bO+dt*k3[8], rO+dt+k3[9], s+dt*k3[10])

        self.V        = V  + (dt/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        self.mI       = mI + (dt/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        self.hI       = hI + (dt/6.0)*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        self.nIgate   = nI + (dt/6.0)*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
        self.mO       = mO + (dt/6.0)*(k1[4]+2*k2[4]+2*k3[4]+k4[4])
        self.hO       = hO + (dt/6.0)*(k1[5]+2*k2[5]+2*k3[5]+k4[5])
        self.nOgate   = nO + (dt/6.0)*(k1[6]+2*k2[6]+2*k3[6]+k4[6])
        self.aO       = aO + (dt/6.0)*(k1[7]+2*k2[7]+2*k3[7]+k4[7])
        self.bO       = bO + (dt/6.0)*(k1[8]+2*k2[8]+2*k3[8]+k4[8])
        self.rO       = rO + (dt/6.0)*(k1[9]+2*k2[9]+2*k3[9]+k4[9])
        self.s        = s  + (dt/6.0)*(k1[10]+2*k2[10]+2*k3[10]+k4[10])

        # clamp
        self.V, self.mI, self.hI, self.nIgate, self.mO, self.hO, self.nOgate, self.aO, self.bO, self.rO, self.s = \
            self._clamp_state(self.V, self.mI, self.hI, self.nIgate, self.mO, self.hO, self.nOgate, self.aO, self.bO, self.rO, self.s)

        return self.detector.update(self.V, t)

def run_and_plot(T=1000.0, dt=0.005):
    net = INGNetwork(n_I=5, n_O=5, dt=dt)
    steps = int(T/dt); t = 0.0; st=[], []
    spike_times=[]; spike_ids=[]
    for _ in range(steps):
        sp = net.step(t)
        if sp:
            for i in sp: spike_times.append(t); spike_ids.append(i)
        t += dt
    spike_times = np.array(spike_times); spike_ids = np.array(spike_ids)

    fig, ax = plt.subplots(figsize=(10,4))
    nI, nO = net.nI, net.nO
    is_I = spike_ids < nI
    ax.scatter(spike_times[is_I], spike_ids[is_I], marker='o', s=15, color='k', label='I cells')
    ax.scatter(spike_times[~is_I], spike_ids[~is_I]-nI + nI + 1, marker='+', s=40, color='k', linewidths=1.2, label='O cells')
    ax.set_ylim(-1, nI + nO + 2); ax.set_xlim(0, T)
    ax.set_xlabel('time (ms)'); ax.set_ylabel('cell index (I bottom, O top)')
    ax.set_title('ING network raster (5 I + 5 O)'); ax.legend(loc='upper right', frameon=False)
    fig.tight_layout(); fig.savefig('intro-computational-neuroscience/list_8/figures/ex_2a.png', dpi=150); plt.close(fig)

    tmin = 200.0; dur_s = (T - tmin)/1000.0
    fI = np.sum((spike_times>=tmin) & (spike_ids < nI)) / (nI * dur_s) if dur_s>0 else 0.0
    fO = np.sum((spike_times>=tmin) & (spike_ids >= nI)) / (nO * dur_s) if dur_s>0 else 0.0
    with open('intro-computational-neuroscience/list_8/ex_2a_report.txt','w',encoding='utf-8') as f:
        f.write(f'I mean rate: {fI:.2f} Hz per neuron\n')
        f.write(f'O mean rate: {fO:.2f} Hz per neuron\n')
    return dict(fI=fI, fO=fO)

if __name__ == '__main__':
    out = run_and_plot()
    print('Saved q2a_raster_stable.png and q2a_report_stable.txt')
    print(out)
