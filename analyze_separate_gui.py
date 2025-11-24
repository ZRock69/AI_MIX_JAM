import os, subprocess, tempfile, shutil
import numpy as np, librosa
from scipy.signal import butter, lfilter

BANDS = [(20,60),(60,250),(250,1000),(1000,4000),(4000,12000)]

def band_energy_from_signal(y, sr, low, high):
    nyq = 0.5*sr
    low_n = max(low/nyq,1e-6)
    high_n = min(high/nyq,0.999999)
    if low_n>=high_n: return 0.0
    b,a = butter(2,[low_n,high_n],btype='band')
    yf = lfilter(b,a,y)
    return float((yf**2).mean())

def analyze_wavefile(path,sr=44100):
    y,sr = librosa.load(path,sr=sr,mono=True)
    S = np.abs(librosa.stft(y,n_fft=2048,hop_length=512))
    centroid = float(librosa.feature.spectral_centroid(S=S).mean())
    rolloff = float(librosa.feature.spectral_rolloff(S=S).mean())
    rms = float(librosa.feature.rms(S=S).mean())
    band_energies = [band_energy_from_signal(y,sr,low,high) for (low,high) in BANDS]
    return {'path':path,'centroid':centroid,'rolloff':rolloff,'rms':rms,'band_energies':band_energies}

def energy_db(e):
    return 10*np.log10(max(e,1e-12))

def suggest_for_stem(name,analysis,mix_band_sum):
    suggestions=[]
    bands = analysis['band_energies']
    if bands[0]>1e-6 and 'bass' not in name.lower() and 'kick' not in name.lower():
        suggestions.append({'type':'HPF','freq':60,'reason':'energy under 60Hz; consider HPF'})
    if bands[1]>max(bands[2:])*1.4:
        suggestions.append({'type':'CUT','freq':200,'q':1.0,'db':-3,'reason':'mud 60-250Hz'})
    if 'voc' in name.lower() or 'voice' in name.lower():
        if analysis['centroid']<1500:
            suggestions.append({'type':'BOOST','freq':4000,'q':1.2,'db':2.0,'reason':'voice centroid low; add presence at 4k'})
    for i,be in enumerate(bands):
        track_db=energy_db(be)
        mix_db=energy_db(mix_band_sum[i])
        diff=mix_db-track_db
        if diff>6.0:
            low,high=BANDS[i]
            suggestions.append({'type':'UNMASK','freq':int((low+high)//2),'q':1.2,'db':2.0,'reason':f'mix louder than stem by {diff:.1f} dB in {low}-{high}Hz'})
    return suggestions

def analyze_mix_file(mix_path):
    tmpd=tempfile.mkdtemp(prefix='sep_')
    try:
        cmd=['spleeter','separate','-p','spleeter:4stems','-o',tmpd,mix_path]
        subprocess.run(cmd,check=True)
        base=os.path.join(tmpd,os.path.splitext(os.path.basename(mix_path))[0])
        expected=['vocals.wav','drums.wav','bass.wav','other.wav']
        stems={}
        for s in expected:
            p=os.path.join(base,s)
            if os.path.exists(p):
                stems[s.replace('.wav','')]=p
        if not stems: raise RuntimeError('Separation failed; ensure Spleeter installed')
        analyses={}
        for name,path in stems.items():
            analyses[name]=analyze_wavefile(path)
        nb=len(BANDS)
        mix_band_sum=[0.0]*nb
        for a in analyses.values():
            for i,be in enumerate(a['band_energies']):
                mix_band_sum[i]+=be
        results={'mix_file':os.path.basename(mix_path),'stems':{}}
        for name,a in analyses.items():
            sug=suggest_for_stem(name,a,mix_band_sum)
            results['stems'][name]={'analysis':a,'suggestions':sug}
        return results
    finally:
        try: shutil.rmtree(tmpd)
        except Exception: pass
