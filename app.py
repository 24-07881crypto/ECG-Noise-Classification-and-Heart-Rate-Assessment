"""
Upload-based ECG Noise Classification & Heart-Rate Reliability Assessment
Educational research prototype. Not a diagnostic or clinical decision-support device.

Input:
- CSV ECG waveform (one numeric sample per row)
- Sampling frequency
- Optional symptom questionnaire (recorded as context only)

The classifiers are trained on procedurally generated synthetic ECG/noise examples.
They are NOT trained on MIT-BIH NSTDB in this starter version. Predictions on real
uploaded ECGs are therefore exploratory and require validation before research claims.
"""
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, resample_poly
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from math import gcd
from datetime import datetime, timezone

FS_MODEL = 250
WINDOW_SECONDS = 10
N = FS_MODEL * WINDOW_SECONDS
CLASSES = ["Clean", "Baseline", "Muscle", "Power line", "Motion"]
BANDS = [(0, .7), (.7, 5), (5, 20), (20, 45), (58, 62), (45, 125)]
FEATURE_NAMES = [
    "relative_power_0_0.7Hz", "relative_power_0.7_5Hz",
    "relative_power_5_20Hz", "relative_power_20_45Hz",
    "relative_power_58_62Hz", "relative_power_45_125Hz",
    "signal_std", "signal_kurtosis", "first_difference_std",
    "mean_absolute_first_difference", "detector_1_hr", "detector_2_hr",
    "detector_disagreement", "detector_1_peak_count", "detector_2_peak_count"
]

st.set_page_config(page_title="ECG Upload Assessment", page_icon="💓", layout="wide")
st.title("💓 ECG Noise Classification & Heart-Rate Reliability Assessment")
st.caption("Upload-based research prototype | Synthetic-model training | Not for diagnosis")

st.warning(
    "Research prototype only. This application does not diagnose heart disease, "
    "interpret symptoms, or determine whether a person is medically safe. Do not use "
    "its output to delay urgent care or replace a clinician's assessment."
)

with st.expander("What this system does — and does not do", expanded=False):
    st.markdown("""
    **It does:** analyze an uploaded ECG waveform, estimate heart rate with two
    transparent peak detectors, classify likely signal contamination, and estimate
    whether the primary rate estimate is within the project's ±5 bpm research tolerance.

    **It does not:** diagnose arrhythmia, heart attack, or any other condition; determine
    the cause of symptoms; or provide medical clearance. The symptom questionnaire is
    stored only as user-provided context and is not fed into the machine-learning models.

    **Model limitation:** the starter classifiers are trained on procedurally generated
    PQRST signals and procedural noise. The paper's intended benchmark additionally uses
    recorded MIT-BIH NSTDB baseline (`bw`), muscle (`ma`), and electrode-motion (`em`)
    noise. Until those recordings are integrated and the model is validated on appropriate
    independent data, predictions for uploaded real-world ECGs are exploratory.
    """)

def clean_ecg(hr, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS_MODEL
    beat_times, bt = [], -0.5
    rr = 60.0 / hr
    while bt < WINDOW_SECONDS + .5:
        beat_times.append(bt)
        bt += max(.32, rr * (1 + rng.normal(0, .025)))
    x = np.zeros_like(t)
    amps = [.12, -.16, 1.0, -.24, .28]
    offsets = [-.19, -.035, 0, .035, .25]
    widths = [.045, .012, .014, .014, .07]
    amp_scale, width_scale = rng.uniform(.8, 1.2), rng.uniform(.85, 1.15)
    for beat in beat_times:
        for a, d, w in zip(amps, offsets, widths):
            x += a*amp_scale*np.exp(-((t-beat-d)**2)/(2*(w*width_scale)**2))
    valid = np.array([b for b in beat_times if 0 <= b <= WINDOW_SECONDS])
    ref = 60/np.mean(np.diff(valid)) if len(valid) > 1 else float(hr)
    return x, ref

def make_noise(kind, rng):
    t = np.arange(N)/FS_MODEL
    if kind == "Clean":
        return np.zeros(N)
    if kind == "Baseline":
        return .30*np.sin(2*np.pi*rng.uniform(.15,.45)*t+rng.uniform(0,2*np.pi)) + .08*np.sin(2*np.pi*.05*t)
    if kind == "Muscle":
        w = rng.normal(0,1,N)
        return .18*sosfiltfilt(butter(4,[20,90],btype="bandpass",fs=FS_MODEL,output="sos"),w)
    if kind == "Power line":
        return .22*np.sin(2*np.pi*60*t+rng.uniform(0,2*np.pi))
    if kind == "Motion":
        out = .12*np.sin(2*np.pi*.3*t+rng.uniform(0,2*np.pi))
        for _ in range(rng.integers(2,6)):
            c, width = rng.uniform(.5,9.5), rng.uniform(.08,.35)
            out += rng.uniform(.25,.65)*rng.choice([-1,1])*np.exp(-((t-c)**2)/(2*width**2))
        return out
    return np.zeros(N)

def corrupt(x, kind, snr, rng):
    n = make_noise(kind,rng)
    if kind == "Clean":
        return x.copy()
    scale = np.sqrt((np.mean(x*x)+1e-12)/((np.mean(n*n)+1e-12)*10**(snr/10)))
    return x + scale*n

def detector(x, lo, hi, prom):
    sos = butter(3,[lo,hi],btype="bandpass",fs=FS_MODEL,output="sos")
    y = sosfiltfilt(sos,x)
    peaks,_ = find_peaks(y,distance=int(.28*FS_MODEL),prominence=prom*np.std(y))
    peaks = peaks[(peaks >= int(.4*FS_MODEL)) & (peaks <= len(y)-int(.4*FS_MODEL))]
    if len(peaks)<2:
        return np.nan,len(peaks)
    return 60*FS_MODEL/np.median(np.diff(peaks)),len(peaks)

def features(x):
    f,p = welch(x,fs=FS_MODEL,nperseg=min(1000,len(x)))
    total = np.trapz(p,f)+1e-12
    vals=[]
    for lo,hi in BANDS:
        m=(f>=lo)&(f<hi)
        vals.append(np.trapz(p[m],f[m]) if m.sum()>1 else 0.0)
    vals=np.asarray(vals)/total
    dx=np.diff(x)
    h1,c1=detector(x,5,20,.7)
    h2,c2=detector(x,8,30,1.0)
    full=np.r_[vals,np.std(x),kurtosis(x,fisher=True,bias=False),np.std(dx),np.mean(np.abs(dx)),
               0 if not np.isfinite(h1) else h1,0 if not np.isfinite(h2) else h2,
               200 if not(np.isfinite(h1) and np.isfinite(h2)) else abs(h1-h2),c1,c2]
    return np.nan_to_num(full,nan=0,posinf=0,neginf=0),h1,h2,c1,c2

@st.cache_resource(show_spinner="Preparing the prototype models…")
def train_models():
    rng=np.random.default_rng(405)
    Xn,yn,Xr,yr,sub=[],[],[],[],[]
    for sid in range(150):
        hr=rng.uniform(45,140)
        clean,ref=clean_ecg(hr,10000+sid)
        variants=[("Clean",24)]+[(c,s) for c in CLASSES[1:] for s in [-6,0,6,12,18,24]]
        for cls,snr in variants:
            sig=corrupt(clean,cls,snr,rng)
            feat,h1,h2,_,_=features(sig)
            Xn.append(feat[:10]); yn.append(cls)
            Xr.append(feat); yr.append(int(np.isfinite(h1) and abs(h1-ref)<=5)); sub.append(sid)
    Xn,yn,Xr,yr,sub=map(np.asarray,(Xn,yn,Xr,yr,sub))
    train_ids,temp=train_test_split(np.arange(150),test_size=.4,random_state=42)
    val_ids,test_ids=train_test_split(temp,test_size=.5,random_state=42)
    tr=np.isin(sub,train_ids); va=np.isin(sub,val_ids); te=np.isin(sub,test_ids)
    nm=RandomForestClassifier(n_estimators=200,min_samples_leaf=3,class_weight="balanced",
                              max_features="sqrt",random_state=405,n_jobs=-1).fit(Xn[tr],yn[tr])
    rm=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight="balanced",
                              max_features="sqrt",random_state=406,n_jobs=-1).fit(Xr[tr],yr[tr])
    q=rm.predict_proba(Xr[va])[:,list(rm.classes_).index(1)]
    candidates=sorted(set([0.,1.]+list(np.round(np.arange(0,1.001,.01),2))+list(q)))
    valid=[]
    for u in candidates:
        keep=q>=u
        if keep.sum()>=50 and np.mean(1-yr[va][keep])<=.05:
            valid.append((keep.mean(),-u,u))
    threshold=max(valid)[2] if valid else 1.01
    pred=nm.predict(Xn[te])
    testq=rm.predict_proba(Xr[te])[:,list(rm.classes_).index(1)]
    met={"accuracy":accuracy_score(yn[te],pred),"macro_f1":f1_score(yn[te],pred,average="macro"),
         "cm":confusion_matrix(yn[te],pred,labels=CLASSES),"coverage":float(np.mean(testq>=threshold)),
         "test_n":int(te.sum())}
    return nm,rm,threshold,met

def prepare_uploaded_signal(values, source_fs, start_seconds, duration_seconds):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if len(values)<source_fs*3:
        raise ValueError("The selected ECG column must contain at least 3 seconds of valid numeric samples.")
    start=int(start_seconds*source_fs)
    end=min(len(values),int((start_seconds+duration_seconds)*source_fs))
    segment=values[start:end]
    if len(segment)<source_fs*3:
        raise ValueError("Not enough samples remain at the selected start time. Move the start time earlier.")
    # Convert to 250 Hz. Rational polyphase resampling avoids simple point-skipping.
    g=gcd(int(source_fs),FS_MODEL)
    converted=resample_poly(segment,FS_MODEL//g,int(source_fs)//g)
    target=N
    if len(converted)<target:
        raise ValueError(f"Selected segment is too short after resampling. Need approximately {WINDOW_SECONDS} seconds.")
    return converted[:target]

nm,rm,threshold,model_metrics=train_models()

st.header("1. Upload ECG recording")
upload=st.file_uploader("Choose an ECG CSV file",type=["csv"],help="CSV should contain one row per sample and at least one numeric ECG column.")
signal=None
if upload is not None:
    try:
        df=pd.read_csv(upload)
        numeric=df.select_dtypes(include=np.number)
        if numeric.empty:
            st.error("No numeric columns found. Check that the ECG samples are stored as numbers.")
        else:
            a,b,c=st.columns(3)
            with a:
                col=st.selectbox("ECG signal column",numeric.columns.tolist())
            with b:
                source_fs=st.number_input("Sampling frequency (Hz)",min_value=50,max_value=5000,value=250,step=1)
            with c:
                duration=st.selectbox("Analysis window", [10, 15, 20, 30], index=0, help="The model analyzes the first 10 seconds of the selected segment.")
            values=numeric[col].dropna().to_numpy(dtype=float)
            max_start=max(0,len(values)/source_fs-duration)
            start=st.slider("Start time in recording (seconds)",0.0,float(max_start),0.0,step=.5) if max_start>0 else 0.0
            if len(values)>0:
                st.caption(f"{len(values):,} numeric samples detected · approximately {len(values)/source_fs:.1f} seconds total")
            try:
                signal=prepare_uploaded_signal(values,int(source_fs),float(start),10)
                st.success("ECG file loaded. The selected 10-second segment is ready for analysis.")
            except Exception as e:
                st.error(str(e))
    except Exception as e:
        st.error(f"Could not read this CSV: {e}")

st.header("2. Optional symptom questionnaire")
st.caption("Symptoms are recorded as context only. They are not used by the ECG machine-learning models and do not produce a diagnosis.")
with st.form("symptom_form"):
    symptom_options=["Chest pain or discomfort","Shortness of breath","Dizziness or lightheadedness",
                     "Fainting or near-fainting","Palpitations / feeling a racing or irregular heartbeat",
                     "Unusual fatigue","Nausea","None of these","Prefer not to say"]
    selected=st.multiselect("Which symptoms, if any, are being reported?",symptom_options)
    onset=st.selectbox("When did the symptoms begin?",["Not applicable / no symptoms","Just now","Within the last 24 hours","Several days ago","More than a week ago","Unsure"])
    severity=st.select_slider("Self-reported severity",options=["Not applicable","Mild","Moderate","Severe"],value="Not applicable")
    notes=st.text_area("Optional notes (avoid names, addresses, or identifying details)",max_chars=500)
    submitted=st.form_submit_button("Save questionnaire responses")
if submitted:
    st.session_state["symptoms"]={"reported":selected,"onset":onset,"severity":severity,"notes":notes}
    st.success("Questionnaire responses saved for this session.")

symptoms=st.session_state.get("symptoms",{"reported":[],"onset":"Not completed","severity":"Not completed","notes":""})
if any(s in symptoms["reported"] for s in ["Chest pain or discomfort","Shortness of breath","Fainting or near-fainting"]):
    st.error("Safety notice: chest pain, severe breathing difficulty, or fainting can require urgent medical attention. Seek emergency medical care now if symptoms are severe, sudden, ongoing, or worsening. Do not wait for this app's analysis.")
elif symptoms["reported"] and "None of these" not in symptoms["reported"] and "Prefer not to say" not in symptoms["reported"]:
    st.info("If symptoms are concerning, persistent, or worsening, contact a qualified healthcare professional. This application cannot assess their cause.")

st.header("3. Analyze uploaded ECG")
if st.button("Run ECG analysis",type="primary",disabled=(signal is None),use_container_width=True):
    feat,h1,h2,count1,count2=features(signal)
    npred=nm.predict(feat[:10].reshape(1,-1))[0]
    nprob=nm.predict_proba(feat[:10].reshape(1,-1))[0]
    q=float(rm.predict_proba(feat.reshape(1,-1))[0,list(rm.classes_).index(1)])
    accepted=q>=threshold
    st.session_state["analysis"]={
        "noise_class":str(npred),"noise_probs":{str(k):float(v) for k,v in zip(nm.classes_,nprob)},
        "hr_primary":None if not np.isfinite(h1) else float(h1),
        "hr_secondary":None if not np.isfinite(h2) else float(h2),
        "peak_count_primary":int(count1),"peak_count_secondary":int(count2),
        "reliability_probability":q,"threshold":float(threshold),"accepted":bool(accepted),
        "symptoms":symptoms,"sampling_frequency_model_hz":FS_MODEL,
        "created_utc":datetime.now(timezone.utc).isoformat()
    }
    st.session_state["analyzed_signal"]=signal.copy()

result=st.session_state.get("analysis")
if result is not None:
    st.subheader("Analysis report")
    x1,x2,x3=st.columns(3)
    x1.metric("Predicted noise class",result["noise_class"])
    x2.metric("Primary estimated heart rate",f'{result["hr_primary"]:.1f} bpm' if result["hr_primary"] is not None else "Unavailable")
    x3.metric("Reliability probability",f'{result["reliability_probability"]*100:.1f}%')
    x4,x5=st.columns(2)
    x4.metric("Second detector estimate",f'{result["hr_secondary"]:.1f} bpm' if result["hr_secondary"] is not None else "Unavailable")
    x5.metric("Detector disagreement",
              f'{abs(result["hr_primary"]-result["hr_secondary"]):.1f} bpm'
              if result["hr_primary"] is not None and result["hr_secondary"] is not None else "Unavailable")
    if result["accepted"]:
        st.success("RETAINED by the prototype reliability gate")
    else:
        st.warning("FLAGGED by the prototype reliability gate — consider checking signal acquisition and obtaining professional review.")
    st.caption(f'Gate threshold: {result["threshold"]:.4f}. This is a validation-selected experimental threshold, not a clinical standard.')
    p1,p2=st.columns([1.2,1])
    with p1:
        fig,ax=plt.subplots(figsize=(9,3.2))
        ax.plot(np.arange(len(st.session_state["analyzed_signal"]))/FS_MODEL,
                st.session_state["analyzed_signal"],lw=.8)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (input units)")
        ax.set_title("Uploaded ECG segment (resampled to 250 Hz)")
        ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)
    with p2:
        st.markdown("**Noise-class probabilities**")
        probdf=pd.DataFrame({"Noise class":list(result["noise_probs"].keys()),
                             "Probability (%)":[v*100 for v in result["noise_probs"].values()]})
        st.dataframe(probdf.sort_values("Probability (%)",ascending=False),hide_index=True,use_container_width=True)
        st.markdown("**Peak detector counts**")
        st.write(f"Detector 1: {result['peak_count_primary']}")
        st.write(f"Detector 2: {result['peak_count_secondary']}")
    st.markdown("**Symptom questionnaire summary**")
    st.write("Reported symptoms: " + (", ".join(symptoms["reported"]) if symptoms["reported"] else "Not reported"))
    st.write(f"Onset: {symptoms['onset']} · Self-reported severity: {symptoms['severity']}")
    if symptoms["notes"]:
        st.write("Notes: " + symptoms["notes"])
    report_json=json.dumps(result,indent=2)
    st.download_button("Download assessment summary (JSON)",data=report_json,
                       file_name="ecg_assessment_summary.json",mime="application/json")
    st.download_button("Download analyzed ECG segment (CSV)",
                       data=pd.DataFrame({"ecg_resampled_250Hz":st.session_state["analyzed_signal"]}).to_csv(index=False),
                       file_name="analyzed_ecg_segment.csv",mime="text/csv")

with st.expander("Prototype model evaluation (synthetic held-out split)"):
    st.write(f"Noise-classification accuracy: **{model_metrics['accuracy']*100:.1f}%**")
    st.write(f"Noise-classification macro-F1: **{model_metrics['macro_f1']:.3f}**")
    st.write(f"Test-window count: **{model_metrics['test_n']}**")
    st.write(f"Reliability-gate coverage on synthetic test split: **{model_metrics['coverage']*100:.1f}%**")
    st.caption("These are internal metrics on a procedurally generated synthetic test split. They are not performance measurements on the uploaded ECG or on patients.")
    fig,ax=plt.subplots(figsize=(6,4))
    im=ax.imshow(model_metrics["cm"],cmap="Blues")
    ax.set_xticks(range(5),CLASSES,rotation=30,ha="right")
    ax.set_yticks(range(5),CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Synthetic test confusion matrix")
    for i in range(5):
        for j in range(5):
            ax.text(j,i,str(model_metrics["cm"][i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax); st.pyplot(fig); plt.close(fig)

st.divider()
st.caption("Research-use prototype. The ±5 bpm target is an engineering tolerance specified in the project, not a medical acceptance criterion. Noise class and heart-rate reliability are not diagnoses. The app's synthetic training metrics must not be represented as clinical validation or as the paper's reported results.")
