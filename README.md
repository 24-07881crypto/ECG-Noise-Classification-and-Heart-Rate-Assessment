# Upload-based ECG Assessment Application

## Features
- Upload an ECG CSV file.
- Select the numeric ECG column and sampling frequency.
- Select the starting point of a 10-second analysis window.
- Complete an optional symptom questionnaire.
- Display the uploaded waveform.
- Classify likely noise as Clean, Baseline, Muscle, Power line, or Motion.
- Estimate heart rate with two peak detectors.
- Estimate probability that the primary detector is within the research project's ±5 bpm tolerance.
- Apply a validation-selected experimental retain/flag threshold.
- Download the assessment summary as JSON and the analyzed segment as CSV.

## Run locally
Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## CSV input format
- One row per time sample.
- At least one numeric column containing ECG samples.
- Enter the sampling frequency in Hz.
- Use at least 10 seconds of data for a full analysis window.
- The app resamples the selected window to 250 Hz for its current feature and detector pipeline.
- Avoid including names, dates of birth, addresses, or other identifying details.

## Critical limitations
This is a research/educational prototype, not a diagnostic device. Symptoms are not used by the ML models and the app does not diagnose disease or determine medical safety.

The starter model is trained using procedurally generated synthetic PQRST signals and procedural noise. It does not yet use the MIT-BIH Noise Stress Test Database recordings (`bw`, `ma`, `em`) described in the research paper. Predictions on real uploaded ECG files are exploratory and may not generalize. Before reporting real-world performance, integrate the intended NSTDB sources with subject/source-time-disjoint partitions and evaluate on appropriate independent ECG data.

The ±5 bpm reliability target is an engineering research tolerance, not a clinical standard. Do not use the output to make care decisions or delay emergency assessment.
