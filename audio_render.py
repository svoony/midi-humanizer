"""Renders a predicted performance through FluidSynth offline, instead of
resynthesizing it live in the browser's GM soundfont. FluidSynth honors MIDI
sustain-pedal CC64 natively, so the PrettyMIDI performance is written out as
a normal .mid file and rendered as-is - no custom pedal-extension logic
needed server-side, unlike the browser's old setTimeout-driven player.
"""
import os
import subprocess
import tempfile
import wave

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLUIDSYNTH_EXE = os.path.join(
    BASE_DIR, "vendor", "fluidsynth", "fluidsynth-v2.5.5-win10-x64-cpp11", "bin", "fluidsynth.exe"
)
SOUNDFONT_PATH = os.path.join(BASE_DIR, "vendor", "soundfonts", "MuseScore_General.sf2")
SAMPLE_RATE = 44100
# FluidSynth's default gain (0.2) is deliberately conservative to avoid
# clipping under dense polyphony, which makes renders much quieter than the
# browser's flat-input playback. Normalize per-piece instead of using one
# fixed gain, since a fixed value would clip loud pieces and stay quiet on
# soft ones.
NORMALIZE_TARGET_PEAK = 0.9
MAX_NORMALIZE_GAIN = 80.0


def _normalize_wav_peak(wav_path):
    with wave.open(wav_path, "rb") as w:
        params = w.getparams()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    peak = np.abs(samples).max()
    if peak == 0:
        return
    gain = min((NORMALIZE_TARGET_PEAK * 32767) / peak, MAX_NORMALIZE_GAIN)
    boosted = np.clip(samples.astype(np.float64) * gain, -32768, 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as w:
        w.setparams(params)
        w.writeframes(boosted.tobytes())


def render_pm_to_wav_bytes(pm):
    """Renders a pretty_midi.PrettyMIDI performance (notes + sustain CCs)
    through FluidSynth and returns the result as WAV bytes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        midi_path = os.path.join(tmp_dir, "in.mid")
        wav_path = os.path.join(tmp_dir, "out.wav")
        pm.write(midi_path)

        subprocess.run(
            [FLUIDSYNTH_EXE, "-ni", "-F", wav_path, "-r", str(SAMPLE_RATE), SOUNDFONT_PATH, midi_path],
            check=True, capture_output=True,
        )
        _normalize_wav_peak(wav_path)

        with open(wav_path, "rb") as f:
            return f.read()
