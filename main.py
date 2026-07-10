"""
OKF - Octave Key Finder
Cross-platform (Android + Windows) BPM / key finder and stem separator.
Material Design UI via KivyMD.
"""

import os
import threading

from kivy.clock import mainthread
from kivy.graphics import Color, Line
from kivy.properties import BooleanProperty, ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivymd.app import MDApp
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem  # noqa: F401 (used by okf.kv)
from kivymd.uix.boxlayout import MDBoxLayout  # noqa: F401 (used by okf.kv)
from kivymd.uix.button import MDRaisedButton  # noqa: F401 (used by okf.kv)
from kivymd.uix.card import MDCard  # noqa: F401 (used by okf.kv)
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel  # noqa: F401 (used by okf.kv)
from kivymd.uix.toolbar import MDTopAppBar  # noqa: F401 (used by okf.kv)

from okf.audio_analysis import AudioAnalyzer
from okf.mp3_convert import ConversionError, convert_mp3_to_wav
from okf.separation import is_torch_available, separate_desktop, separate_dsp

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "okf_output")
CONVERTED_DIR = os.path.join(OUTPUT_DIR, "converted")


def uses_desktop_separation():
    return platform not in ("android", "ios") and is_torch_available()


class WaveformView(BoxLayout):
    waveform_data = ListProperty([])

    def on_waveform_data(self, *args):
        self.canvas.after.clear()
        if not self.waveform_data:
            return
        with self.canvas.after:
            Color(0.40, 0.78, 1.0, 1)
            w, h = self.width, self.height
            n = len(self.waveform_data)
            points = []
            for i, sample in enumerate(self.waveform_data):
                x = self.x + (i / max(1, n - 1)) * w
                y = self.y + (h / 2) + (sample * h / 2 * 0.9)
                points.extend([x, y])
            if len(points) >= 4:
                Line(points=points, width=1.2)

    def on_size(self, *args):
        self.on_waveform_data()

    def on_pos(self, *args):
        self.on_waveform_data()


class RootWidget(BoxLayout):
    status_text = StringProperty("Pick a WAV file to begin.")
    bpm_text = StringProperty("--")
    key_text = StringProperty("--")
    confidence_text = StringProperty("")
    separation_mode_text = StringProperty("")
    waveform_data = ListProperty([])
    is_busy = BooleanProperty(False)
    current_filepath = StringProperty("")

    convert_status_text = StringProperty(
        "The analyzer only reads WAV files. Pick an MP3 here to convert it first."
    )
    convert_source_name = StringProperty("")
    converted_wav_path = StringProperty("")
    convert_busy = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fm_mode = "wav"
        self.file_manager = MDFileManager(
            exit_manager=self.exit_file_manager,
            select_path=self.select_path,
            ext=[".wav"],
        )

    # -- shared file manager, filtered per tab -----------------------------

    def open_wav_chooser(self):
        self._fm_mode = "wav"
        self.file_manager.ext = [".wav"]
        self.file_manager.show(os.path.expanduser("~"))

    def open_mp3_chooser(self):
        self._fm_mode = "mp3"
        self.file_manager.ext = [".mp3"]
        self.file_manager.show(os.path.expanduser("~"))

    def exit_file_manager(self, *args):
        self.file_manager.close()

    def select_path(self, path):
        self.exit_file_manager()
        if self._fm_mode == "mp3":
            self.convert_source_name = os.path.basename(path)
            self.convert_status_text = f"Selected {self.convert_source_name}. Ready to convert."
            self._mp3_source_path = path
            self.converted_wav_path = ""
        else:
            self.current_filepath = path
            self.run_analysis(path)

    # -- MP3 -> WAV conversion tab -----------------------------------------

    def run_conversion(self):
        source = getattr(self, "_mp3_source_path", "")
        if self.convert_busy or not source:
            return
        self.convert_busy = True
        self.convert_status_text = f"Converting {os.path.basename(source)}..."
        threading.Thread(target=self._convert_thread, args=(source,), daemon=True).start()

    def _convert_thread(self, source):
        try:
            os.makedirs(CONVERTED_DIR, exist_ok=True)
            base = os.path.splitext(os.path.basename(source))[0]
            out_path = os.path.join(CONVERTED_DIR, f"{base}.wav")
            convert_mp3_to_wav(source, out_path)
            self._on_conversion_done(out_path, None)
        except ConversionError as e:
            self._on_conversion_done(None, str(e))
        except Exception as e:
            self._on_conversion_done(None, f"Unexpected error: {e}")

    @mainthread
    def _on_conversion_done(self, out_path, error):
        self.convert_busy = False
        if error:
            self.convert_status_text = f"Conversion failed: {error}"
            return
        self.converted_wav_path = out_path
        self.convert_status_text = f"Converted -> {os.path.basename(out_path)}. Ready to analyze."

    def use_converted_in_analyzer(self):
        if not self.converted_wav_path:
            return
        self.current_filepath = self.converted_wav_path
        if "bottom_nav" in self.ids:
            self.ids.bottom_nav.switch_tab("analyze")
        self.run_analysis(self.converted_wav_path)

    def run_analysis(self, filepath):
        if self.is_busy:
            return
        self.is_busy = True
        self.status_text = f"Analyzing {os.path.basename(filepath)}..."
        threading.Thread(target=self._analyze_thread, args=(filepath,), daemon=True).start()

    def _analyze_thread(self, filepath):
        try:
            result = AudioAnalyzer.analyze_file(filepath)
            self._on_analysis_done(result, None)
        except Exception as e:
            self._on_analysis_done(None, str(e))

    @mainthread
    def _on_analysis_done(self, result, error):
        self.is_busy = False
        if error:
            self.status_text = f"Error: {error}"
            return

        self.bpm_text = f"{result['bpm']:.1f}"
        key_line = result["key"]
        if result["key_confidence"] < 0.05:
            key_line += f"  (or {result['alternate_key']}?)"
        self.key_text = key_line
        self.confidence_text = (
            f"tempo confidence {result['bpm_confidence']:.2f}  ·  "
            f"key confidence {result['key_confidence']:.2f}"
        )
        self.waveform_data = result["waveform_data"]
        self.status_text = (
            f"Done · {result['duration']:.1f}s · {result['sample_rate']} Hz"
        )

    def run_separation(self):
        if self.is_busy or not self.current_filepath:
            return
        self.is_busy = True
        desktop_mode = uses_desktop_separation()
        self.separation_mode_text = (
            "Demucs (desktop, 4 stems)" if desktop_mode else "DSP fallback (Android-safe)"
        )
        self.status_text = "Separating stems... this can take a while."
        threading.Thread(
            target=self._separate_thread, args=(self.current_filepath, desktop_mode), daemon=True
        ).start()

    def _separate_thread(self, filepath, desktop_mode):
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            if desktop_mode:
                stems = separate_desktop(filepath, OUTPUT_DIR)
            else:
                stems = separate_dsp(filepath, OUTPUT_DIR)
            self._on_separation_done(stems, None)
        except Exception as e:
            self._on_separation_done(None, str(e))

    @mainthread
    def _on_separation_done(self, stems, error):
        self.is_busy = False
        if error:
            self.status_text = f"Separation error: {error}"
            return
        names = ", ".join(stems.keys())
        self.status_text = f"Stems ready in {OUTPUT_DIR}: {names}"


class OKFApp(MDApp):
    def build(self):
        self.title = "OKF - Octave Key Finder"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.accent_palette = "Amber"
        return RootWidget()


if __name__ == "__main__":
    OKFApp().run()
