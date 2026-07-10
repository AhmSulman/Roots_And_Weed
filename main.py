"""
Roots and Weed
Cross-platform (Android + Windows) BPM / key finder and stem separator.
Material Design UI via KivyMD.
"""

import os
import threading

import kivy
from kivy.clock import mainthread
from kivy.core.text import LabelBase
from kivy.graphics import Color, Line
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    NumericProperty,
    StringProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivymd.app import MDApp
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem  # noqa: F401 (used by rootsandweed.kv)
from kivymd.uix.boxlayout import MDBoxLayout  # noqa: F401 (used by rootsandweed.kv)
from kivymd.uix.button import MDRaisedButton  # noqa: F401 (used by rootsandweed.kv)
from kivymd.uix.card import MDCard  # noqa: F401 (used by rootsandweed.kv)
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel  # noqa: F401 (used by rootsandweed.kv)
from kivymd.uix.progressbar import MDProgressBar  # noqa: F401 (used by rootsandweed.kv)
from kivymd.uix.toolbar import MDTopAppBar  # noqa: F401 (used by rootsandweed.kv)

from okf.audio_analysis import AudioAnalyzer
from okf.mp3_convert import ConversionError, convert_mp3_to_wav
from okf.separation import is_torch_available, separate_with_cache

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "okf_output")
CONVERTED_DIR = os.path.join(OUTPUT_DIR, "converted")

# Two font families distinct from KivyMD's stock Roboto everywhere, both
# already bundled inside Kivy itself (kivy/data/fonts) -- zero extra assets,
# zero extra APK size, guaranteed present on Android too since it ships as
# part of the core Kivy package.
_KIVY_FONTS_DIR = os.path.join(os.path.dirname(kivy.__file__), "data", "fonts")
LabelBase.register(
    name="HeadingFont",
    fn_regular=os.path.join(_KIVY_FONTS_DIR, "DejaVuSans.ttf"),
)
LabelBase.register(
    name="ReadoutFont",
    fn_regular=os.path.join(_KIVY_FONTS_DIR, "RobotoMono-Regular.ttf"),
)


def uses_desktop_separation():
    return platform not in ("android", "ios") and is_torch_available()


class WaveformView(BoxLayout):
    waveform_data = ListProperty([])

    def on_waveform_data(self, *args):
        self.canvas.after.clear()
        if not self.waveform_data:
            return
        with self.canvas.after:
            Color(0.86, 0.24, 0.24, 1)  # muted red waveform trace
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
    status_text = StringProperty("Pick a WAV or MP3 file to begin.")
    bpm_text = StringProperty("--")
    key_text = StringProperty("--")
    confidence_text = StringProperty("")
    separation_mode_text = StringProperty("")
    waveform_data = ListProperty([])
    is_busy = BooleanProperty(False)
    current_filepath = StringProperty("")
    separating = BooleanProperty(False)
    separation_progress = NumericProperty(0)  # 0-100, drives the loading bar

    convert_status_text = StringProperty(
        "Optional: convert an MP3 to a WAV file on disk. Analysis reads MP3 directly."
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
            ext=[".wav", ".mp3"],
        )

    # -- shared file manager, filtered per tab -----------------------------

    def open_wav_chooser(self):
        self._fm_mode = "wav"
        self.file_manager.ext = [".wav", ".mp3"]
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
        key_line = f"{result['key']}  ·  rel. {result['relative_key']}"
        if result["key_confidence"] < 0.05 and result["alternate_key"] != result["relative_key"]:
            key_line += f"  (or {result['alternate_key']}?)"
        self.key_text = key_line
        self.confidence_text = (
            f"tempo confidence {result['bpm_confidence']:.2f}  ·  "
            f"key confidence {result['key_confidence']:.2f}"
        )
        self.waveform_data = result["waveform_data"]
        self.status_text = (
            f"Done in {result['analysis_seconds']:.2f}s · "
            f"{result['duration']:.1f}s · {result['sample_rate']} Hz"
        )

    def run_separation(self):
        if self.is_busy or not self.current_filepath:
            return
        self.is_busy = True
        self.separating = True
        self.separation_progress = 0
        desktop_mode = uses_desktop_separation()
        self.separation_mode_text = (
            "Demucs (desktop, vocals + instrumental)"
            if desktop_mode
            else "DSP fallback (vocals + instrumental, Android-safe)"
        )
        bar = self.ids.sep_progress
        if desktop_mode:
            # Demucs exposes no progress hook: animate instead of guessing
            bar.type = "indeterminate"
            bar.start()
        else:
            bar.type = None  # determinate: value driven by real DSP stages
        self.status_text = "Separating stems... this can take a while."
        threading.Thread(
            target=self._separate_thread, args=(self.current_filepath, desktop_mode), daemon=True
        ).start()

    @mainthread
    def _set_separation_progress(self, fraction):
        self.separation_progress = fraction * 100.0

    def _separate_thread(self, filepath, desktop_mode):
        try:
            stems, meta, from_cache = separate_with_cache(
                filepath, OUTPUT_DIR, desktop_mode,
                progress=self._set_separation_progress,
            )
            self._on_separation_done(stems, None, meta, from_cache)
        except Exception as e:
            self._on_separation_done(None, str(e), None, False)

    @mainthread
    def _on_separation_done(self, stems, error, meta, from_cache):
        self.is_busy = False
        self.separating = False
        bar = self.ids.sep_progress
        if bar.type == "indeterminate":
            bar.stop()
        if error:
            self.separation_progress = 0
            self.status_text = f"Separation error: {error}"
            return
        self.separation_progress = 100
        names = ", ".join(stems.keys())
        took = meta["elapsed_seconds"]
        if from_cache:
            self.status_text = (
                f"Stems retrieved from cache (originally {took:.1f}s): {names}"
            )
        else:
            self.status_text = f"Stems ready in {took:.1f}s ({OUTPUT_DIR}): {names}"


class RootsAndWeedApp(MDApp):
    def build(self):
        self.title = "Roots and Weed"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
        self.theme_cls.primary_hue = "700"  # deep, muted red rather than neon
        self.theme_cls.accent_palette = "DeepOrange"
        self.theme_cls.accent_hue = "700"
        return RootWidget()


if __name__ == "__main__":
    RootsAndWeedApp().run()
