"""
UNISPEAK UI — Python Tkinter recreation of the HTML/CSS design.
Matches the dark theme, sidebar, camera view, translation panel, history,
and all interactive elements (mode switching, copy text, start/stop camera).

Uses pure tkinter for maximum compatibility.
"""

import os
import sys
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageTk, ImageFont, ImageFilter
import cv2
import threading
import math
import random

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ─── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.jpeg")
LOGO_FALLBACK = "/Users/daiyaan/Downloads/WhatsApp Image 2026-04-10 at 12.04.12.jpeg"
AVATAR_PATH = os.path.join(ASSETS_DIR, "user_avatar.jpg")
CAMERA_BG_PATH = os.path.join(ASSETS_DIR, "camera_bg.jpg")

# ─── Color Palette (matching CSS vars) ────────────────────────────────
C = {
    "primary":       "#38bdf8",
    "primary_hover": "#2da3db",
    "bg_body":       "#0b0f19",
    "bg_sidebar":    "#111827",
    "text_dark":     "#f8fafc",
    "text_gray":     "#94a3b8",
    "border":        "#1e293b",
    "bg_card":       "#161e2d",
    "red":           "#ff3b3b",
    "muted":         "#8c93a1",
    "search_text":   "#a0a8b9",
    "pill_bg":       "#111827",
}


def make_circular(img, size):
    """Return a circular-cropped PIL Image of given size."""
    img = img.resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size - 1, size - 1), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img.convert("RGBA"), (0, 0), mask)
    return out


def load_image(path, fallback=None, size=None, circular=False):
    """Load an image, optionally resize / crop circular."""
    for p in [path, fallback]:
        if p and os.path.isfile(p):
            try:
                img = Image.open(p).convert("RGBA")
                if circular and size:
                    return make_circular(img, size[0])
                if size:
                    return img.resize(size, Image.LANCZOS)
                return img
            except Exception:
                continue
    return None


def make_rounded_rect(width, height, radius, fill, border=None, border_width=1):
    """Create a rounded-rectangle PIL Image."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if border:
        draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius,
                               fill=fill, outline=border, width=border_width)
    else:
        draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius, fill=fill)
    return img


class UnispeakApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("UNISPEAK")
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)
        self.root.configure(bg=C["bg_body"])

        # Try to load Inter font, fallback to system font
        available = tkfont.families()
        if "Inter" in available:
            self.FONT = "Inter"
        elif "SF Pro Display" in available:
            self.FONT = "SF Pro Display"
        elif "Helvetica Neue" in available:
            self.FONT = "Helvetica Neue"
        else:
            self.FONT = "Helvetica"

        # State
        self.current_mode = "sign"
        self.camera_running = False
        self.cap = None
        self._camera_after_id = None
        self._photo_refs = []  # prevent GC

        # Speech-to-text state
        self.stt_recording = False
        self._stt_thread = None
        self._stt_pulse_id = None
        self._stt_wave_id = None
        self._pulse_phase = 0
        self._wave_bars = []
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None

        # Load assets
        self._load_assets()

        # Build UI
        self._build_sidebar()
        self._build_main()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────── ASSETS ──────────────────────────────
    def _load_assets(self):
        logo = load_image(LOGO_PATH, LOGO_FALLBACK, size=(80, 80), circular=True)
        self.logo_photo = ImageTk.PhotoImage(logo) if logo else None

        avatar = load_image(AVATAR_PATH, size=(36, 36), circular=True)
        self.avatar_photo = ImageTk.PhotoImage(avatar) if avatar else None

        # No background image — blank camera area with placeholder text

    # ──────────────────────────── SIDEBAR ─────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=C["bg_sidebar"], width=250)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # ─ Logo ─
        logo_frame = tk.Frame(sb, bg=C["bg_sidebar"])
        logo_frame.pack(fill="x", padx=24, pady=(24, 8))

        if self.logo_photo:
            tk.Label(logo_frame, image=self.logo_photo, bg=C["bg_sidebar"]).pack(anchor="w")
        else:
            c = tk.Canvas(logo_frame, width=80, height=80, bg=C["bg_sidebar"],
                          highlightthickness=0)
            c.pack(anchor="w")
            c.create_oval(2, 2, 78, 78, fill="white", outline="white")

        tk.Label(logo_frame, text="UNISPEAK", bg=C["bg_sidebar"],
                 fg=C["text_dark"], font=(self.FONT, 20, "bold")).pack(anchor="w", pady=(12, 0))

        # ─ Menu ─
        menu_frame = tk.Frame(sb, bg=C["bg_sidebar"])
        menu_frame.pack(fill="x", pady=(20, 0))

        menu_items = [
            ("sign",     "🎥   Sign Language"),
            ("braille",  "⠿   Braille Detection"),
            ("speech",   "🎤   Speech to Text"),
            ("history",  "🕐   History"),
            ("settings", "⚙️   Settings"),
        ]

        self.menu_buttons = {}
        for key, text in menu_items:
            btn = tk.Label(
                menu_frame, text=text, bg=C["bg_sidebar"], fg=C["text_gray"],
                font=(self.FONT, 14, "bold"), anchor="w", padx=24, pady=14,
                cursor="hand2",
            )
            btn.pack(fill="x")
            btn.bind("<Button-1>", lambda e, k=key: self._on_menu_click(k))
            btn.bind("<Enter>", lambda e, b=btn: self._menu_hover(b, True))
            btn.bind("<Leave>", lambda e, b=btn, k=key: self._menu_hover(b, False, k))
            self.menu_buttons[key] = btn

        self._set_active("sign")

        # ─ Bottom user ─
        spacer = tk.Frame(sb, bg=C["bg_sidebar"])
        spacer.pack(fill="both", expand=True)

        user_frame = tk.Frame(sb, bg=C["bg_sidebar"])
        user_frame.pack(fill="x", padx=24, pady=24)

        if self.avatar_photo:
            tk.Label(user_frame, image=self.avatar_photo,
                     bg=C["bg_sidebar"]).pack(side="left", padx=(0, 12))
        else:
            c = tk.Canvas(user_frame, width=36, height=36, bg=C["bg_sidebar"],
                          highlightthickness=0)
            c.pack(side="left", padx=(0, 12))
            c.create_oval(1, 1, 35, 35, fill=C["primary"], outline=C["primary"])

        info = tk.Frame(user_frame, bg=C["bg_sidebar"])
        info.pack(side="left")
        tk.Label(info, text="Alex Rivera", bg=C["bg_sidebar"], fg=C["text_dark"],
                 font=(self.FONT, 13, "bold")).pack(anchor="w")
        tk.Label(info, text="Premium User", bg=C["bg_sidebar"], fg=C["text_gray"],
                 font=(self.FONT, 11)).pack(anchor="w")

    def _set_active(self, key):
        for k, btn in self.menu_buttons.items():
            if k == key:
                btn.configure(bg=C["bg_body"], fg=C["primary"])
            else:
                btn.configure(bg=C["bg_sidebar"], fg=C["text_gray"])

    def _menu_hover(self, btn, entering, key=None):
        if entering:
            if btn.cget("fg") != C["primary"]:
                btn.configure(bg="#151c2a")
        else:
            if key and key == self.current_mode:
                btn.configure(bg=C["bg_body"])
            elif key in ("sign", "braille", "speech", "history", "settings") and key != self.current_mode:
                btn.configure(bg=C["bg_sidebar"])
            else:
                btn.configure(bg=C["bg_sidebar"])

    def _on_menu_click(self, key):
        if key in ("sign", "braille", "speech", "history", "settings"):
            self.current_mode = key
            self._set_active(key)
            self._switch_mode(key)

    # ──────────────────────────── MODE SWITCH ────────────────────────
    def _switch_mode(self, mode):
        # Stop any running processes when switching away
        if mode != "speech" and self.stt_recording:
            self._stop_stt_recording()
        if mode not in ("sign", "braille") and self.camera_running:
            self._stop_camera()

        # ── Hide everything first ──
        self.history_full_panel.pack_forget()
        self.settings_panel.pack_forget()
        self.center_frame.pack_forget()
        self.right_history_frame.pack_forget()

        if mode == "history":
            self.history_full_panel.pack(fill="both", expand=True)

        elif mode == "settings":
            self.settings_panel.pack(fill="both", expand=True)

        else:
            # Restore center + right sidebar layout
            self.center_frame.pack(side="left", fill="both", expand=True, padx=(0, 24))
            self.right_history_frame.pack(side="right", fill="y")

            # Reset all center children, then re-show the correct ones
            self.speech_panel.pack_forget()
            self.camera_outer.pack_forget()
            self.trans_outer.pack_forget()
            self.bottom_btn_frame.pack_forget()

            if mode == "speech":
                self.speech_panel.pack(fill="both", expand=True)
                self.trans_outer.pack(fill="x", pady=(16, 0))
                self.live_label.configure(text="  SPEECH TO TEXT")
                self.trans_text.configure(
                    text="Press the microphone button to start recording your speech."
                )
            else:
                self.camera_outer.pack(fill="both", expand=True)
                self.trans_outer.pack(fill="x", pady=(16, 0))
                self.bottom_btn_frame.pack(fill="x", pady=(16, 0))

                if mode == "sign":
                    self.live_label.configure(text="  LIVE SIGN ANALYSIS")
                    self.trans_text.configure(
                        text="Hello, I am currently demonstrating the new translation engine. "
                             "The system is accurately detecting my movements."
                    )
                else:
                    self.live_label.configure(text="  LIVE BRAILLE DETECTION")
                    self.trans_text.configure(
                        text='Scanning... Braille cells mapped successfully. '
                             'Transcript: "Welcome to the digital monolith."'
                    )

    # ──────────────────────────── MAIN AREA ──────────────────────────
    def _build_main(self):
        main = tk.Frame(self.root, bg=C["bg_body"])
        main.pack(side="left", fill="both", expand=True)

        # ─── Top bar ─────────────────────────────────────────
        topbar = tk.Frame(main, bg=C["bg_body"], height=80)
        topbar.pack(fill="x", padx=40, pady=(10, 10))
        topbar.pack_propagate(False)

        # Search
        search_frame = tk.Frame(topbar, bg=C["bg_card"], highlightbackground=C["border"],
                                highlightthickness=1, bd=0)
        search_frame.pack(side="left", pady=18)

        tk.Label(search_frame, text="🔍", bg=C["bg_card"], fg=C["search_text"],
                 font=(self.FONT, 13)).pack(side="left", padx=(12, 4))

        self.search_entry = tk.Entry(
            search_frame, bg=C["bg_card"], fg=C["text_dark"],
            insertbackground=C["text_dark"], font=(self.FONT, 13),
            relief="flat", width=35, bd=0,
        )
        self.search_entry.pack(side="left", padx=(0, 12), pady=10, ipady=2)
        self.search_entry.insert(0, "Search archive...")
        self.search_entry.configure(fg=C["search_text"])
        self.search_entry.bind("<FocusIn>", self._search_focus_in)
        self.search_entry.bind("<FocusOut>", self._search_focus_out)

        # Top-right icons
        icons_frame = tk.Frame(topbar, bg=C["bg_body"])
        icons_frame.pack(side="right", pady=18)
        for icon in ["📹", "📶", "👤"]:
            tk.Label(icons_frame, text=icon, bg=C["bg_body"], fg=C["search_text"],
                     font=(self.FONT, 16), cursor="hand2").pack(side="left", padx=10)

        # ─── Content ─────────────────────────────────────────
        content = tk.Frame(main, bg=C["bg_body"])
        content.pack(fill="both", expand=True, padx=40, pady=(0, 30))
        self.content_frame = content

        # Center column
        center = tk.Frame(content, bg=C["bg_body"])
        center.pack(side="left", fill="both", expand=True, padx=(0, 24))
        self.center_frame = center

        # Camera
        self._build_camera(center)
        # Speech-to-text panel (hidden by default)
        self._build_speech_panel(center)
        # Translation box
        self._build_translation(center)
        # Bottom buttons
        self._build_bottom_buttons(center)

        # Right column (sidebar history)
        self._build_history(content)

        # Full-page panels (hidden by default)
        self._build_history_full_panel(content)
        self._build_settings_panel(content)

    def _search_focus_in(self, e):
        if self.search_entry.get() == "Search archive...":
            self.search_entry.delete(0, tk.END)
            self.search_entry.configure(fg=C["text_dark"])

    def _search_focus_out(self, e):
        if not self.search_entry.get():
            self.search_entry.insert(0, "Search archive...")
            self.search_entry.configure(fg=C["search_text"])

    # ─── Camera ──────────────────────────────────────────────────────
    def _build_camera(self, parent):
        cam_outer = tk.Frame(parent, bg=C["bg_body"])
        self.camera_outer = cam_outer
        cam_outer.pack(fill="both", expand=True)

        self.camera_canvas = tk.Canvas(cam_outer, bg="#1c2732", highlightthickness=0, bd=0)
        self.camera_canvas.pack(fill="both", expand=True)

        # Placeholder text shown when camera is off
        self._placeholder_id = self.camera_canvas.create_text(
            0, 0, text="📷  Start Camera to Translate",
            fill=C["text_gray"], font=(self.FONT, 18), tags="placeholder"
        )

        self.camera_canvas.bind("<Configure>", self._on_camera_resize)

        # Overlay: live badge
        badge_frame = tk.Frame(self.camera_canvas, bg="#1a2233")

        self.badge_dot = tk.Label(badge_frame, text="●", fg=C["red"], bg="#1a2233",
                                  font=(self.FONT, 8))
        self.badge_dot.pack(side="left", padx=(10, 4), pady=5)

        self.live_label = tk.Label(badge_frame, text="  LIVE SIGN ANALYSIS",
                                   fg="white", bg="#1a2233",
                                   font=(self.FONT, 9, "bold"))
        self.live_label.pack(side="left", padx=(0, 10), pady=5)

        self.badge_window = self.camera_canvas.create_window(0, 16, anchor="ne",
                                                              window=badge_frame)

        # Overlay: controls pill
        pill_frame = tk.Frame(self.camera_canvas, bg=C["pill_bg"])

        # Red stop circle
        stop_btn = tk.Label(pill_frame, text="⏹", fg="white", bg=C["red"],
                            font=(self.FONT, 10), cursor="hand2", padx=6, pady=2)
        stop_btn.pack(side="left", padx=(16, 8), pady=8)
        stop_btn.bind("<Button-1>", lambda e: self._stop_camera())

        # Mic
        mic_btn = tk.Label(pill_frame, text="🎙️", bg=C["pill_bg"],
                           font=(self.FONT, 14), cursor="hand2")
        mic_btn.pack(side="left", padx=8, pady=8)

        # Equalizer / settings
        eq_btn = tk.Label(pill_frame, text="☰", bg=C["pill_bg"], fg=C["text_dark"],
                          font=(self.FONT, 14), cursor="hand2")
        eq_btn.pack(side="left", padx=(8, 16), pady=8)

        self.pill_window = self.camera_canvas.create_window(0, 0, anchor="s",
                                                             window=pill_frame)

    def _on_camera_resize(self, event):
        w, h = event.width, event.height
        # Reposition badge top-right
        self.camera_canvas.coords(self.badge_window, w - 16, 16)
        # Reposition pill bottom-center
        self.camera_canvas.coords(self.pill_window, w // 2, h - 20)
        # Keep placeholder centered
        self.camera_canvas.coords(self._placeholder_id, w // 2, h // 2)

    # ─── Translation Box ─────────────────────────────────────────────
    def _build_translation(self, parent):
        trans_outer = tk.Frame(parent, bg=C["bg_card"], highlightbackground=C["border"],
                               highlightthickness=1, bd=0)
        trans_outer.pack(fill="x", pady=(16, 0))
        self.trans_outer = trans_outer

        header = tk.Frame(trans_outer, bg=C["bg_card"])
        header.pack(fill="x", padx=24, pady=(20, 10))

        tk.Label(header, text="REAL-TIME TRANSLATION", bg=C["bg_card"],
                 fg=C["muted"], font=(self.FONT, 10, "bold")).pack(side="left")

        copy_btn = tk.Label(header, text="📋 Copy Text", bg=C["bg_card"],
                            fg=C["primary"], font=(self.FONT, 11, "bold"),
                            cursor="hand2")
        copy_btn.pack(side="right")
        copy_btn.bind("<Button-1>", lambda e: self._copy_text())

        self.trans_text = tk.Label(
            trans_outer,
            text="Hello, I am currently demonstrating the new translation engine. "
                 "The system is accurately detecting my movements.",
            bg=C["bg_card"], fg=C["text_dark"],
            font=(self.FONT, 18), wraplength=650, justify="left", anchor="w",
        )
        self.trans_text.pack(fill="x", padx=24, pady=(4, 24))

    # ─── Bottom Buttons ──────────────────────────────────────────────
    def _build_bottom_buttons(self, parent):
        btn_frame = tk.Frame(parent, bg=C["bg_body"])
        btn_frame.pack(fill="x", pady=(16, 0))
        self.bottom_btn_frame = btn_frame

        # Start Camera
        self.start_btn = tk.Label(
            btn_frame, text="📹  Start Camera", bg=C["primary"], fg="white",
            font=(self.FONT, 14, "bold"), cursor="hand2", pady=14,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.start_btn.bind("<Button-1>", lambda e: self._start_camera())
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.configure(bg=C["primary_hover"]))
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.configure(bg=C["primary"]))

        # Stop Session
        self.stop_btn = tk.Label(
            btn_frame, text="🔴  Stop Session", bg=C["bg_card"], fg=C["text_dark"],
            font=(self.FONT, 14, "bold"), cursor="hand2", pady=14,
            highlightbackground=C["border"], highlightthickness=1,
        )
        self.stop_btn.pack(side="right", fill="x", expand=True, padx=(8, 0))
        self.stop_btn.bind("<Button-1>", lambda e: self._stop_camera())
        self.stop_btn.bind("<Enter>", lambda e: self.stop_btn.configure(bg=C["border"]))
        self.stop_btn.bind("<Leave>", lambda e: self.stop_btn.configure(bg=C["bg_card"]))

    # ─── History ─────────────────────────────────────────────────────
    def _build_history(self, parent):
        right = tk.Frame(parent, bg=C["bg_body"], width=280)
        self.right_history_frame = right
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Title row
        title_row = tk.Frame(right, bg=C["bg_body"])
        title_row.pack(fill="x", pady=(0, 12))

        tk.Label(title_row, text="HISTORY", bg=C["bg_body"], fg=C["text_dark"],
                 font=(self.FONT, 14, "bold")).pack(side="left")
        tk.Label(title_row, text="⋮", bg=C["bg_body"], fg=C["text_gray"],
                 font=(self.FONT, 16, "bold"), cursor="hand2").pack(side="right")

        # Scrollable area
        canvas = tk.Canvas(right, bg=C["bg_body"], highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(right, orient="vertical", command=canvas.yview)
        self.history_inner = tk.Frame(canvas, bg=C["bg_body"])

        self.history_inner.bind("<Configure>",
                                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.history_inner, anchor="nw", tags="inner")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        # Hide scrollbar for cleaner look, enable mouse wheel
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Bind canvas resize to resize inner frame width
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure("inner", width=e.width))

        history_data = [
            {"time": "10:42 AM", "text": "Where is the nearest subway station please?", "active": False},
            {"time": "09:15 AM", "text": "I would like to order a black coffee with no sugar.", "active": False},
            {"time": "YESTERDAY", "text": "Meeting scheduled for tomorrow at the main office.", "active": False},
            {"time": "YESTERDAY", "text": "Thank you very much for your help today.", "active": True},
        ]

        for item in history_data:
            self._make_history_card(self.history_inner, item)

    def _make_history_card(self, parent, data):
        active = data.get("active", False)
        bg = C["bg_body"] if active else C["bg_card"]
        border_col = C["primary"] if active else C["border"]

        card = tk.Frame(parent, bg=bg, highlightbackground=border_col,
                        highlightthickness=1, bd=0)
        card.pack(fill="x", pady=(0, 12))

        inner = tk.Frame(card, bg=bg)
        inner.pack(fill="x", padx=16, pady=14)

        # Top row
        top = tk.Frame(inner, bg=bg)
        top.pack(fill="x")

        time_color = C["primary"] if active else C["muted"]
        tk.Label(top, text=data["time"], bg=bg, fg=time_color,
                 font=(self.FONT, 9, "bold")).pack(side="left")

        icon = "🔖" if active else "↗"
        icon_fg = C["primary"] if active else C["muted"]
        tk.Label(top, text=icon, bg=bg, fg=icon_fg,
                 font=(self.FONT, 11)).pack(side="right")

        # Text
        txt = tk.Label(inner, text=data["text"], bg=bg, fg=C["text_dark"],
                       font=(self.FONT, 13), wraplength=230, justify="left", anchor="w")
        txt.pack(fill="x", pady=(8, 0))

        # Left accent bar for active card
        if active:
            accent = tk.Frame(card, bg=C["primary"], width=3)
            accent.place(x=0, y=0, relheight=1.0)

        # Click handler
        def on_click(event):
            self.trans_text.configure(text=data["text"])

        for widget in [card, inner, top, txt]:
            widget.bind("<Button-1>", on_click)
            widget.configure(cursor="hand2")

    # ──────────────────────────── COPY TEXT ───────────────────────────
    def _copy_text(self):
        text = self.trans_text.cget("text")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        # Flash feedback
        self.trans_text.configure(fg=C["primary"])
        self.root.after(400, lambda: self.trans_text.configure(fg=C["text_dark"]))

    # ──────────────────────────── CAMERA ─────────────────────────────
    def _start_camera(self):
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.trans_text.configure(
                text="⚠️ Could not access camera. Please check permissions."
            )
            return
        self.camera_running = True
        self.start_btn.configure(text="📹  Camera Running...", bg="#2da3db")
        # Hide placeholder
        self.camera_canvas.itemconfigure(self._placeholder_id, state="hidden")
        self._update_frame()

    def _stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._camera_after_id:
            self.root.after_cancel(self._camera_after_id)
            self._camera_after_id = None
        self.start_btn.configure(text="📹  Start Camera", bg=C["primary"])

        # Restore blank placeholder
        self.camera_canvas.delete("feed")
        self.camera_canvas.itemconfigure(self._placeholder_id, state="normal")
        self.camera_canvas.tag_raise(self.badge_window)
        self.camera_canvas.tag_raise(self.pill_window)

    def _update_frame(self):
        if not self.camera_running or not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cw = self.camera_canvas.winfo_width()
            ch = self.camera_canvas.winfo_height()
            if cw > 1 and ch > 1:
                frame = cv2.resize(frame, (cw, ch))
            img = Image.fromarray(frame)
            self._feed_photo = ImageTk.PhotoImage(img)
            self.camera_canvas.delete("placeholder")
            self.camera_canvas.delete("feed")
            self.camera_canvas.create_image(0, 0, anchor="nw",
                                            image=self._feed_photo, tags="feed")
            self.camera_canvas.tag_raise(self.badge_window)
            self.camera_canvas.tag_raise(self.pill_window)
        self._camera_after_id = self.root.after(33, self._update_frame)

    # ──────────────────────────── SPEECH TO TEXT ──────────────────────
    def _build_speech_panel(self, parent):
        """Build the speech-to-text recording panel (hidden by default)."""
        self.speech_panel = tk.Frame(parent, bg="#1c2732")
        # Don't pack yet — it's shown only when the speech mode is activated

        # Inner container for centering
        inner = tk.Frame(self.speech_panel, bg="#1c2732")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        # Title
        tk.Label(inner, text="🎤  Speech to Text", bg="#1c2732",
                 fg=C["text_dark"], font=(self.FONT, 20, "bold")).pack(pady=(0, 4))
        tk.Label(inner, text="Tap the microphone to start recording",
                 bg="#1c2732", fg=C["text_gray"],
                 font=(self.FONT, 12)).pack(pady=(0, 30))

        # Waveform canvas (animated bars)
        self.wave_canvas = tk.Canvas(inner, bg="#1c2732", width=320, height=60,
                                     highlightthickness=0, bd=0)
        self.wave_canvas.pack(pady=(0, 20))
        self._init_wave_bars()

        # Pulsing ring canvas + microphone button
        mic_container = tk.Frame(inner, bg="#1c2732")
        mic_container.pack(pady=(0, 20))

        self.pulse_canvas = tk.Canvas(mic_container, bg="#1c2732", width=140, height=140,
                                      highlightthickness=0, bd=0)
        self.pulse_canvas.pack()

        # Outer pulse rings (animated)
        self._pulse_ring_outer = self.pulse_canvas.create_oval(
            10, 10, 130, 130, outline=C["primary"], width=2, state="hidden"
        )
        self._pulse_ring_mid = self.pulse_canvas.create_oval(
            20, 20, 120, 120, outline=C["primary"], width=1, state="hidden"
        )

        # Main mic circle button
        self._mic_circle = self.pulse_canvas.create_oval(
            30, 30, 110, 110, fill=C["primary"], outline=""
        )
        self._mic_text = self.pulse_canvas.create_text(
            70, 70, text="🎙️", font=(self.FONT, 28)
        )
        self.pulse_canvas.tag_bind(self._mic_circle, "<Button-1>",
                                    lambda e: self._toggle_stt_recording())
        self.pulse_canvas.tag_bind(self._mic_text, "<Button-1>",
                                    lambda e: self._toggle_stt_recording())
        self.pulse_canvas.configure(cursor="hand2")

        # Status label
        self.stt_status = tk.Label(inner, text="Ready", bg="#1c2732",
                                    fg=C["text_gray"], font=(self.FONT, 13))
        self.stt_status.pack(pady=(0, 10))

        # Recording time label
        self.stt_time_label = tk.Label(inner, text="", bg="#1c2732",
                                        fg=C["primary"], font=(self.FONT, 11, "bold"))
        self.stt_time_label.pack()

    def _init_wave_bars(self):
        """Create static waveform bars on the wave canvas."""
        self._wave_bar_ids = []
        num_bars = 32
        bar_width = 6
        gap = 4
        total_width = num_bars * (bar_width + gap) - gap
        start_x = (320 - total_width) / 2
        for i in range(num_bars):
            x = start_x + i * (bar_width + gap)
            # Start with small bars
            h = 4
            y_top = 30 - h / 2
            y_bot = 30 + h / 2
            bar_id = self.wave_canvas.create_rectangle(
                x, y_top, x + bar_width, y_bot,
                fill=C["primary"], outline=""
            )
            self._wave_bar_ids.append(bar_id)

    def _animate_wave(self):
        """Animate waveform bars while recording."""
        if not self.stt_recording:
            # Reset bars to idle
            for bar_id in self._wave_bar_ids:
                self.wave_canvas.coords(
                    bar_id,
                    self.wave_canvas.coords(bar_id)[0], 28,
                    self.wave_canvas.coords(bar_id)[0] + 6, 32
                )
            return

        self._pulse_phase += 0.15
        num_bars = len(self._wave_bar_ids)
        bar_width = 6
        gap = 4
        total_width = num_bars * (bar_width + gap) - gap
        start_x = (320 - total_width) / 2

        for i, bar_id in enumerate(self._wave_bar_ids):
            # Create organic-looking wave
            h = abs(math.sin(self._pulse_phase + i * 0.3)) * 24 + random.uniform(2, 8)
            x = start_x + i * (bar_width + gap)
            y_top = 30 - h / 2
            y_bot = 30 + h / 2
            self.wave_canvas.coords(bar_id, x, y_top, x + bar_width, y_bot)

            # Color gradient based on height
            intensity = min(255, int(100 + h * 5))
            color = f"#{56:02x}{max(100, intensity):02x}{248:02x}"
            self.wave_canvas.itemconfigure(bar_id, fill=color)

        self._stt_wave_id = self.root.after(80, self._animate_wave)

    def _animate_pulse(self):
        """Animate the pulsing rings around the mic button."""
        if not self.stt_recording:
            self.pulse_canvas.itemconfigure(self._pulse_ring_outer, state="hidden")
            self.pulse_canvas.itemconfigure(self._pulse_ring_mid, state="hidden")
            return

        self._pulse_phase += 0.08
        # Pulsing effect via opacity simulation (expand/contract rings)
        scale = 1.0 + 0.15 * math.sin(self._pulse_phase * 2)
        cx, cy = 70, 70
        r_outer = 60 * scale
        r_mid = 50 * scale

        self.pulse_canvas.coords(self._pulse_ring_outer,
                                  cx - r_outer, cy - r_outer,
                                  cx + r_outer, cy + r_outer)
        self.pulse_canvas.coords(self._pulse_ring_mid,
                                  cx - r_mid, cy - r_mid,
                                  cx + r_mid, cy + r_mid)

        # Vary the ring width for breathing effect
        width_outer = max(1, int(3 * abs(math.sin(self._pulse_phase * 1.5))))
        self.pulse_canvas.itemconfigure(self._pulse_ring_outer, width=width_outer)

        self.pulse_canvas.itemconfigure(self._pulse_ring_outer, state="normal")
        self.pulse_canvas.itemconfigure(self._pulse_ring_mid, state="normal")

        self._stt_pulse_id = self.root.after(50, self._animate_pulse)

    def _toggle_stt_recording(self):
        """Toggle speech-to-text recording on/off."""
        if self.stt_recording:
            self._stop_stt_recording()
        else:
            self._start_stt_recording()

    def _start_stt_recording(self):
        """Start capturing audio and converting to text."""
        if not SR_AVAILABLE:
            self.trans_text.configure(
                text="⚠️ speech_recognition library not installed.\n"
                     "Run: pip install SpeechRecognition PyAudio"
            )
            return

        self.stt_recording = True
        self._pulse_phase = 0

        # Update visuals
        self.pulse_canvas.itemconfigure(self._mic_circle, fill=C["red"])
        self.stt_status.configure(text="🔴  Listening...", fg=C["red"])
        self.trans_text.configure(text="Listening... Speak now.")

        # Start animations
        self._animate_wave()
        self._animate_pulse()

        # Start audio capture in background thread
        self._stt_thread = threading.Thread(target=self._stt_listen, daemon=True)
        self._stt_thread.start()

    def _stop_stt_recording(self):
        """Stop recording."""
        self.stt_recording = False

        # Cancel animations
        if self._stt_wave_id:
            self.root.after_cancel(self._stt_wave_id)
            self._stt_wave_id = None
        if self._stt_pulse_id:
            self.root.after_cancel(self._stt_pulse_id)
            self._stt_pulse_id = None

        # Reset visuals
        self.pulse_canvas.itemconfigure(self._mic_circle, fill=C["primary"])
        self.pulse_canvas.itemconfigure(self._pulse_ring_outer, state="hidden")
        self.pulse_canvas.itemconfigure(self._pulse_ring_mid, state="hidden")
        self.stt_status.configure(text="Ready", fg=C["text_gray"])
        self.stt_time_label.configure(text="")

        # Reset wave bars
        num_bars = len(self._wave_bar_ids)
        bar_width = 6
        gap = 4
        total_width = num_bars * (bar_width + gap) - gap
        start_x = (320 - total_width) / 2
        for i, bar_id in enumerate(self._wave_bar_ids):
            x = start_x + i * (bar_width + gap)
            self.wave_canvas.coords(bar_id, x, 28, x + bar_width, 32)
            self.wave_canvas.itemconfigure(bar_id, fill=C["primary"])

    def _stt_listen(self):
        """Background thread: capture audio and recognize speech."""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                while self.stt_recording:
                    try:
                        self.root.after(0, lambda: self.stt_status.configure(
                            text="🔴  Listening...", fg=C["red"]
                        ))
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        self.root.after(0, lambda: self.stt_status.configure(
                            text="⏳  Processing...", fg="#facc15"
                        ))
                        text = self.recognizer.recognize_google(audio)
                        # Append to existing text
                        self.root.after(0, lambda t=text: self._append_stt_text(t))
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        self.root.after(0, lambda: self.stt_status.configure(
                            text="🔴  Could not understand, try again...", fg="#fb923c"
                        ))
                    except sr.RequestError as e:
                        self.root.after(0, lambda: self.trans_text.configure(
                            text=f"⚠️ Speech recognition service error: {e}"
                        ))
                        break
        except OSError:
            self.root.after(0, lambda: self.trans_text.configure(
                text="⚠️ No microphone found. Please connect a microphone."
            ))
        finally:
            self.root.after(0, self._stop_stt_recording)

    def _append_stt_text(self, new_text):
        """Append recognized text to the translation box."""
        current = self.trans_text.cget("text")
        if current in ("", "Listening... Speak now.",
                        "Press the microphone button to start recording your speech."):
            self.trans_text.configure(text=new_text)
        else:
            self.trans_text.configure(text=current + " " + new_text)
        self.stt_status.configure(text="🔴  Listening...", fg=C["red"])

    # ──────────────────────────── HISTORY FULL PANEL ──────────────────
    def _build_history_full_panel(self, parent):
        """Full-page scrollable history list, shown when History is clicked."""
        self.history_full_panel = tk.Frame(parent, bg=C["bg_body"])

        # Header row
        header = tk.Frame(self.history_full_panel, bg=C["bg_body"])
        header.pack(fill="x", pady=(0, 20))

        tk.Label(header, text="🕐  Translation History", bg=C["bg_body"],
                 fg=C["text_dark"], font=(self.FONT, 22, "bold")).pack(side="left")

        clear_btn = tk.Label(header, text="🗑  Clear All", bg="#7f1d1d", fg="#fca5a5",
                             font=(self.FONT, 11, "bold"), padx=14, pady=6, cursor="hand2")
        clear_btn.pack(side="right")
        clear_btn.bind("<Enter>", lambda e: clear_btn.configure(bg="#991b1b"))
        clear_btn.bind("<Leave>", lambda e: clear_btn.configure(bg="#7f1d1d"))

        # Search bar
        search_row = tk.Frame(self.history_full_panel, bg=C["bg_card"],
                              highlightbackground=C["border"], highlightthickness=1, bd=0)
        search_row.pack(fill="x", pady=(0, 16))

        tk.Label(search_row, text="🔍", bg=C["bg_card"], fg=C["text_gray"],
                 font=(self.FONT, 13)).pack(side="left", padx=(12, 4))
        hist_search = tk.Entry(search_row, bg=C["bg_card"], fg=C["search_text"],
                               insertbackground=C["text_dark"], font=(self.FONT, 13),
                               relief="flat", bd=0)
        hist_search.pack(side="left", fill="x", expand=True, padx=(0, 12), pady=10, ipady=2)
        hist_search.insert(0, "Search history...")

        # Scrollable list
        list_canvas = tk.Canvas(self.history_full_panel, bg=C["bg_body"],
                                highlightthickness=0, bd=0)
        list_scroll = tk.Scrollbar(self.history_full_panel, orient="vertical",
                                   command=list_canvas.yview)
        list_inner = tk.Frame(list_canvas, bg=C["bg_body"])

        list_inner.bind("<Configure>",
                        lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all")))
        list_canvas.create_window((0, 0), window=list_inner, anchor="nw", tags="hist_full")
        list_canvas.configure(yscrollcommand=list_scroll.set)
        list_canvas.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        list_canvas.bind("<Configure>",
                         lambda e: list_canvas.itemconfigure("hist_full", width=e.width - 20))

        # History entries grouped by date
        full_history = [
            {"time": "10:42 AM", "date": "TODAY",
             "text": "Where is the nearest subway station please?", "mode": "sign"},
            {"time": "10:30 AM", "date": "TODAY",
             "text": "Can you help me find the nearest hospital?", "mode": "sign"},
            {"time": "09:15 AM", "date": "TODAY",
             "text": "I would like to order a black coffee with no sugar.", "mode": "braille"},
            {"time": "08:45 AM", "date": "TODAY",
             "text": "Good morning, how are you doing today?", "mode": "speech"},
            {"time": "04:30 PM", "date": "YESTERDAY",
             "text": "Meeting scheduled for tomorrow at the main office.", "mode": "sign"},
            {"time": "02:15 PM", "date": "YESTERDAY",
             "text": "Thank you very much for your help today.", "mode": "braille"},
            {"time": "11:00 AM", "date": "YESTERDAY",
             "text": "Please turn left at the next intersection.", "mode": "sign"},
            {"time": "09:30 AM", "date": "YESTERDAY",
             "text": "The weather forecast says it will rain tomorrow.", "mode": "speech"},
            {"time": "03:45 PM", "date": "APR 13",
             "text": "I need to book a flight to New York for next week.", "mode": "sign"},
            {"time": "01:20 PM", "date": "APR 13",
             "text": "The presentation has been rescheduled to Friday.", "mode": "speech"},
            {"time": "10:00 AM", "date": "APR 12",
             "text": "Welcome to the digital accessibility workshop.", "mode": "braille"},
            {"time": "09:00 AM", "date": "APR 12",
             "text": "Please remember to bring your ID card tomorrow.", "mode": "sign"},
        ]

        current_date = ""
        for item in full_history:
            if item["date"] != current_date:
                current_date = item["date"]
                date_frame = tk.Frame(list_inner, bg=C["bg_body"])
                date_frame.pack(fill="x", pady=(16, 8))
                tk.Label(date_frame, text=current_date, bg=C["bg_body"],
                         fg=C["primary"], font=(self.FONT, 11, "bold")).pack(side="left")
                sep = tk.Frame(date_frame, bg=C["border"], height=1)
                sep.pack(side="left", fill="x", expand=True, padx=(12, 0), pady=1)
            self._make_history_full_card(list_inner, item)

    def _make_history_full_card(self, parent, data):
        """Single card in the full history list."""
        card = tk.Frame(parent, bg=C["bg_card"], highlightbackground=C["border"],
                        highlightthickness=1, bd=0)
        card.pack(fill="x", pady=(0, 6))

        inner = tk.Frame(card, bg=C["bg_card"])
        inner.pack(fill="x", padx=20, pady=14)

        # Top row — time, mode badge, copy
        top = tk.Frame(inner, bg=C["bg_card"])
        top.pack(fill="x")

        tk.Label(top, text=data["time"], bg=C["bg_card"], fg=C["muted"],
                 font=(self.FONT, 10, "bold")).pack(side="left")

        mode_colors = {"sign": "#22c55e", "braille": "#a78bfa", "speech": "#f97316"}
        mode_labels = {"sign": "🎥 Sign", "braille": "⠿ Braille", "speech": "🎤 Speech"}
        mode = data.get("mode", "sign")
        tk.Label(top, text=mode_labels.get(mode, "Sign"),
                 bg=mode_colors.get(mode, C["primary"]), fg="white",
                 font=(self.FONT, 9, "bold"), padx=8, pady=2).pack(side="left", padx=(12, 0))

        copy_l = tk.Label(top, text="📋", bg=C["bg_card"], fg=C["muted"],
                          font=(self.FONT, 12), cursor="hand2")
        copy_l.pack(side="right")
        copy_l.bind("<Button-1>", lambda e, t=data["text"]: self._copy_specific(t))

        # Text
        txt = tk.Label(inner, text=data["text"], bg=C["bg_card"], fg=C["text_dark"],
                       font=(self.FONT, 14), wraplength=700, justify="left", anchor="w")
        txt.pack(fill="x", pady=(10, 0))

        # Hover effect
        def hover_in(e):
            card.configure(highlightbackground=C["primary"])
        def hover_out(e):
            card.configure(highlightbackground=C["border"])
        for w in [card, inner, top, txt]:
            w.bind("<Enter>", hover_in)
            w.bind("<Leave>", hover_out)
            w.configure(cursor="hand2")

    def _copy_specific(self, text):
        """Copy a specific string to clipboard with flash feedback."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    # ──────────────────────────── SETTINGS PANEL ─────────────────────
    def _build_settings_panel(self, parent):
        """Full-page settings panel with toggles, dropdowns, info rows."""
        self.settings_panel = tk.Frame(parent, bg=C["bg_body"])

        # Header
        header = tk.Frame(self.settings_panel, bg=C["bg_body"])
        header.pack(fill="x", pady=(0, 10))
        tk.Label(header, text="⚙️  Settings", bg=C["bg_body"],
                 fg=C["text_dark"], font=(self.FONT, 22, "bold")).pack(side="left")

        # Scrollable content
        s_canvas = tk.Canvas(self.settings_panel, bg=C["bg_body"],
                             highlightthickness=0, bd=0)
        s_scroll = tk.Scrollbar(self.settings_panel, orient="vertical",
                                command=s_canvas.yview)
        s_inner = tk.Frame(s_canvas, bg=C["bg_body"])

        s_inner.bind("<Configure>",
                     lambda e: s_canvas.configure(scrollregion=s_canvas.bbox("all")))
        s_canvas.create_window((0, 0), window=s_inner, anchor="nw", tags="set_inner")
        s_canvas.configure(yscrollcommand=s_scroll.set)
        s_canvas.pack(side="left", fill="both", expand=True)
        s_scroll.pack(side="right", fill="y")
        s_canvas.bind("<Configure>",
                      lambda e: s_canvas.itemconfigure("set_inner", width=e.width - 20))

        # ── Language ──
        self._settings_section(s_inner, "🌐  Language")
        lc = self._settings_card(s_inner)
        self._dropdown_row(lc, "Input Language",
                           ["English", "Spanish", "French", "Hindi", "Arabic", "Chinese"],
                           "English")
        self._dropdown_row(lc, "Output Language",
                           ["English", "Spanish", "French", "Hindi", "Arabic", "Chinese"],
                           "English")

        # ── Detection ──
        self._settings_section(s_inner, "🎯  Detection")
        dc = self._settings_card(s_inner)
        self._toggle_row(dc, "Auto-detect language", True)
        self._toggle_row(dc, "High accuracy mode", False)
        self._toggle_row(dc, "Show confidence scores", False)

        # ── Camera ──
        self._settings_section(s_inner, "📹  Camera")
        cc = self._settings_card(s_inner)
        self._toggle_row(cc, "Mirror video", True)
        self._dropdown_row(cc, "Video quality", ["480p", "720p", "1080p"], "720p")

        # ── Speech Recognition ──
        self._settings_section(s_inner, "🎤  Speech Recognition")
        sc = self._settings_card(s_inner)
        self._toggle_row(sc, "Auto-punctuation", True)
        self._toggle_row(sc, "Continuous listening", True)
        self._dropdown_row(sc, "Speech engine",
                           ["Google (Online)", "Sphinx (Offline)"], "Google (Online)")

        # ── Data & Privacy ──
        self._settings_section(s_inner, "💾  Data & Privacy")
        pc = self._settings_card(s_inner)
        self._toggle_row(pc, "Auto-save translations", True)
        self._toggle_row(pc, "Send anonymous analytics", False)

        # Clear history button
        clr = tk.Label(pc, text="🗑  Clear All History", bg="#7f1d1d", fg="#fca5a5",
                       font=(self.FONT, 12, "bold"), pady=10, cursor="hand2")
        clr.pack(fill="x", pady=(12, 0))
        clr.bind("<Enter>", lambda e: clr.configure(bg="#991b1b"))
        clr.bind("<Leave>", lambda e: clr.configure(bg="#7f1d1d"))

        # ── About ──
        self._settings_section(s_inner, "ℹ️  About")
        ac = self._settings_card(s_inner)
        self._info_row(ac, "Version", "2.0.0")
        self._info_row(ac, "Build", "2026.04.15")
        self._info_row(ac, "Engine", "Unispeak Neural v3")

        # Bottom spacer
        tk.Frame(s_inner, bg=C["bg_body"], height=30).pack()

    # ── Settings helpers ──────────────────────────────────────────────
    def _settings_section(self, parent, title):
        """Section header inside settings."""
        tk.Label(parent, text=title, bg=C["bg_body"], fg=C["text_gray"],
                 font=(self.FONT, 12, "bold")).pack(anchor="w", pady=(20, 8))

    def _settings_card(self, parent):
        """Card container for a group of settings rows."""
        card = tk.Frame(parent, bg=C["bg_card"], highlightbackground=C["border"],
                        highlightthickness=1, bd=0)
        card.pack(fill="x", pady=(0, 4))
        inner = tk.Frame(card, bg=C["bg_card"])
        inner.pack(fill="x", padx=20, pady=14)
        return inner

    def _toggle_row(self, parent, text, default=False):
        """A settings row with a label and a custom toggle switch."""
        row = tk.Frame(parent, bg=C["bg_card"])
        row.pack(fill="x", pady=6)

        tk.Label(row, text=text, bg=C["bg_card"], fg=C["text_dark"],
                 font=(self.FONT, 13)).pack(side="left")

        var = tk.BooleanVar(value=default)
        tgl = tk.Canvas(row, width=46, height=26, bg=C["bg_card"],
                        highlightthickness=0, bd=0, cursor="hand2")
        tgl.pack(side="right")

        def draw():
            tgl.delete("all")
            if var.get():
                tgl.create_oval(0, 0, 26, 26, fill=C["primary"], outline="")
                tgl.create_rectangle(13, 0, 33, 26, fill=C["primary"], outline="")
                tgl.create_oval(20, 0, 46, 26, fill=C["primary"], outline="")
                tgl.create_oval(23, 3, 43, 23, fill="white", outline="")
            else:
                tgl.create_oval(0, 0, 26, 26, fill="#374151", outline="")
                tgl.create_rectangle(13, 0, 33, 26, fill="#374151", outline="")
                tgl.create_oval(20, 0, 46, 26, fill="#374151", outline="")
                tgl.create_oval(3, 3, 23, 23, fill="white", outline="")

        def on_click(e):
            var.set(not var.get())
            draw()

        tgl.bind("<Button-1>", on_click)
        draw()
        return var

    def _dropdown_row(self, parent, text, options, default):
        """A settings row with a label and a dropdown selector."""
        row = tk.Frame(parent, bg=C["bg_card"])
        row.pack(fill="x", pady=6)

        tk.Label(row, text=text, bg=C["bg_card"], fg=C["text_dark"],
                 font=(self.FONT, 13)).pack(side="left")

        var = tk.StringVar(value=default)
        dd = tk.OptionMenu(row, var, *options)
        dd.configure(bg=C["border"], fg=C["text_dark"], font=(self.FONT, 11),
                     highlightthickness=0, bd=0, activebackground=C["primary"],
                     activeforeground="white", relief="flat")
        dd["menu"].configure(bg=C["bg_card"], fg=C["text_dark"], font=(self.FONT, 11),
                             activebackground=C["primary"], activeforeground="white", bd=0)
        dd.pack(side="right")
        return var

    def _info_row(self, parent, label, value):
        """A read-only info row (label + value)."""
        row = tk.Frame(parent, bg=C["bg_card"])
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, bg=C["bg_card"], fg=C["text_gray"],
                 font=(self.FONT, 13)).pack(side="left")
        tk.Label(row, text=value, bg=C["bg_card"], fg=C["text_dark"],
                 font=(self.FONT, 13, "bold")).pack(side="right")

    # ──────────────────────────── CLEANUP ────────────────────────────
    def _on_close(self):
        self._stop_camera()
        if self.stt_recording:
            self._stop_stt_recording()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = UnispeakApp()
    app.run()
