from cefpython3 import cefpython as cef
import tkinter as tk
import platform
import sys
import os

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x700")
        self.root.title("Metro Harita Görüntüleyici")

        # Frame: Üstte buton, altta tarayıcı
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn = tk.Button(top_frame, text="Optimizasyon Ayarları", command=self.open_settings)
        btn.pack(pady=5)

        self.browser_frame = BrowserFrame(root)
        self.browser_frame.pack(fill=tk.BOTH, expand=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Optimizasyon Parametreleri")
        settings_win.geometry("300x250")

        labels = ["Max Mesafe", "Yeni İstasyon Sayısı", "Popülasyon Katsayısı", "Bağlantı Katsayısı"]
        self.entries = []

        for i, label_text in enumerate(labels):
            tk.Label(settings_win, text=label_text).pack()
            entry = tk.Entry(settings_win)
            entry.pack()
            self.entries.append(entry)

        tk.Button(settings_win, text="Başlat", command=self.run_optimization).pack(pady=10)

    def run_optimization(self):
        values = [entry.get() for entry in self.entries]
        print("Girilen Parametreler:", values)
        # Buraya optimizasyonu başlatacak fonksiyonu çağırabilirsin

    def on_close(self):
        self.browser_frame.shutdown()
        self.root.destroy()
        sys.exit()

class BrowserFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.browser = None
        self.embed_browser()

    def embed_browser(self):
        window_info = cef.WindowInfo()
        rect = [0, 0, self.winfo_width(), self.winfo_height()]
        window_info.SetAsChild(self.winfo_id(), rect)
        map_path = os.path.abspath("maps/initial_metro_with_lines.html").replace("\\", "/")

        self.browser = cef.CreateBrowserSync(window_info,
                                             url=f"file:///{map_path}")
        self.bind("<Configure>", self.on_configure)

    def on_configure(self, event):
        if self.browser:
            self.browser.SetBounds(0, 0, event.width, event.height)

    def shutdown(self):
        if self.browser:
            cef.Shutdown()

def main():
    sys.excepthook = cef.ExceptHook  # CEF hata yakalayıcı
    settings = {}
    cef.Initialize(settings)
    root = tk.Tk()
    app = MainApp(root)
    cef.MessageLoop()
    root.mainloop()

if __name__ == "__main__":
    main()
