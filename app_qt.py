import sys
import subprocess
import requests
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import QUrl, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineProfile

STREAMLIT_URL = 'http://localhost:8501?qt=1'


# ----------------------
# Start Streamlit Process
# ----------------------
def start_streamlit():
    return subprocess.Popen(
        [
            'streamlit',
            'run',
            'app.py',
            '--server.headless',
            'true',
            '--client.toolbarMode',
            'viewer',
            '--server.port',
            '8501',
            '--server.enableCORS',
            'false',
            '--server.enableXsrfProtection',
            'false',
            '--browser.gatherUsageStats',
            'false',
            '--global.developmentMode',
            'false',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ----------------------
# PySide6 Main Window
# ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OnePower Explorer')
        self.resize(1300, 1000)

        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)

        profile = QWebEngineProfile.defaultProfile()
        profile.downloadRequested.connect(self.on_download_requested)

        # Timer to check server readiness
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_server)
        self.timer.start(1000)

    def on_download_requested(self, download):
        # Suggested filename from Streamlit
        suggested_name = download.downloadFileName()

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save file', suggested_name, 'CSV Files (*.csv);;All Files (*)'
        )

        if path:
            download.setDownloadDirectory(str(Path(path).parent))
            download.setDownloadFileName(Path(path).name)
            download.accept()
        else:
            download.cancel()

    def check_server(self):
        try:
            requests.get(STREAMLIT_URL)
            self.timer.stop()

            # Send data to Streamlit via query params
            # url_with_params = STREAMLIT_URL + "?value=HelloFromPySide"
            self.browser.setUrl(QUrl(STREAMLIT_URL))
        except requests.exceptions.ConnectionError:
            pass


# ----------------------
# Main Entry Point
# ----------------------
if __name__ == '__main__':
    streamlit_process = start_streamlit()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    exit_code = app.exec()

    streamlit_process.terminate()
    sys.exit(exit_code)
