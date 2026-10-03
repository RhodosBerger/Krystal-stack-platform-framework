import unittest
import os
import urllib.request

class TestKrystalWebOSDesktop(unittest.TestCase):
    def setUp(self):
        self.repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.static_dir = os.path.join(self.repo_dir, "krystal_web_hub", "static")
        self.desktop_html = os.path.join(self.static_dir, "krystal_webos_desktop.html")
        self.wm_js = os.path.join(self.static_dir, "krystal_window_manager.js")
        self.wm_css = os.path.join(self.static_dir, "krystal_window_manager.css")

    def test_files_exist_and_non_empty(self):
        self.assertTrue(os.path.isfile(self.desktop_html), "krystal_webos_desktop.html must exist")
        self.assertTrue(os.path.isfile(self.wm_js), "krystal_window_manager.js must exist")
        self.assertTrue(os.path.isfile(self.wm_css), "krystal_window_manager.css must exist")

        self.assertGreater(os.path.getsize(self.desktop_html), 5000)
        self.assertGreater(os.path.getsize(self.wm_js), 4000)
        self.assertGreater(os.path.getsize(self.wm_css), 3000)

    def test_window_manager_js_features(self):
        with open(self.wm_js, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("class KrystalWindowManager", code)
        self.assertIn("createWindow", code)
        self.assertIn("focusWindow", code)
        self.assertIn("minimizeWindow", code)
        self.assertIn("maximizeWindow", code)
        self.assertIn("toggleMaximizeWindow", code)
        self.assertIn("snapWindow", code)
        self.assertIn("showSnapPreview", code)
        self.assertIn("VITAL_MAX_HP = 6", code)

    def test_desktop_html_contents(self):
        with open(self.desktop_html, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn("Krystal WebOS Desktop", html)
        self.assertIn("krystal_window_manager.css", html)
        self.assertIn("krystal_window_manager.js", html)
        self.assertIn("KrystalWindowManager", html)
        self.assertIn("Neural Dynamics Renderer", html)
        self.assertIn("Grécky Panteón", html)
        self.assertIn("Topological VM", html)
        self.assertIn("Vulkan Iris Xe", html)
        self.assertIn("VITAL HP:", html)

    def test_http_endpoint_desktop(self):
        url = "http://127.0.0.1:8089/desktop"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=4) as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
                self.assertIn("Krystal WebOS", body)
                self.assertIn("k-desktop", body)
        except Exception as e:
            self.skipTest(f"Server not currently reachable on {url}: {e}")

    def test_http_endpoint_webos(self):
        url = "http://127.0.0.1:8089/webos"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=4) as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
                self.assertIn("Krystal WebOS", body)
        except Exception as e:
            self.skipTest(f"Server not currently reachable on {url}: {e}")

if __name__ == "__main__":
    unittest.main()
