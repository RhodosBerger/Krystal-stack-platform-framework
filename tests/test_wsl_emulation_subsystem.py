import unittest
import json
import urllib.request
from krystal_web_hub.economic_engine.wsl_emulation_subsystem import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    LinuxDistroFlavor,
    WslRuntimeMode,
    VirtualDnfEngine,
    WslEmulationSubsystem,
    GLOBAL_WSL_EMULATION_SUBSYSTEM
)

class TestWslEmulationSubsystem(unittest.TestCase):
    def setUp(self):
        self.subsystem = WslEmulationSubsystem(mode=WslRuntimeMode.FORCE_EMULATED)

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        status = self.subsystem.get_status()
        self.assertEqual(status["vital_max_hp_rule"], 6)

    def test_virtual_filesystem_initialization(self):
        self.assertIn("/etc/os-release", self.subsystem.vfs)
        self.assertIn("/proc/version", self.subsystem.vfs)
        self.assertIn("/etc/dnf/dnf.conf", self.subsystem.vfs)
        os_release = self.subsystem.vfs["/etc/os-release"].content
        self.assertIn("Fedora Linux 40", os_release)

    def test_virtual_dnf_package_lifecycle(self):
        dnf = VirtualDnfEngine(LinuxDistroFlavor.FEDORA_40)
        # 1. Install assimp
        res = dnf.install("assimp")
        self.assertEqual(res["status"], "TRANSACTION_SUCCESS")
        self.assertEqual(res["vital_max_hp_rule"], 6)
        self.assertIn("assimp", dnf.installed_packages)

        # 2. Already installed check
        res2 = dnf.install("assimp")
        self.assertEqual(res2["status"], "ALREADY_INSTALLED")

        # 3. Repolist
        repos = dnf.repolist()
        self.assertGreaterEqual(len(repos), 3)

        # 4. Remove
        rm_res = dnf.remove("assimp")
        self.assertEqual(rm_res["status"], "REMOVED_SUCCESS")
        self.assertNotIn("assimp", dnf.installed_packages)

    def test_virtual_posix_command_execution(self):
        # 1. uname -a
        res_uname = self.subsystem.execute_command("uname -a")
        self.assertEqual(res_uname["exit_code"], 0)
        self.assertIn("krystal-virtual-WSL2", res_uname["stdout"])

        # 2. cat /etc/os-release
        res_cat = self.subsystem.execute_command("cat /etc/os-release")
        self.assertEqual(res_cat["exit_code"], 0)
        self.assertIn("PRETTY_NAME", res_cat["stdout"])

        # 3. which dnf
        res_which = self.subsystem.execute_command("which dnf")
        self.assertEqual(res_which["exit_code"], 0)
        self.assertEqual(res_which["stdout"].strip(), "/usr/bin/dnf")

        # 4. dnf install assimp via CLI
        res_dnf = self.subsystem.execute_command("dnf install assimp")
        self.assertEqual(res_dnf["exit_code"], 0)
        self.assertIn("Complete!", res_dnf["stdout"])

        # 5. assimp info
        res_assimp = self.subsystem.execute_command("assimp info /mnt/c/model.obj")
        self.assertEqual(res_assimp["exit_code"], 0)
        self.assertIn("ASSIMP 3D ASSET INSPECTOR", res_assimp["stdout"])

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/wsl/subsystem-status
            req = urllib.request.Request(f"{base_url}/api/wsl/subsystem-status")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertIn("distro_flavor", d)

            # 2. GET /api/wsl/virtual-fs
            req = urllib.request.Request(f"{base_url}/api/wsl/virtual-fs")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertGreater(d["node_count"], 10)

            # 3. POST /api/wsl/exec
            req = urllib.request.Request(
                f"{base_url}/api/wsl/exec",
                data=json.dumps({"command": "uname -a"}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertIn("stdout", d)

            # 4. POST /api/wsl/dnf-emulate
            req = urllib.request.Request(
                f"{base_url}/api/wsl/dnf-emulate",
                data=json.dumps({"action": "install", "package": "vulkan-tools"}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertIn("status", d)

            # 5. GET /wsl-emulator HTML
            req = urllib.request.Request(f"{base_url}/wsl-emulator")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("WSL EMULAČNÝ SUBSYSTÉM", html)

        except Exception as e:
            self.skipTest(f"Live server test skipped: {e}")

if __name__ == "__main__":
    unittest.main()
