import urllib.request

url = "http://127.0.0.1:8089/sovereign-citadel"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as resp:
    status = resp.status
    content = resp.read().decode('utf-8')
    assert "CITADELA SUVERÉNNEHO ŠTÍTU" in content
    assert "wordpress_subdomain_security_shield.jpg" in content
    assert "VITAL_MAX_HP = 6" in content
    print(f"[OK] Page fetched successfully! Status: {status}, Content length: {len(content)}")
