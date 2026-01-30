import os
import sys
import time
from backend.worker import celery_app

def test_knowledge_engine():
    print("🚀 Verifying Knowledge Engine & Presets...")

    # 1. Trigger Blog Crawler
    print("\n[1] Triggering 'check_industry_blogs_task'...")
    crawler_task = celery_app.send_task("tasks.check_industry_blogs")
    print(f"    Task ID: {crawler_task.id}")
    
    # Poll for result
    for _ in range(10):
        if crawler_task.ready():
            result = crawler_task.get()
            print("    ✅ Crawler Result:")
            print(f"       Topic: {result['data']['title']}")
            print(f"       Source: {result['data']['source']}")
            break
        time.sleep(1)
    else:
        print("    ❌ Crawler Task Timeout")

    # 2. Trigger Propedeutics Generator
    topic = "Aluminum6061"
    print(f"\n[2] Triggering 'generate_propedeutics_task' for {topic}...")
    gen_task = celery_app.send_task("tasks.generate_propedeutics", args=[topic])
    print(f"    Task ID: {gen_task.id}")

    # Poll for result
    for _ in range(10):
        if gen_task.ready():
            result = gen_task.get()
            print("    ✅ Generator Result:")
            print(f"       Status: {result['status']}")
            print(f"       Snippet: {result.get('content_snippet', 'No content')}")
            break
        time.sleep(1)
    else:
        print("    ❌ Generator Task Timeout")

if __name__ == "__main__":
    test_knowledge_engine()
