#!/usr/bin/env python3
"""
LOG TRANSFORMER
Converts raw logs to readable formats for different audiences.
"""

import json
import os
from datetime import datetime
from typing import List, Dict

class LogTransformer:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir

    def technical_to_csv(self, output_file="logs/technical_export.csv"):
        """
        Convert technical.jsonl to CSV for Excel analysis.
        """
        with open(f"{self.log_dir}/technical.jsonl", "r") as f:
            lines = f.readlines()

        with open(output_file, "w") as csv:
            # Header
            csv.write("Timestamp,Level,Message,RPM,Load,Vibration\n")
            
            for line in lines:
                try:
                    entry = json.loads(line)
                    ctx = entry.get("context", {})
                    csv.write(f"{entry['timestamp']},{entry['level']},{entry['message']},"
                             f"{ctx.get('rpm', '')} ,{ctx.get('load', '')},{ctx.get('vibration', '')}\n")
                except:
                    pass
        
        print(f"✅ CSV exported to: {output_file}")

    def operator_to_html(self, output_file="logs/operator_report.html"):
        """
        Convert operator.txt to styled HTML report.
        """
        with open(f"{self.log_dir}/operator.txt", "r") as f:
            lines = f.readlines()

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial; background: #f5f5f5; padding: 20px; }
                .log-entry { background: white; margin:10px 0; padding:15px; border-radius:8px; }
                .warning { border-left: 4px solid orange; }
                .error { border-left: 4px solid red; }
            </style>
        </head>
        <body>
            <h1>🛠️ Operator Alert Log</h1>
        """
        
        for line in lines:
            css_class = "warning" if "⚠️" in line else "error"
            html += f'<div class="log-entry {css_class}">{line.strip()}</div>\n'
            
        html += "</body></html>"
        
        with open(output_file, "w") as f:
            f.write(html)
        
        print(f"✅ HTML report: {output_file}")

    def aggregate_stats(self) -> Dict:
        """
        Generate summary statistics from technical logs.
        """
        stats = {
            "total_events": 0,
            "warnings": 0,
            "errors": 0,
            "avg_load": 0,
            "max_vibration": 0
        }
        
        with open(f"{self.log_dir}/technical.jsonl", "r") as f:
            lines = f.readlines()
            
        loads = []
        vibs = []
        
        for line in lines:
            try:
                entry = json.loads(line)
                stats["total_events"] += 1
                
                if entry["level"] == "WARNING":
                    stats["warnings"] += 1
                elif entry["level"] == "ERROR":
                    stats["errors"] += 1
                
                ctx = entry.get("context", {})
                if "load" in ctx:
                    loads.append(ctx["load"])
                if "vibration" in ctx:
                    vibs.append(ctx["vibration"])
            except:
                pass
        
        if loads:
            stats["avg_load"] = sum(loads) / len(loads)
        if vibs:
            stats["max_vibration"] = max(vibs)
        
        return stats

# Usage
if __name__ == "__main__":
    transformer = LogTransformer()
    transformer.technical_to_csv()
    transformer.operator_to_html()
    
    stats = transformer.aggregate_stats()
    print("\n📊 Log Statistics:")
    print(json.dumps(stats, indent=2))
