"""
Welcome page view
"""
from django.shortcuts import render, redirect
from django.http import HttpResponse

def home_view(request):
    """Homepage with links to all sections"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fanuc Rise - CNC Cognitive Platform</title>
        <style>
            body {
                font-family: 'Inter', system-ui, sans-serif;
                background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
                color: #e4e4e7;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                text-align: center;
                max-width: 800px;
                padding: 40px;
            }
            h1 {
                font-size: 3rem;
                background: linear-gradient(45deg, #38bdf8, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 20px;
            }
            p {
                font-size: 1.2rem;
                color: #a1a1aa;
                margin-bottom: 40px;
            }
            .links {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 40px;
            }
            .link-card {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 30px 20px;
                text-decoration: none;
                color: #e4e4e7;
                transition: all 0.3s;
            }
            .link-card:hover {
                background: rgba(56, 189, 248, 0.1);
                border-color: #38bdf8;
                transform: translateY(-5px);
            }
            .link-card h3 {
                margin: 0 0 10px 0;
                font-size: 1.3rem;
            }
            .link-card p {
                margin: 0;
                font-size: 0.9rem;
                color: #71717a;
            }
            .status {
                display: inline-block;
                background: #10b981;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔧 FANUC RISE</h1>
            <div class="status">✅ Server Running</div>
            <p>Cognitive CNC Platform - AI-Driven Manufacturing Intelligence</p>
            
            <div class="links">
                <a href="/api/" class="link-card">
                    <h3>📊 REST API</h3>
                    <p>Browse API endpoints</p>
                </a>
                
                <a href="/admin/" class="link-card">
                    <h3>🔧 Admin Panel</h3>
                    <p>Manage system data</p>
                </a>
                
                <a href="/api/machines/" class="link-card">
                    <h3>🏭 Machines</h3>
                    <p>CNC machine registry</p>
                </a>
                
                <a href="/api/analytics/dashboard/" class="link-card">
                    <h3>📈 Analytics</h3>
                    <p>Dashboard metrics</p>
                </a>
            </div>
            
            <div style="margin-top: 60px; color: #71717a; font-size: 0.9rem;">
                <p>Django version 6.0.1 | Database: SQLite | Phase 45 Complete</p>
                <p>Flask microservice: <a href="http://localhost:5000/health" style="color: #38bdf8;">http://localhost:5000/health</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return HttpResponse(html)
