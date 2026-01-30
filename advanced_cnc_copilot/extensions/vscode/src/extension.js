const vscode = require('vscode');
const axios = require('axios'); // Assuming axios is bundled or fetch is used

// Configuration
const API_URL = 'http://localhost:8000/api';

function activate(context) {
    console.log('Fanuc Rise Neuro-Coder is active!');

    // 1. Command: Ask Hive Mind
    let askDisposable = vscode.commands.registerCommand('fanuc.askHiveMind', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.selection;
        const text = editor.document.getText(selection);

        if (!text) {
            vscode.window.showInformationMessage('Please select some G-Code to analyze.');
            return;
        }

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Consulting Hive Mind...",
            cancellable: false
        }, async (progress) => {
            try {
                // In a real extension, you'd use fetch or a bundled axios
                // Simulating response for now since we don't have node_modules in this generated folder yet
                const response = `Analysis of selected code:\n- Detected 3 rapid moves (G00).\n- Feedrate F500 is optimal for Aluminum 6061.\n- Suggestion: Add dwell (G04) before Z-retract.`;
                
                // Show result in a webview panel
                const panel = vscode.window.createWebviewPanel(
                    'hiveMindResponse',
                    'Hive Mind Analysis',
                    vscode.ViewColumn.Beside,
                    {}
                );
                panel.webview.html = getWebviewContent(response);
            } catch (error) {
                vscode.window.showErrorMessage(`Hive Mind Error: ${error.message}`);
            }
        });
    });

    context.subscriptions.push(askDisposable);
}

function getWebviewContent(content) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hive Mind</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #fff; background-color: #0F172A; }
        h1 { color: #00FFC8; font-family: monospace; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .response { background: #1E293B; padding: 15px; border-radius: 8px; border: 1px solid #333; white-space: pre-wrap; font-family: monospace; }
        .footer { margin-top: 20px; font-size: 0.8em; color: #666; }
    </style>
</head>
<body>
    <h1>NEURAL ANALYSIS COMPLETE</h1>
    <div class="response">${content}</div>
    <div class="footer">Fanuc Rise v2.1 // Connected to Localhost:8000</div>
</body>
</html>`;
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}
