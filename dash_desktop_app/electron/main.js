const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let pythonProcess = null;

function createWindow() {
  const pythonPath = path.join(__dirname, '..', 'run.py');
  pythonProcess = spawn('python3', [pythonPath]);

  pythonProcess.stdout.on('data', (data) => {
    const msg = data.toString();
    console.log(`[PYTHON STDOUT]: ${msg}`);

    if (msg.includes('Dash is running on')) {
      // Create the Electron window after Dash is ready
      mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
          contextIsolation: true,
        },
      });

      mainWindow.loadURL('http://127.0.0.1:8050');

      mainWindow.on('closed', () => {
        if (pythonProcess) {
          pythonProcess.kill();
        }
        mainWindow = null;
      });
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    const msg = data.toString();

    // Avoid printing harmless GET/POST request logs as errors
    if (msg.includes('"GET') || msg.includes('"POST')) {
      console.log(`[PYTHON ACCESS]: ${msg}`);
    } else {
      console.error(`[PYTHON STDERR]: ${msg}`);
    }
  });

  pythonProcess.on('close', (code) => {
    console.log(`[PYTHON]: exited with code ${code}`);
  });
}

app.whenReady().then(createWindow);

// macOS-friendly behavior
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (pythonProcess) {
      pythonProcess.kill();
    }
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
