# MANUAL: FOCAS Library Integration

To enable real-time hardware communication with Fanuc CNCs, you must install the proprietary FOCAS library.

## Step 1: Obtain the DLL
1. Contact Fanuc or your machine distributor.
2. Request the **FOCAS2 Library (Ethernet version)**.
3. You are looking for:
   * **Windows**: `Fwlib32.dll` (and associated files: `Fwlibe1.dll`, `Fwlib32.lib`)
   * **Linux**: `libfwlib32.so`

## Step 2: Install
1. Create a folder named `libs` in the root of the project:
   ```bash
   mkdir libs
   ```
2. Copy `Fwlib32.dll` (or .so) into `libs/`.

## Step 3: Verification
1. Restart the backend: `repair_and_launch.bat`.
2. Check the console logs.
   * **Success**: `✅ Loaded FOCAS Library`
   * **Fallback**: `⚠️ FOCAS Library not found. Running in SIMULATION MODE.`

## Note on Licensing
The FOCAS library is Copyright © FANUC CORPORATION. It cannot be distributed with this open-source repo.
