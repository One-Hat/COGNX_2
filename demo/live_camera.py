import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from mnema.model import MNEMA
from instrument.counters import EnergyInstrument

def run_live_demonstrator():
    print("=== Launching MNEMA Real-Time Edge Demonstrator ===")
    
    # Initialize Model & Instrument (4 object training slots: 0, 1, 2, 3)
    model = MNEMA(n_in=784, n_s=16384, k=64, d=4, budget_bytes=32768)
    instrument = EnergyInstrument(tech_card_path="instrument/tech/asic_45nm.yaml")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Check camera connection/permissions.")
        return

    prev_frame_gray = None
    slot_names = {0: "Object 1", 1: "Object 2", 2: "Object 3", 3: "Object 4"}
    trained_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    test_mode = False
    status_msg = "Ready. Press 1-4 to train live object into slot."
    last_pred = "-"
    last_conf = 0.0
    last_alpha = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Crop center square region for input
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
        
        # Convert to 28x28 grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Compute event difference (temporal contrast)
        if prev_frame_gray is None:
            prev_frame_gray = resized.copy()
        
        diff = cv2.absdiff(resized, prev_frame_gray)
        prev_frame_gray = resized.copy()

        # Normalize to [0, 1] intensity vector
        x_input = (resized.astype(np.float32) / 255.0).flatten()

        key = cv2.waitKey(1) & 0xFF
        
        # Keyboard Interactions
        target_slot = None
        if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            target_slot = key - ord('1')
            trained_counts[target_slot] += 1
            status_msg = f"Training {slot_names[target_slot]} (Count: {trained_counts[target_slot]})"
            
            with instrument:
                out = model.step(x_input, y=target_slot, is_training=True, instrument=instrument)
                last_pred = slot_names[out["prediction"]]
                last_conf = out["confidence"]
                last_alpha = out["fast_weight_alpha"]

        elif key == ord('s'):
            with instrument:
                replayed = model.consolidator.consolidate(model.store, model.cortex, instrument=instrument)
                model.controller.reset_sleep_pressure()
            status_msg = f"Sleep Consolidation complete. Replayed {replayed} traces."

        elif key == ord('t'):
            test_mode = not test_mode
            status_msg = f"Inference Mode: {'ON' if test_mode else 'OFF'}"

        elif key == ord('q'):
            break

        # Inference evaluation during test mode or idle
        if test_mode and target_slot is None:
            with instrument:
                out = model.step(x_input, is_training=False, instrument=instrument)
                last_pred = slot_names[out["prediction"]]
                last_conf = out["confidence"]
                last_alpha = out["fast_weight_alpha"]

        # --- Build UI Canvas ---
        canvas = np.zeros((h, w + 420, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        
        # Highlight central receptive field
        cv2.rectangle(canvas, (start_x, start_y), (start_x + min_dim, start_y + min_dim), (0, 255, 0), 2)

        # Inset 28x28 event raster preview (scaled up)
        event_vis = cv2.resize(diff, (160, 160), interpolation=cv2.INTER_NEAREST)
        event_vis_bgr = cv2.cvtColor(event_vis, cv2.COLOR_GRAY2BGR)
        canvas[20:180, 20:180] = event_vis_bgr
        cv2.rectangle(canvas, (20, 20), (180, 180), (0, 255, 255), 1)
        cv2.putText(canvas, "Event Delta [28x28]", (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Telemetry HUD Sidebar
        hud_x = w + 20
        cv2.putText(canvas, "MNEMA EDGE HUD", (hud_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)
        cv2.line(canvas, (hud_x, 50), (hud_x + 380, 50), (80, 80, 80), 1)

        # Predictions & Arbitration
        cv2.putText(canvas, f"Prediction : {last_pred}", (hud_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, f"Confidence : {last_conf:.2f}", (hud_x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        # Arbitration Ratio (Alpha: Fast vs Slow)
        alpha_pct = int(last_alpha * 100)
        cv2.putText(canvas, f"Readout [Fast F : {alpha_pct}% | Slow C : {100-alpha_pct}%]", (hud_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

        # Modulator Bus
        bus = model.controller.bus
        cv2.putText(canvas, "Neuromodulatory Bus:", (hud_x, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(canvas, f"  Novelty (nu)    : {bus.nu:.2f}", (hud_x, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"  Sleep Pres (phi): {bus.phi:.1f} / {model.controller.phi_thresh}", (hud_x, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Energy Odometer
        proj = instrument.project_energy()
        cv2.putText(canvas, "Audited Hardware Telemetry:", (hud_x, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(canvas, f"  SynOps   : {instrument.counts.synops:,}", (hud_x, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"  Adds     : {instrument.counts.adds:,}", (hud_x, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"  Dense MAC: 0 (Pure Sparse)", (hud_x, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"  Energy   : {proj['energy_microjoules']:.2f} uJ [ASIC 45nm]", (hud_x, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1)

        # Slot Training Counts
        cv2.putText(canvas, "Trained Samples per Slot:", (hud_x, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
        for s_idx in range(4):
            cv2.putText(canvas, f"  Slot {s_idx+1}: {trained_counts[s_idx]}", (hud_x + (s_idx % 2) * 160, 445 + (s_idx // 2) * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Controls & Status bar
        cv2.line(canvas, (hud_x, 480), (hud_x + 380, 480), (80, 80, 80), 1)
        cv2.putText(canvas, "[1-4] Train Slot | [s] Sleep Replay", (hud_x, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        cv2.putText(canvas, "[t] Toggle Test | [q] Quit", (hud_x, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        
        cv2.putText(canvas, f"Status: {status_msg}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("MNEMA - Neuromorphic Edge Demonstrator", canvas)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_demonstrator()