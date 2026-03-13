import SimpleITK as sitk
import numpy as np
import os
import csv
import random
import json
import argparse
import re
from glob import glob
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Generate a Client-Side PACS HTML Viewer for BraTS Models.")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory containing all modality subfolders.")
    parser.add_argument("--ref", type=str, default="t1c", help="Name of the reference subfolder (Ground Truth).")
    parser.add_argument("--seg", type=str, default="seg", help="Name of the segmentation subfolder.")
    parser.add_argument("--bg", type=str, nargs='*', default=["t1n", "t2f", "t2w"], help="Names of background modality subfolders.")
    parser.add_argument("--fake", type=str, nargs='+', required=True, help="Names of synthetic model subfolders.")
    parser.add_argument("--out_dir", type=str, default="pacs_demo", help="Output directory for the HTML viewer and PNG slices.")
    parser.add_argument("--num_slices", type=int, default=10, help="Number of context-aware slices to extract per patient.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible blind A/B shuffling.")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for parallel processing.")
    return parser.parse_args()

def find_file(directory, p_id):
    """Smart globbing to handle BraTS suffixes (e.g. -t1c.nii.gz, -seg.nii.gz)"""
    matches = glob(os.path.join(directory, f"{p_id}*.nii.gz"))
    return matches[0] if matches else None

def get_clever_slices(ref_path, seg_path, requested_slices):
    """Context-Aware Slicing Engine."""
    ref_arr = sitk.GetArrayFromImage(sitk.ReadImage(ref_path))
    brain_coords = np.argwhere(ref_arr > 0)
    if brain_coords.size == 0: return None, None
    
    bz_min, bz_max = brain_coords[:, 0].min(), brain_coords[:, 0].max()
    y1, y2 = brain_coords[:, 1].min(), brain_coords[:, 1].max()
    x1, x2 = brain_coords[:, 2].min(), brain_coords[:, 2].max()
    
    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    tumor_coords = np.argwhere(seg_arr > 0)
    
    if tumor_coords.size == 0:
        b_center = (bz_min + bz_max) // 2
        spread = int((bz_max - bz_min) * 0.25)
        start, end = b_center - spread, b_center + spread
    else:
        tz_min, tz_max = tumor_coords[:, 0].min(), tumor_coords[:, 0].max()
        t_center = (tz_min + tz_max) // 2
        t_height = tz_max - tz_min
        window_size = max(t_height * 1.2, 25)
        start = max(bz_min, int(t_center - (window_size / 2)))
        end = min(bz_max, int(t_center + (window_size / 2)))
            
    slices = np.unique(np.linspace(start, end, requested_slices).astype(int))
    return slices.tolist(), (y1, y2, x1, x2)

def to_uint8_image(arr, mask):
    brain_pixels = arr[mask]
    if brain_pixels.size == 0: return np.zeros_like(arr, dtype=np.uint8)
    p_low, p_high = np.percentile(brain_pixels, [0.5, 99.5])
    arr = np.clip(arr, p_low, p_high)
    mean, std = np.mean(arr[mask]), np.std(arr[mask])
    
    normalized = (arr - mean) / (std + 1e-8)
    normalized[~mask] = -3.5 
    
    clipped = np.clip(normalized, -2.0, 4.0)
    scaled = ((clipped - -2.0) / (4.0 - -2.0)) * 255.0
    return scaled.astype(np.uint8)

def process_single_patient(args):
    p_id, ref_file, seg_file, bg_files, fake_files, out_base, num_slices = args
    try:
        patient_out_dir = os.path.join(out_base, "slices", p_id)
        os.makedirs(patient_out_dir, exist_ok=True)

        brain_mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(ref_file)) > 0
        slices, crop = get_clever_slices(ref_file, seg_file, num_slices)
        if slices is None: return None
        y1, y2, x1, x2 = crop

        volumes = {"ref": to_uint8_image(sitk.GetArrayFromImage(sitk.ReadImage(ref_file)), brain_mask_vol)}
        
        for i, bg in enumerate(bg_files):
            if bg and os.path.exists(bg):
                volumes[f"bg_{i}"] = to_uint8_image(sitk.GetArrayFromImage(sitk.ReadImage(bg)), brain_mask_vol)
        
        for i, fake in enumerate(fake_files):
            if fake and os.path.exists(fake):
                volumes[f"fake_{i}"] = to_uint8_image(sitk.GetArrayFromImage(sitk.ReadImage(fake)), brain_mask_vol)

        for z in slices:
            for mod_key, vol_arr in volumes.items():
                slice_img = vol_arr[z, y1:y2, x1:x2]
                Image.fromarray(slice_img).save(os.path.join(patient_out_dir, f"{mod_key}_{z}.png"), optimize=True)

        return p_id, slices
    except Exception as e:
        print(f"Error on {p_id}: {e}")
        return None

def generate_html(patient_metadata, bg_count, fake_count, out_html, bg_names):
    metadata_json = json.dumps(patient_metadata)
    
    bg_html = "".join([f'<div class="pacs-cell"><div class="cell-title">Real {bg_names[i]}</div><div class="img-wrapper"><img class="base-img" id="img-bg_{i}"></div></div>' for i in range(bg_count)])
    
    fake_html = ""
    for i in range(fake_count):
        color = "#569cd6" if i == 0 else "#c586c0"
        fake_html += f"""
        <div class="pacs-cell"><div class="cell-title" style="color: {color};">Model {chr(65+i)}</div><div class="img-wrapper">
            <img class="base-img" id="img-base-fake_{i}">
            <img class="overlay-img" id="img-syn-fake_{i}">
        </div></div>"""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Client-Side PACS Viewer</title>
        <style>
            :root {{ --bg-main: #0a0a0c; --bg-panel: #141417; --bg-hover: #1f1f24; --border-color: #2a2a30; --accent-blue: #007acc; --accent-red: #c62828; --text-main: #e0e0e0; --text-muted: #888; }}
            body {{ display: flex; height: 100vh; margin: 0; font-family: -apple-system, sans-serif; background: var(--bg-main); color: var(--text-main); overflow: hidden; user-select: none; }}
            #sidebar {{ width: 250px; background: var(--bg-panel); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; z-index: 10; }}
            #sidebar-header {{ padding: 12px; border-bottom: 1px solid var(--border-color); text-align: center; }}
            #progress-container {{ width: 100%; height: 6px; background: #222; margin-top: 10px; border-radius: 3px; overflow: hidden; }}
            #progress-fill {{ width: 0%; height: 100%; background: var(--accent-blue); transition: width 0.3s ease; }}
            #patient-list {{ flex: 1; overflow-y: auto; }}
            .patient-link {{ padding: 10px 15px; cursor: pointer; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; font-size: 13px; }}
            .patient-link:hover {{ background: var(--bg-hover); }}
            .active {{ background: var(--bg-hover) !important; border-left: 3px solid var(--accent-blue); padding-left: 12px; color: #fff; font-weight: bold; }}
            .status {{ font-size: 11px; padding: 2px 6px; border-radius: 8px; background: #2a2a2a; color: var(--text-muted); }}
            .status.done {{ background: #1e4620; color: #4caf50; border: 1px solid #2e7d32; }}
            
            #main {{ flex: 1; display: flex; flex-direction: column; min-width: 0; position: relative; }}
            #toolbar {{ background: var(--bg-panel); border-bottom: 1px solid var(--border-color); padding: 10px 20px; gap: 12px; z-index: 10; }}
            .toolbar-row {{ display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 8px; }}
            .toolbar-group {{ display: flex; align-items: center; gap: 12px; }}
            .btn {{ padding: 6px 12px; border: 1px solid transparent; border-radius: 4px; font-size: 13px; cursor: pointer; background: #2d2d30; color: #fff; }}
            .btn:hover {{ background: #3e3e42; }}
            .btn-danger {{ background: var(--accent-red); }}
            .btn-export {{ background: #2e7d32; border: 1px solid #4caf50; }}
            .btn-import {{ background: #ff9800; border: 1px solid #f57c00; color: #000; }}
            
            .slider-group {{ display: flex; align-items: center; gap: 8px; background: #000; padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border-color); }}
            input[type=range] {{ cursor: pointer; accent-color: var(--accent-blue); }}
            
            #scoring-panel {{ display: flex; justify-content: center; gap: 30px; background: #000; padding: 10px 20px; border-radius: 8px; border: 1px solid var(--border-color); }}
            .model-score-group {{ display: flex; flex-direction: column; gap: 8px; border-right: 1px solid var(--border-color); padding-right: 30px; }}
            .model-score-group:last-child {{ border-right: none; padding-right: 0; }}
            .model-title {{ font-weight: 600; font-size: 14px; text-align: center; margin-bottom: 4px; }}
            .rating-row {{ display: flex; align-items: center; justify-content: flex-end; gap: 10px; }}
            .rate-btn {{ width: 32px; height: 32px; border-radius: 16px; border: 1px solid #444; background: #1e1e1e; color: #aaa; font-weight: bold; cursor: pointer; }}
            .rate-btn.selected {{ background: var(--accent-blue); color: #fff; border-color: #66b2ff; transform: scale(1.1); }}

            #viewer-container {{ flex: 1; padding: 20px; background: var(--bg-main); display: flex; flex-direction: column; align-items: center; justify-content: center; }}
            .pacs-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; width: 100%; max-width: 1200px; }}
            .pacs-cell {{ background: #000; border: 1px solid #333; position: relative; border-radius: 4px; }}
            .cell-title {{ text-align: center; padding: 6px; font-weight: bold; font-size: 13px; background: #111; color: #ccc; border-bottom: 1px solid #333; }}
            .img-wrapper {{ position: relative; width: 100%; aspect-ratio: 1; }}
            .base-img, .overlay-img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }}
            
            /* Default to showing the full Synthetic image (opacity 1) */
            .overlay-img {{ opacity: 1; transition: opacity 0.1s; }}
            
            #shortcuts-legend {{ position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.8); border: 1px solid var(--border-color); padding: 15px; border-radius: 8px; font-size: 12px; pointer-events: none; z-index: 50; }}
            .key {{ background: #333; padding: 2px 6px; border-radius: 4px; color: #fff; font-family: monospace; border: 1px solid #555; }}
        </style>
    </head>
    <body>
        <div id="sidebar">
            <div id="sidebar-header">
                <strong>Patient Roster</strong>
                <div id="progress-container"><div id="progress-fill"></div></div>
                <div id="progress-text" style="font-size: 11px; margin-top: 5px; color: var(--accent-blue);">0 / 0 Evaluated</div>
            </div>
            <div id="patient-list"></div>
            <div style="padding: 10px; border-top: 1px solid var(--border-color);">
                 <button class="btn btn-danger" style="width: 100%;" onclick="resetEvaluation()">Reset All</button>
            </div>
        </div>
        
        <div id="main">
            <div id="toolbar">
                <div class="toolbar-row">
                    <div class="toolbar-group">
                        <button class="btn" onclick="navigatePatient(-1)">&#9664;</button>
                        <div id="current-label" style="font-family: monospace; font-weight: bold; font-size: 1.2rem; min-width: 150px; text-align:center;">ID</div>
                        <button class="btn" onclick="navigatePatient(1)">&#9654;</button>
                    </div>
                    
                    <div class="toolbar-group">
                        <div class="slider-group">
                            <span id="slice-label" style="min-width:60px;">Slice: 0</span>
                            <input type="range" id="slice-slider" min="0" value="0" oninput="updateSlice(this.value)">
                        </div>
                        <div class="slider-group">
                            <span>Overlay:</span>
                            <span style="color:#fff;">Real</span>
                            <input type="range" id="opacity-slider" min="0" max="1" step="0.1" value="1" oninput="updateOpacity(this.value)">
                            <span style="color:#fff;">Syn</span>
                        </div>
                    </div>

                    <div class="toolbar-group">
                        <label style="font-size: 13px; color: #aaa; cursor:pointer;"><input type="checkbox" id="auto-next-toggle" checked> Auto-Next</label>
                        <label for="csv-upload" class="btn btn-import" style="display:inline-block; margin-bottom:0;">Load CSV</label>
                        <input type="file" id="csv-upload" accept=".csv" onchange="importCSV(event)" style="display:none;">
                        <button class="btn btn-export" onclick="exportCSV()">Export CSV</button>
                    </div>
                </div>
                
                <div class="toolbar-row" style="justify-content: center;">
                    <div id="scoring-panel">
                        {"".join([f'''<div class="model-score-group">
                            <div class="model-title" style="color: {"#569cd6" if i==0 else "#c586c0"};">Model {chr(65+i)}</div>
                            <div class="rating-row"><span class="rating-label" style="font-size:12px; width:80px; text-align:right; color:#ccc;">Quality</span><div class="rating-btn-group" id="grp-fake_{i}_q"></div></div>
                            <div class="rating-row"><span class="rating-label" style="font-size:12px; width:80px; text-align:right; color:#ccc;">Clin Value</span><div class="rating-btn-group" id="grp-fake_{i}_v"></div></div>
                        </div>''' for i in range(fake_count)])}
                    </div>
                </div>
            </div>
            
            <div id="viewer-container">
                <div class="pacs-grid">
                    {bg_html}
                    <div class="pacs-cell"><div class="cell-title">Real Reference</div><div class="img-wrapper"><img class="base-img" id="img-ref"></div></div>
                    {fake_html}
                </div>
            </div>
            
            <div id="shortcuts-legend">
                <div style="margin-bottom: 6px; font-weight:bold; color:#fff;">Viewer Shortcuts</div>
                <div><span class="key">Wheel</span> Scroll Volume</div>
                <div style="margin-top: 4px;"><span class="key">Ctrl</span> + <span class="key">Wheel</span> Adjust Opacity</div>
                <div style="margin-top: 4px;"><span class="key">T</span> Toggle Overlay</div>
            </div>
        </div>

        <script>
            const patientData = {metadata_json};
            const patients = Object.keys(patientData).sort();
            const fakeCount = {fake_count};
            const bgCount = {bg_count};
            let currentIndex = 0;
            let currentSlices = [];
            let evaluations = {{}};
            let criteriaKeys = [];
            
            for(let i=0; i<fakeCount; i++) {{
                criteriaKeys.push(`fake_${{i}}_q`);
                criteriaKeys.push(`fake_${{i}}_v`);
            }}

            const sliceSlider = document.getElementById('slice-slider');
            const opacitySlider = document.getElementById('opacity-slider');
            const savedData = localStorage.getItem('pacs_viewer_evaluations');
            if (savedData) try {{ evaluations = JSON.parse(savedData); }} catch(e) {{}}

            criteriaKeys.forEach(key => {{
                const group = document.getElementById('grp-' + key);
                for(let i=1; i<=5; i++) {{
                    const btn = document.createElement('button');
                    btn.className = 'rate-btn'; btn.innerText = i;
                    btn.onclick = () => setScore(key, i);
                    group.appendChild(btn);
                }}
            }});

            const listElement = document.getElementById('patient-list');
            patients.forEach((pid, index) => {{
                const div = document.createElement('div');
                div.className = 'patient-link'; div.id = 'link-' + index;
                const isScored = isFullyScored(evaluations[pid]);
                div.innerHTML = `<span>${{pid}}</span> <span class="status ${{isScored?'done':''}}">${{isScored?'Scored':'Pending'}}</span>`;
                div.onclick = () => loadPatient(index);
                listElement.appendChild(div);
            }});

            function isFullyScored(obj) {{
                if(!obj) return false;
                return criteriaKeys.every(k => obj[k] !== null && obj[k] !== undefined);
            }}
            
            function updateProgress() {{
                const total = patients.length;
                const done = Object.values(evaluations).filter(isFullyScored).length;
                document.getElementById('progress-text').innerText = `${{done}} / ${{total}} Evaluated`;
                document.getElementById('progress-fill').style.width = ((done/total)*100) + "%";
            }}

            function loadPatient(index) {{
                if (index < 0 || index >= patients.length) return;
                document.getElementById('link-' + currentIndex).classList.remove('active');
                currentIndex = index;
                const pid = patients[currentIndex];
                
                document.getElementById('current-label').innerText = pid;
                document.getElementById('link-' + currentIndex).classList.add('active');
                document.getElementById('link-' + currentIndex).scrollIntoView({{behavior: "smooth", block: "nearest"}});

                currentSlices = patientData[pid];
                sliceSlider.max = currentSlices.length - 1;
                updateSlice(Math.floor(currentSlices.length / 2));

                document.querySelectorAll('.rate-btn').forEach(btn => btn.classList.remove('selected'));
                if(evaluations[pid]) {{
                    criteriaKeys.forEach(key => {{
                        if(evaluations[pid][key]) {{
                            document.getElementById('grp-' + key).children[evaluations[pid][key] - 1].classList.add('selected');
                        }}
                    }});
                }}
            }}

            function updateSlice(index) {{
                sliceSlider.value = index;
                const z = currentSlices[index];
                document.getElementById('slice-label').innerText = `Slice: ${{z}}`;
                
                const dir = `slices/${{patients[currentIndex]}}/`;
                document.getElementById('img-ref').src = dir + `ref_${{z}}.png`;
                
                for(let i=0; i<bgCount; i++) document.getElementById(`img-bg_${{i}}`).src = dir + `bg_${{i}}_${{z}}.png`;
                
                for(let i=0; i<fakeCount; i++) {{
                    document.getElementById(`img-base-fake_${{i}}`).src = dir + `ref_${{z}}.png`;
                    document.getElementById(`img-syn-fake_${{i}}`).src = dir + `fake_${{i}}_${{z}}.png`;
                }}
            }}

            document.getElementById('viewer-container').addEventListener('wheel', (e) => {{
                e.preventDefault(); 
                if (e.shiftKey || e.ctrlKey || e.metaKey) {{
                    let currOpacity = Number(opacitySlider.value);
                    let wheelDelta = (Math.abs(e.deltaY) > Math.abs(e.deltaX)) ? e.deltaY : e.deltaX;
                    let newOpacity = Math.max(0, Math.min(1, currOpacity + ((wheelDelta < 0) ? 0.1 : -0.1)));
                    opacitySlider.value = newOpacity.toFixed(2);
                    updateOpacity(newOpacity);
                }} else {{
                    let currSlice = parseInt(sliceSlider.value);
                    if (e.deltaY > 0 && currSlice < sliceSlider.max) updateSlice(currSlice + 1);
                    else if (e.deltaY < 0 && currSlice > 0) updateSlice(currSlice - 1);
                }}
            }}, {{ passive: false }});

            document.addEventListener('keydown', (e) => {{
                if (e.key.toLowerCase() === 't') {{
                    let curr = Number(opacitySlider.value);
                    let newOp = curr > 0.5 ? 0 : 1;
                    opacitySlider.value = newOp;
                    updateOpacity(newOp);
                }}
            }});

            function updateOpacity(val) {{
                for(let i=0; i<fakeCount; i++) document.getElementById(`img-syn-fake_${{i}}`).style.opacity = val;
            }}

            function navigatePatient(dir) {{ loadPatient(currentIndex + dir); }}

            function setScore(key, val) {{
                const pid = patients[currentIndex];
                if(!evaluations[pid]) evaluations[pid] = {{}};
                evaluations[pid][key] = val;

                const grp = document.getElementById('grp-' + key);
                Array.from(grp.children).forEach(b => b.classList.remove('selected'));
                grp.children[val - 1].classList.add('selected');

                localStorage.setItem('pacs_viewer_evaluations', JSON.stringify(evaluations));

                if(isFullyScored(evaluations[pid])) {{
                    document.getElementById('link-' + currentIndex).innerHTML = `<span>${{pid}}</span> <span class="status done">Scored</span>`;
                    updateProgress();
                    if(document.getElementById('auto-next-toggle').checked && currentIndex < patients.length - 1) {{
                        setTimeout(() => navigatePatient(1), 250);
                    }}
                }}
            }}

            function resetEvaluation() {{
                if(confirm("WARNING: This will delete the current ratings in this browser. Please save into csv before doing that.")) {{
                    localStorage.removeItem('pacs_viewer_evaluations');
                    location.reload();
                }}
            }}

            function exportCSV() {{
                let headers = "Patient_ID";
                for(let i=0; i<fakeCount; i++) headers += `,Model_${{chr(65+i)}}_Quality,Model_${{chr(65+i)}}_ClinValue`;
                let csv = "data:text/csv;charset=utf-8," + headers + "\\n";
                
                patients.forEach(pid => {{
                    csv += pid;
                    for(let i=0; i<fakeCount; i++) {{
                        let q = evaluations[pid] ? (evaluations[pid][`fake_${{i}}_q`] || '') : '';
                        let v = evaluations[pid] ? (evaluations[pid][`fake_${{i}}_v`] || '') : '';
                        csv += `,${{q}},${{v}}`;
                    }}
                    csv += "\\n";
                }});
                const link = document.createElement("a");
                link.setAttribute("href", encodeURI(csv));
                link.setAttribute("download", "radiologist_evaluation.csv");
                document.body.appendChild(link); link.click(); document.body.removeChild(link);
            }}

            function chr(ascii) {{ return String.fromCharCode(ascii); }}

            if(patients.length > 0) {{ updateProgress(); loadPatient(0); }}
        </script>
    </body>
    </html>
    """
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    args = get_args()
    os.makedirs(os.path.join(args.out_dir, "slices"), exist_ok=True)
    random.seed(args.seed)

    # Resolve full paths using the base directory
    ref_dir = os.path.join(args.base_dir, args.ref)
    seg_dir = os.path.join(args.base_dir, args.seg)
    bg_dirs = [os.path.join(args.base_dir, d) for d in args.bg]
    fake_dirs = [os.path.join(args.base_dir, d) for d in args.fake]

    ref_files = sorted(glob(os.path.join(ref_dir, "*.nii.gz")))
    patients = []
    
    for f in ref_files:
        fname = os.path.basename(f)
        p_id = re.sub(r'[-_][a-zA-Z0-9]+\.nii\.gz$', '', fname)
        if p_id == fname: p_id = fname.replace('.nii.gz', '')
        patients.append(p_id)
        
    patients = sorted(list(set(patients)))
    print(f"Found {len(patients)} unique patients. Preparing extraction...")

    task_args = []
    shuffling_records = []
    
    for p_id in patients:
        ref_file = find_file(ref_dir, p_id)
        seg_file = find_file(seg_dir, p_id)
        
        if not ref_file or not seg_file: continue
            
        bg_files = [find_file(d, p_id) for d in bg_dirs]
        fake_files = [find_file(d, p_id) for d in fake_dirs]
        
        combined = list(zip(args.fake, fake_files))
        random.shuffle(combined)
        shuffled_dirs, shuffled_files = zip(*combined)
        
        task_args.append((p_id, ref_file, seg_file, bg_files, list(shuffled_files), args.out_dir, args.num_slices))
        
        record = {'patient_id': p_id}
        for i, source_dir in enumerate(shuffled_dirs):
            record[f'Model_{chr(65+i)}'] = os.path.basename(os.path.normpath(source_dir))
        shuffling_records.append(record)

    key_path = os.path.join(args.out_dir, "evaluation_key.csv")
    if shuffling_records:
        with open(key_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=shuffling_records[0].keys())
            writer.writeheader()
            writer.writerows(shuffling_records)

    print(f"Extracting volumes using {args.workers} workers...")
    patient_metadata = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process_single_patient, task_args), total=len(task_args)):
            if result:
                p_id, slices = result
                patient_metadata[p_id] = slices

    print("Building HTML Viewer...")
    out_html = os.path.join(args.out_dir, "index.html")
    generate_html(patient_metadata, len(args.bg), len(args.fake), out_html, args.bg)
    print(f"✅ Success! Open '{out_html}' in your browser. Decoding key saved to {key_path}")

if __name__ == "__main__":
    main()