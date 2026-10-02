import cv2, json, numpy as np, supervision as sv
from ultralytics import YOLO
from insightface.app import FaceAnalysis

cfg = json.load(open("config.json"))
yolo = YOLO(cfg["detector_weights"])
fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(160, 160))
tracker = sv.ByteTrack(lost_track_buffer=cfg["exit_timeout_frames"])

def embed(frame, box, pad=0.3):
    x1,y1,x2,y2 = map(int, box); w,h = x2-x1, y2-y1
    x1,y1 = max(0,int(x1-pad*w)), max(0,int(y1-pad*h))
    x2,y2 = int(x2+pad*w), int(y2+pad*h)
    faces = fa.get(frame[y1:y2, x1:x2])          # padded crop so InsightFace can align
    if not faces: return None
    f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return f.normed_embedding                      # 512-d, L2-normalized

def match(emb, gallery):                           # gallery: {face_id: embedding}
    best_id, best_sim = None, -1
    for fid, g in gallery.items():
        s = float(np.dot(emb, g))
        if s > best_sim: best_id, best_sim = fid, s
    return best_id if best_sim >= cfg["similarity_threshold"] else None

cap = cv2.VideoCapture(cfg["video_source"])
track2face, last_seen, last_crop = {}, {}, {}
idx = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    idx += 1
    if idx % cfg["detection_skip_frames"]: continue

    dets = sv.Detections.from_ultralytics(yolo(frame, conf=cfg["det_conf"], verbose=False)[0])
    tracked = tracker.update_with_detections(dets)

    for box, tid in zip(tracked.xyxy, tracked.tracker_id):
        x1,y1,x2,y2 = map(int, box)
        if min(x2-x1, y2-y1) < cfg["min_face_size"]: continue
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        last_seen[tid], last_crop[tid] = idx, crop
        if tid not in track2face:
            emb = embed(frame, box)
            if emb is None: continue
            fid = match(emb, gallery)
            if fid is None:
                fid = db.register_face(emb); gallery[fid] = emb
                log.info(f"NEW FACE registered id={fid}")
            else:
                log.info(f"Face recognized id={fid}")
            track2face[tid] = fid
            events.log_event(fid, "entry", crop)      # saves image + DB row + log line
        # else: log.debug tracking

    # exits
    for tid in [t for t,l in last_seen.items() if idx-l > cfg["exit_timeout_frames"] and t in track2face]:
        events.log_event(track2face.pop(tid), "exit", last_crop.pop(tid)); last_seen.pop(tid)

# flush remaining tracks at end of stream as exits
for tid, fid in track2face.items():
    events.log_event(fid, "exit", last_crop[tid])