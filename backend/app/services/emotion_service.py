from __future__ import annotations

import csv
import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from app import db
from app.config import Settings
from app.schemas import EMOTIONS, NEGATIVE_EMOTIONS, EmotionName, EmotionScore, FaceDetection, FrameResult, InferenceResponse, VideoInferenceResponse

LABEL_CN = {
    "anger": "\u751f\u6c14",
    "contempt": "\u8f7b\u8511",
    "disgust": "\u538c\u6076",
    "fear": "\u5bb3\u6015",
    "happy": "\u5f00\u5fc3",
    "neutral": "\u5e73\u9759",
    "sad": "\u4f24\u5fc3",
    "surprise": "\u60ca\u8bb6",
}


class EmotionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._replay_rows: list[dict[str, str]] | None = None
        self._replay_index = 0
        self._face_cascade = None
        self._local_emotion_model = None
        self._local_emotion_transform = None
        self._emotion_calibrator = None
        self._emotion_calibrator_schema = None
        self._video_label_priors = None
        self._identity_service = None
        self._reid_model = None
        self._reid_transform = None
        self._reid_backbone_failed = False

    def _identity(self):
        """Lazy IdentityService (cross-video anonymous re-ID)."""
        if self._identity_service is None:
            from app.services.identity_service import IdentityService
            self._identity_service = IdentityService(self.settings)
        return self._identity_service

    _CASCADES_INITED = False
    _FRONTAL_CASCADES: list = []
    _PROFILE_CASCADE = None

    def _init_cascades(self):
        if self._CASCADES_INITED:
            return
        import cv2
        d = cv2.data.haarcascades
        # Load multiple frontal cascades
        for name in ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]:
            cc = cv2.CascadeClassifier(d + name)
            if not cc.empty():
                self._FRONTAL_CASCADES.append(cc)
        # Load profile cascade for side faces
        pc = cv2.CascadeClassifier(d + "haarcascade_profileface.xml")
        if not pc.empty():
            self._PROFILE_CASCADE = pc
        self.__class__._CASCADES_INITED = True

    @staticmethod
    def _nms_bboxes(bboxes: list[list[int]], iou_thresh: float = 0.45) -> list[list[int]]:
        """Non-Maximum Suppression: merge overlapping detections, keep larger ones."""
        if not bboxes:
            return []

        def iou(a: list[int], b: list[int]) -> float:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            ix1 = max(ax, bx)
            iy1 = max(ay, by)
            ix2 = min(ax + aw, bx + bw)
            iy2 = min(ay + ah, by + bh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0

        sorted_boxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
        keep: list[list[int]] = []
        for box in sorted_boxes:
            if all(iou(box, kept) <= iou_thresh for kept in keep):
                keep.append(box)
        return keep

    def _detect_faces(self, image_bytes: bytes) -> tuple[list[list[int]], tuple[int, int]]:
        """Detect faces. Tries MediaPipe BlazeFace first; falls back to multi-cascade Haar with score filtering."""
        from io import BytesIO

        try:
            pil_img = Image.open(BytesIO(image_bytes))
            w, h = pil_img.size

            # Try MediaPipe (if model available)
            bboxes = self._detect_faces_mediapipe(pil_img, w, h)
            
            # --- DEBUG: Completely disable Haar fallback temporarily to force MediaPipe debugging ---
            # if not bboxes:
            #    bboxes = self._detect_faces_haar_scored(image_bytes, w, h)

            bboxes = bboxes[:3]
            bboxes.sort(key=lambda b: b[0])
            return bboxes, (w, h)
        except Exception:
            return [], (360, 240)

    # ---- MediaPipe BlazeFace (deep learning, accurate) ----
    _MP_FACE_DETECTOR = None

    def _get_mp_face_detector(self):
        if self._MP_FACE_DETECTOR is None:
            from mediapipe.tasks import python
            from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
            model_path = Path(__file__).parent.parent / "models" / "face_detection_short_range.tflite"
            if model_path.exists():
                options = FaceDetectorOptions(
                    base_options=python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                self.__class__._MP_FACE_DETECTOR = FaceDetector.create_from_options(options)
        return self._MP_FACE_DETECTOR

    def _looks_like_face_region(self, rgb, box: list[int], frame_area: int) -> bool:
        """Cheap second-stage verifier for MediaPipe false positives.

        MediaPipe may assign high face scores to balloons, hair clumps, clothing folds,
        machine parts, or dense foliage. This verifier rejects candidates whose cropped
        region has implausible size/aspect/color/texture for a visible human face.
        It is intentionally conservative because the demo prioritizes fewer false boxes.
        """
        try:
            import cv2
            import numpy as np

            h, w = rgb.shape[:2]
            x, y, bw, bh = [int(v) for v in box]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))
            area = bw * bh
            if area < frame_area * 0.012 or area > frame_area * 0.38:
                return False
            ratio = bw / max(1, bh)
            if not (0.62 <= ratio <= 1.45):
                return False

            crop = rgb[y:y + bh, x:x + bw]
            if crop.size == 0:
                return False

            # Focus on the central face-like area; outer margins often include hair/background.
            mx = int(bw * 0.18)
            my = int(bh * 0.18)
            core = crop[my:max(my + 1, bh - my), mx:max(mx + 1, bw - mx)]
            if core.size == 0:
                core = crop

            hsv = cv2.cvtColor(core, cv2.COLOR_RGB2HSV)
            ycrcb = cv2.cvtColor(core, cv2.COLOR_RGB2YCrCb)
            H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

            # Broad skin mask: catches light/dark skin under common lighting while rejecting
            # saturated balloons/green foliage/metal/cloth. Require only a modest ratio.
            skin_hsv = ((H <= 25) | (H >= 170)) & (S >= 18) & (S <= 180) & (V >= 45)
            skin_ycc = (Cr >= 125) & (Cr <= 180) & (Cb >= 80) & (Cb <= 140) & (Y >= 35)
            skin_ratio = float(np.mean(skin_hsv | skin_ycc))

            # Very saturated colorful regions are usually balloons/clothes/objects, not faces.
            saturated_ratio = float(np.mean((S > 165) & (V > 80)))

            gray = cv2.cvtColor(core, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_ratio = float(np.mean(edges > 0))
            brightness = float(np.mean(gray))

            # Require some plausible skin unless the box is large and smooth enough.
            if skin_ratio < 0.055:
                return False
            if saturated_ratio > 0.42 and skin_ratio < 0.18:
                return False
            if edge_ratio > 0.32 and skin_ratio < 0.16:
                return False
            if brightness < 18 or brightness > 245:
                return False
            return True
        except Exception:
            # On verifier failure, prefer rejecting rather than drawing a false box.
            return False

    def _detect_faces_mediapipe(self, pil_img, w: int, h: int) -> list[list[int]]:
        """Run BlazeFace, return [x,y,w,h] bboxes. Filters by score, NMS merges, keeps best."""
        import traceback
        try:
            detector = self._get_mp_face_detector()
            if detector is None:
                print("MediaPipe detector is None, falling back")
                return []
            from mediapipe import Image as MpImage, ImageFormat
            import numpy as np
            rgb = np.array(pil_img.convert('RGB'))
            mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
            if not result.detections:
                return []
            # Collect (bbox, score) and sort
            candidates = []
            frame_area = w * h
            for det in result.detections:
                bb = det.bounding_box
                score = det.categories[0].score if det.categories else 0.0
                if not bb:
                    continue

                # Keep a high detector score, then verify appearance. High detector score alone
                # still misfires on balloons, hair, foliage, clothes, and machine parts.
                if score < 0.64:
                    continue

                raw_box = [bb.origin_x, bb.origin_y, bb.width, bb.height]
                area = bb.width * bb.height
                if area < frame_area * 0.012 or area > frame_area * 0.38:
                    continue
                ratio = bb.width / max(1.0, bb.height)
                if not (0.62 <= ratio <= 1.45):
                    continue
                cx = bb.origin_x + bb.width / 2
                cy = bb.origin_y + bb.height / 2
                if cx < w * 0.03 or cx > w * 0.97 or cy < h * 0.03 or cy > h * 0.97:
                    continue
                if not self._looks_like_face_region(rgb, raw_box, frame_area):
                    continue

                candidates.append({
                    'bbox': raw_box,
                    'score': score,
                })
            if not candidates:
                return []
            # Sort by score descending
            candidates.sort(key=lambda c: c['score'], reverse=True)
            # NMS + near-duplicate pruning. MediaPipe can return one true face plus a
            # nearby hair/forehead/object box; IoU may be low, so also suppress boxes
            # whose centers are close and sizes are comparable.
            kept = []
            for c in candidates:
                overlap = False
                for k in kept:
                    ax, ay, aw, ah = c['bbox']
                    bx, by, bw, bh = k['bbox']
                    ix1, iy1 = max(ax, bx), max(ay, by)
                    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = aw * ah + bw * bh - inter
                    iou = inter / union if union > 0 else 0.0
                    acx, acy = ax + aw / 2, ay + ah / 2
                    bcx, bcy = bx + bw / 2, by + bh / 2
                    center_dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
                    near_dist = max(aw, ah, bw, bh) * 0.75
                    size_ratio = (aw * ah) / max(1, bw * bh)
                    near_duplicate = center_dist < near_dist and 0.45 <= size_ratio <= 2.2
                    if iou > 0.35 or near_duplicate:
                        overlap = True
                        break
                if not overlap:
                    kept.append(c)
            # Clip and format
            final_boxes = []
            for k in kept[:3]:
                x, y, bw, bh = k['bbox']
                margin_x = int(bw * 0.05)
                margin_y = int(bh * 0.1)
                x = max(0, min(x - margin_x, w - 1))
                y = max(0, min(y - margin_y * 2, h - 1))
                bw = max(1, min(bw + margin_x * 2, w - x))
                bh = max(1, min(bh + margin_y * 3, h - y))
                final_boxes.append([x, y, bw, bh])
            return final_boxes
        except Exception as e:
            print(f"MediaPipe error: {e}")
            return []
    # ---- Haar Cascade fallback (for when MediaPipe is unavailable) ----    _CASCADES_INITED = False
    _FRONTAL_CASCADES: list = []
    _PROFILE_CASCADE = None

    def _init_cascades(self):
        if self._CASCADES_INITED:
            return
        import cv2
        d = cv2.data.haarcascades
        for name in ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]:
            cc = cv2.CascadeClassifier(d + name)
            if not cc.empty():
                self._FRONTAL_CASCADES.append(cc)
        pc = cv2.CascadeClassifier(d + "haarcascade_profileface.xml")
        if not pc.empty():
            self._PROFILE_CASCADE = pc
        self.__class__._CASCADES_INITED = True

    def _detect_faces_haar_scored(self, image_bytes: bytes, w: int, h: int) -> list[list[int]]:
        """Conservative multi-cascade Haar fallback.

        To reduce background false positives, only frontal cascades create new detections.
        Side/profile continuity is handled by the tracker holding an existing track briefly,
        rather than allowing profile cascade to create new boxes from background texture.
        """
        import cv2
        import numpy as np
        try:
            pil_img = Image.open(BytesIO(image_bytes))
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            self._init_cascades()
            min_sz = max(56, min(w, h) // 12)
            frame_area = max(1, w * h)

            def _clip_box(box: list[int]) -> list[int] | None:
                x, y, bw, bh = [int(v) for v in box]
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                bw = max(1, min(bw, w - x))
                bh = max(1, min(bh, h - y))
                area = bw * bh
                if bw < min_sz or bh < min_sz:
                    return None
                if area < frame_area * 0.018 or area > frame_area * 0.45:
                    return None
                ratio = bw / max(bh, 1)
                if not (0.72 <= ratio <= 1.33):
                    return None
                return [x, y, bw, bh]

            def _iou(a: list[int], b: list[int]) -> float:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                ix1, iy1 = max(ax, bx), max(ay, by)
                ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = aw * ah + bw * bh - inter
                return inter / union if union > 0 else 0.0

            def _nms(boxes: list[list[int]], iou_thresh: float = 0.35) -> list[list[int]]:
                if not boxes:
                    return []
                sorted_b = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
                kept: list[list[int]] = []
                for b in sorted_b:
                    if all(_iou(b, k) <= iou_thresh for k in kept):
                        kept.append(b)
                return kept

            def _score_dets(dets: list[list[int]], iou_th: float = 0.18) -> list[tuple[list[int], int]]:
                scored = []
                for i, a in enumerate(dets):
                    hits = 0
                    for j, b in enumerate(dets):
                        if i != j and _iou(a, b) > iou_th:
                            hits += 1
                    scored.append((a, hits))
                return scored

            candidates: list[list[int]] = []

            # Frontal path only. Keep it fairly strict to avoid background false positives.
            frontal_raw: list[list[int]] = []
            for cascade in self._FRONTAL_CASCADES:
                for gray_in in (gray_eq, gray):
                    for sf, mn in [(1.10, 5), (1.16, 6)]:
                        faces = cascade.detectMultiScale(
                            gray_in,
                            scaleFactor=sf,
                            minNeighbors=mn,
                            minSize=(min_sz, min_sz),
                        )
                        for fx, fy, fw, fh in faces:
                            box = _clip_box([fx, fy, fw, fh])
                            if box:
                                frontal_raw.append(box)

            for box, hits in _score_dets(frontal_raw):
                # Must be supported by at least one overlapping cascade pass.
                if hits >= 1:
                    candidates.append(box)

            candidates = _nms(candidates, iou_thresh=0.35)
            candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
            return candidates[:2]
        except Exception:
            return []
    async def predict_image(self, image_bytes: bytes, filename: str | None = None) -> InferenceResponse:
        mode = self.settings.inference_mode.lower()
        image_size = self._read_image_size(image_bytes)
        if mode == "cloud":
            return await self._predict_cloud(image_bytes, filename)
        if mode == "replay":
            return self._predict_replay(filename, image_size=image_size)
        if mode == "local":
            # Real on-device inference: MediaPipe detection + local ViT emotion.
            return self._predict_local(image_bytes, filename, image_size=image_size)
        return self._predict_mock(image_bytes, filename, image_size=image_size)

    def _read_image_size(self, image_bytes: bytes) -> tuple[int, int] | None:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                return img.size
        except Exception:
            return None

    _FACE_FALLBACK_ZONES = [
        [0.15, 0.06, 0.28, 0.46],  # person-1: left
        [0.57, 0.08, 0.28, 0.44],  # person-2: right
        [0.34, 0.30, 0.26, 0.42],  # person-3: center-lower
    ]

    def _predict_mock(self, image_bytes: bytes, filename: str | None = None, image_size: tuple[int, int] | None = None, prev_emotions: list[str | None] | None = None) -> InferenceResponse:
        # Try real face detection first
        face_bboxes, img_size = self._detect_faces(image_bytes)
        if image_size:
            img_size = image_size

        if face_bboxes:
            detections = self._build_detections(face_bboxes, image_bytes, filename, img_size, prev_emotions)
        else:
            # Do not fabricate faces for normal uploaded images/videos. The old demo fallback
            # produced three fixed boxes that looked like persistent false positives on shelves,
            # balloons, hair, or body regions. Keep real inference honest: no face means no box.
            detections = []

        w, h = img_size
        return InferenceResponse(
            source=filename or "upload",
            mode="mock",
            num_faces=len(detections),
            detections=detections,
            meta={
                "mock": True,
                "face_source": "mediapipe" if face_bboxes else "none",
                "sha256_prefix": hashlib.sha256(image_bytes).hexdigest()[:12],
                "bbox_source": "detected_region" if face_bboxes else "none",
                "image_size": [w, h],
                "image_width": w,
                "image_height": h,
            },
        )

    def _predict_local(self, image_bytes: bytes, filename: str | None = None, image_size: tuple[int, int] | None = None, prev_emotions: list[str | None] | None = None) -> InferenceResponse:
        """Real local inference: MediaPipe face detection + on-device ViT emotion.

        This is the "no-cloud, real model" mode. It crops every detected face,
        runs the local ViT (calibrated weight preferred) and reports its actual
        ranked probabilities. No fabricated/hash-based labels are used.
        """
        import cv2

        face_bboxes, img_size = self._detect_faces(image_bytes)
        if image_size:
            img_size = image_size

        if not face_bboxes:
            w, h = img_size
            return InferenceResponse(
                source=filename or "upload",
                mode="local",
                num_faces=0,
                detections=[],
                meta={
                    "face_source": "none",
                    "emotion_source": "local_vit_b16",
                    "image_size": [w, h],
                    "image_width": w,
                    "image_height": h,
                    "note": "no face detected",
                },
            )

        # Decode once into a BGR frame for square face-crop extraction.
        import numpy as np
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Build neutral skeleton detections, then overwrite emotions with real model output.
        request_id = str(uuid.uuid4())
        detections = self._build_detections(face_bboxes, image_bytes, filename, img_size, prev_emotions)
        classified = []
        track_embeddings: dict[str, list] = {}
        det_crops: dict[str, bytes] = {}
        for det in detections:
            crop_bytes = self._crop_face_jpeg_from_frame(frame, det.bbox)
            if crop_bytes:
                det_crops[det.track_id] = crop_bytes
            ranked, emb = (
                self._classify_and_embed_face_crop_local(crop_bytes) if crop_bytes else ([], None)
            )
            if ranked:
                det = self._apply_ranked_emotions_to_detection(det, ranked)
            if emb is not None:
                track_embeddings.setdefault(det.track_id, []).append(emb)
            classified.append(det)

        # Anonymous re-ID: assign a stable pseudonym per detected face.
        identity_decisions = []
        for det in classified:
            decision = self._resolve_track_identity(
                det.track_id, track_embeddings.get(det.track_id) or [], request_id
            )
            if decision:
                det.anonymous_id = decision["anon_code"]
                identity_decisions.append(decision)

        # Session-only claim thumbnails (RAM buffer, claim-to-burn).
        if det_crops and identity_decisions:
            try:
                from app.thumb_buffer import thumb_buffer
                for tid, jpeg in det_crops.items():
                    for dec in identity_decisions:
                        if dec["track_id"] == tid:
                            thumb_buffer.put(request_id, tid, dec["anon_code"], jpeg)
                            break
            except Exception as exc:
                print(f"Thumb buffer registration error: {type(exc).__name__}: {str(exc)[:120]}")

        w, h = img_size
        return InferenceResponse(
            request_id=request_id,
            source=filename or "upload",
            mode="local",
            num_faces=len(classified),
            detections=classified,
            track_identity_decisions=identity_decisions,
            meta={
                "mock": False,
                "local_model": True,
                "face_source": "mediapipe" if face_bboxes else "none",
                "emotion_source": "local_vit_b16",
                "sha256_prefix": hashlib.sha256(image_bytes).hexdigest()[:12],
                "bbox_source": "detected_region",
                "image_size": [w, h],
                "image_width": w,
                "image_height": h,
            },
        )

    def _build_detections(
        self,
        bboxes: list[list[int]],
        image_bytes: bytes,
        filename: str | None = None,
        image_size: tuple[int, int] = (360, 240),
        prev_emotions: list[str | None] | None = None,
        camera_id: str = "camera-1",
        anon_secret: str = "dev-only-change-me",
    ) -> list[FaceDetection]:
        """Build FaceDetection list from raw bboxes with deterministic mock emotions.

        ``anonymous_id`` is the operator-facing pseudonym derived from the
        camera + the short-lived track key. The raw ``person-N`` label stays
        internal only (used to match tracks inside one request).
        """
        from app import db as _db
        digest = hashlib.sha256(image_bytes + (filename or "").encode("utf-8")).digest()
        palette = ["happy", "neutral", "sad", "fear", "anger", "surprise", "contempt", "disgust"]

        detections = []
        for i, (fx, fy, fw, fh) in enumerate(bboxes):
            track_label = f"person-{i + 1}"
            anon_id = _db.pseudonymise(camera_id, track_label, anon_secret)
            idx = (digest[0] + i * 37 + digest[(1 + i) % len(digest)] * 7) % len(EMOTIONS)

            if prev_emotions and i < len(prev_emotions) and prev_emotions[i] in EMOTIONS:
                if digest[(2 + i * 3) % len(digest)] % 10 < 7:  # 70% temporal coherence
                    idx = EMOTIONS.index(prev_emotions[i])  # type: ignore

            primary = palette[idx]
            base_conf = 0.50 + (digest[(1 + i * 5) % len(digest)] / 255) * 0.40

            scores = {emotion: 0.02 for emotion in EMOTIONS}
            scores[primary] = base_conf
            secondary = palette[(idx + 3) % len(palette)]
            scores[secondary] += min(0.25, 1.0 - base_conf - 0.08)
            total = sum(scores.values())
            normalized = {k: v / total for k, v in scores.items()}
            ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)

            emotions = [
                EmotionScore(emotion=EmotionName(l), confidence=max(0.0, min(1.0, p)), label_cn=LABEL_CN.get(l))
                for l, p in ranked[:3]
            ]
            neg_score = sum(normalized.get(e, 0.0) for e in NEGATIVE_EMOTIONS)

            detections.append(FaceDetection(
                track_id=track_label,
                anonymous_id=anon_id,
                bbox=[fx, fy, fw, fh],
                face_confidence=0.95 - i * 0.01,
                emotions=emotions,
                primary_emotion=EmotionName(primary),
                primary_confidence=max(0.0, min(1.0, base_conf)),
                negative_score=max(0.0, min(1.0, neg_score)),
            ))

        return detections

    def _predict_replay(self, filename: str | None = None, image_size: tuple[int, int] | None = None) -> InferenceResponse:
        rows = self._load_replay_rows()
        if not rows:
            ranked = [("neutral", 0.72), ("happy", 0.18), ("sad", 0.10)]
            return self._build_response_from_ranked(ranked, mode="replay", source=filename or "replay-empty", image_size=image_size)

        row = rows[self._replay_index % len(rows)]
        self._replay_index += 1
        ranked: list[tuple[str, float]] = []
        for i in range(1, 4):
            label = row.get(f"top{i}_label") or row.get("pred_label")
            prob = float(row.get(f"top{i}_prob") or (row.get("pred_conf") or 0.0))
            if label:
                ranked.append((label, prob))
        if not ranked:
            ranked = [(row.get("pred_label", "neutral"), float(row.get("pred_conf", 0.7)))]

        return self._build_response_from_ranked(
            ranked,
            mode="replay",
            source=filename or row.get("filename") or "replay",
            meta={
                "true_label": row.get("true_label"),
                "pred_label": row.get("pred_label"),
                "pred_conf": row.get("pred_conf"),
                "source_path": row.get("path"),
            },
            image_size=image_size,
        )

    def _filter_detections_by_local_face_verifier(self, detections: list[FaceDetection], image_bytes: bytes) -> list[FaceDetection]:
        """Apply local bbox verifier to detections returned by remote/cloud services.

        Cloud responses may include their own bboxes. If we trust those blindly, local
        MediaPipe/filter changes do nothing. This keeps bbox policy local and consistent.
        """
        try:
            import numpy as np
            with Image.open(BytesIO(image_bytes)) as img:
                rgb = np.array(img.convert("RGB"))
                w, h = img.size
            frame_area = max(1, w * h)
            kept: list[FaceDetection] = []
            for det in detections:
                if self._looks_like_face_region(rgb, [int(v) for v in det.bbox], frame_area):
                    kept.append(det)
            return kept
        except Exception:
            return detections

    async def _predict_cloud(self, image_bytes: bytes, filename: str | None = None) -> InferenceResponse:
        if not self.settings.cloud_inference_url:
            raise RuntimeError("EMOTION_CLOUD_INFERENCE_URL is required when EMOTION_INFERENCE_MODE=cloud")

        async with httpx.AsyncClient(timeout=self.settings.cloud_timeout_seconds) as client:
            files = {"file": (filename or "frame.jpg", image_bytes, "application/octet-stream")}
            response = await client.post(self.settings.cloud_inference_url, files=files)
            response.raise_for_status()
            payload = response.json()

        if "detections" in payload:
            raw_detections = [self._normalize_detection(d) for d in payload.get("detections", [])]
            detections = self._filter_detections_by_local_face_verifier(raw_detections, image_bytes)
            return InferenceResponse(
                source=filename or "upload",
                mode="cloud",
                num_faces=len(detections),
                detections=detections,
                meta={
                    "remote_payload_keys": sorted(payload.keys()),
                    "remote_detection_count": len(raw_detections),
                    "local_verified_detection_count": len(detections),
                },
            )

        ranked = self._rank_from_payload(payload)
        # If the cloud endpoint returns only emotion probabilities and no bbox, do not
        # fabricate a center box. Use local verified face boxes; if none, return no boxes.
        face_bboxes, img_size = self._detect_faces(image_bytes)
        if not face_bboxes:
            return InferenceResponse(
                source=filename or "upload",
                mode="cloud",
                num_faces=0,
                detections=[],
                meta={**payload, "bbox_source": "none", "local_verified_detection_count": 0},
            )
        detections = self._build_detections(face_bboxes, image_bytes, filename, img_size)
        ranked_norm = [(label, float(prob)) for label, prob in ranked if label in EMOTIONS]
        if ranked_norm:
            total = sum(prob for _, prob in ranked_norm)
            if total > 1.0:
                ranked_norm = [(label, prob / total) for label, prob in ranked_norm]
            primary_label, primary_conf = ranked_norm[0]
            emotions = [
                EmotionScore(emotion=EmotionName(label), confidence=max(0.0, min(1.0, prob)), label_cn=LABEL_CN.get(label))
                for label, prob in ranked_norm[:3]
            ]
            score_map = {label: prob for label, prob in ranked_norm}
            negative_score = sum(score_map.get(label, 0.0) for label in NEGATIVE_EMOTIONS)
            for d in detections:
                d.primary_emotion = EmotionName(primary_label)
                d.primary_confidence = max(0.0, min(1.0, primary_conf))
                d.emotions = emotions
                d.negative_score = max(0.0, min(1.0, negative_score))
        return InferenceResponse(
            source=filename or "upload",
            mode="cloud",
            num_faces=len(detections),
            detections=detections,
            meta={**payload, "bbox_source": "local_verified_mediapipe", "local_verified_detection_count": len(detections)},
        )

    def _load_replay_rows(self) -> list[dict[str, str]]:
        if self._replay_rows is not None:
            return self._replay_rows

        path = self.settings.replay_csv_path
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            self._replay_rows = []
            return self._replay_rows

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            self._replay_rows = list(csv.DictReader(f))
        return self._replay_rows

    def _build_response_from_ranked(
        self,
        ranked: list[tuple[str, float]],
        mode: str,
        source: str,
        meta: dict[str, Any] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> InferenceResponse:
        ranked = [(label, float(prob)) for label, prob in ranked if label in EMOTIONS]
        if not ranked:
            ranked = [("neutral", 1.0)]
        total = sum(prob for _, prob in ranked)
        if total <= 0:
            ranked = [(ranked[0][0], 1.0)]
        elif total > 1.0:
            ranked = [(label, prob / total) for label, prob in ranked]

        primary_label, primary_conf = ranked[0]
        emotions = [
            EmotionScore(emotion=EmotionName(label), confidence=max(0.0, min(1.0, prob)), label_cn=LABEL_CN.get(label))
            for label, prob in ranked[:3]
        ]
        score_map = {label: prob for label, prob in ranked}
        negative_score = sum(score_map.get(label, 0.0) for label in NEGATIVE_EMOTIONS)

        if image_size:
            image_w, image_h = image_size
            bbox = [
                int(image_w * 0.25),
                int(image_h * 0.14),
                int(image_w * 0.50),
                int(image_h * 0.68),
            ]
        else:
            image_w, image_h = 360, 240
            bbox = [90, 34, 180, 164]

        detection = FaceDetection(
            track_id="person-1",
            bbox=bbox,
            face_confidence=0.98,
            emotions=emotions,
            primary_emotion=EmotionName(primary_label),
            primary_confidence=max(0.0, min(1.0, primary_conf)),
            negative_score=max(0.0, min(1.0, negative_score)),
        )
        response_meta = dict(meta or {})
        response_meta.setdefault("bbox_source", "demo_estimated_region")
        response_meta.setdefault("image_size", [image_w, image_h])
        response_meta.setdefault("image_width", image_w)
        response_meta.setdefault("image_height", image_h)
        return InferenceResponse(
            source=source,
            mode=mode,
            num_faces=1,
            detections=[detection],
            meta=response_meta,
        )

    def _normalize_detection(self, payload: dict[str, Any]) -> FaceDetection:
        emotions_payload = payload.get("emotions") or []
        emotions = []
        for e in emotions_payload[:3]:
            label = e.get("emotion") or e.get("label")
            if label in EMOTIONS:
                emotions.append(EmotionScore(
                    emotion=EmotionName(label),
                    confidence=float(e.get("confidence", e.get("prob", 0.0))),
                    label_cn=e.get("label_cn") or LABEL_CN.get(label),
                ))
        if not emotions:
            label = payload.get("primary_emotion", "neutral")
            if label not in EMOTIONS:
                label = "neutral"
            emotions = [EmotionScore(emotion=EmotionName(label), confidence=float(payload.get("primary_confidence", 1.0)), label_cn=LABEL_CN.get(label))]
        primary = emotions[0]
        negative_score = float(payload.get("negative_score", 0.0))
        if negative_score <= 0:
            negative_score = sum(e.confidence for e in emotions if e.emotion.value in NEGATIVE_EMOTIONS)
        return FaceDetection(
            track_id=str(payload.get("track_id", "person-1")),
            bbox=list(payload.get("bbox", [0, 0, 0, 0])),
            face_confidence=float(payload.get("face_confidence", 1.0)),
            emotions=emotions,
            primary_emotion=primary.emotion,
            primary_confidence=primary.confidence,
            negative_score=max(0.0, min(1.0, negative_score)),
        )

    def _rank_from_payload(self, payload: dict[str, Any]) -> list[tuple[str, float]]:
        if "probabilities" in payload and isinstance(payload["probabilities"], dict):
            return sorted(payload["probabilities"].items(), key=lambda item: float(item[1]), reverse=True)
        label = payload.get("emotion") or payload.get("primary_emotion") or "neutral"
        conf = float(payload.get("confidence") or payload.get("primary_confidence") or 1.0)
        return [(label, conf)]

    def _resolve_track_identity(
        self, track_id: str, embeddings: list, request_id: str | None = None
    ) -> dict | None:
        """Decide the anonymous identity of one video track from its embeddings.

        Returns a TrackIdentityDecision-shaped dict, or None when re-ID is
        disabled / no usable descriptor was extracted.
        """
        try:
            from app.services.identity_service import aggregate_embeddings

            emb = aggregate_embeddings(list(embeddings or []))
            if emb is None:
                return None
            service = self._identity()
            if not service.enabled():
                return None
            decision = service.decide(emb)
            n_samples = max(1, len(list(embeddings or [])))
            out: dict = {
                "track_id": track_id,
                "anon_code": "",
                "child_id": None,
                "child_nickname": None,
                "decision": decision.get("decision", "new"),
                "similarity": decision.get("top1_sim"),
                "candidate_anon_code": None,
                "candidate_nickname": None,
                "margin": decision.get("margin"),
                "sample_count": n_samples,
                "pending_id": None,
            }
            kind = out["decision"]
            if kind == "match":
                anon_code = decision["top1_anon"]
                service.touch(anon_code, emb)
                child_id, nickname = service.child_display(anon_code)
                out.update({"anon_code": anon_code, "child_id": child_id, "child_nickname": nickname})
            elif kind == "new":
                row = service.register_new(emb)
                out["anon_code"] = row["anon_code"]
            else:  # pending: temporary fresh identity + one track-level confirmation row
                row = service.register_new(emb)
                out["anon_code"] = row["anon_code"]
                out["candidate_anon_code"] = decision.get("top1_anon")
                candidate_nickname = None
                if decision.get("top1_anon"):
                    candidate = db.get_identity(decision["top1_anon"])
                    if candidate and candidate.get("child_id"):
                        child = db.get_child(candidate["child_id"])
                        candidate_nickname = (child or {}).get("nickname")
                out["candidate_nickname"] = candidate_nickname
                pending_id = str(uuid.uuid4())
                db.insert_pending_confirmation(
                    pending_id=pending_id,
                    anon_code=out["anon_code"],
                    candidate_anon_code=out["candidate_anon_code"],
                    candidate_nickname=candidate_nickname,
                    similarity=float(out["similarity"] or 0.0),
                    margin=float(out["margin"] or 0.0),
                    sample_count=n_samples,
                    event_request_id=request_id,
                    track_label=track_id,
                )
                out["pending_id"] = pending_id
            return out
        except Exception as exc:
            # Identity resolution must never break emotion inference.
            print(f"Identity resolution error: {type(exc).__name__}: {str(exc)[:160]}")
            return None

    def _apply_ranked_emotions_to_detection(self, det: FaceDetection, ranked: list[tuple[str, float]]) -> FaceDetection:
        """Overwrite a detection's mock emotion with real ranked model probabilities."""
        ranked_norm = [(label, float(prob)) for label, prob in ranked if label in EMOTIONS]
        if not ranked_norm:
            return det
        total = sum(prob for _, prob in ranked_norm)
        if total <= 0:
            ranked_norm = [(ranked_norm[0][0], 1.0)]
        elif total > 1.0:
            ranked_norm = [(label, prob / total) for label, prob in ranked_norm]
        ranked_norm.sort(key=lambda item: item[1], reverse=True)
        primary_label, primary_conf = ranked_norm[0]
        emotions = [
            EmotionScore(emotion=EmotionName(label), confidence=max(0.0, min(1.0, prob)), label_cn=LABEL_CN.get(label))
            for label, prob in ranked_norm[:3]
        ]
        score_map = {label: prob for label, prob in ranked_norm}
        negative_score = sum(score_map.get(label, 0.0) for label in NEGATIVE_EMOTIONS)
        det.primary_emotion = EmotionName(primary_label)
        det.primary_confidence = max(0.0, min(1.0, primary_conf))
        det.emotions = emotions
        det.negative_score = max(0.0, min(1.0, negative_score))
        return det

    def _crop_face_jpeg_from_frame(self, frame, bbox: list[int], expand: float = 0.22) -> bytes | None:
        """Crop a square, face-centered region from an OpenCV BGR frame.

        The emotion ViT is trained on square face-like images. Sending rectangular crops
        with varying hair/body/background ratio hurts classification stability, so use a
        square crop centered slightly above the bbox center with modest context.
        """
        try:
            import cv2
            h, w = frame.shape[:2]
            x, y, bw, bh = [int(v) for v in bbox]
            cx = x + bw / 2
            # Shift center a little upward to keep eyes/nose/mouth central and reduce clothing.
            cy = y + bh / 2 - bh * 0.06
            side = max(bw, bh) * (1.0 + expand * 2.0)
            x1 = int(round(cx - side / 2))
            y1 = int(round(cy - side / 2))
            x2 = int(round(cx + side / 2))
            y2 = int(round(cy + side / 2))

            # Pad with reflected border when face is near image edge, instead of making crop non-square.
            pad_l = max(0, -x1)
            pad_t = max(0, -y1)
            pad_r = max(0, x2 - w)
            pad_b = max(0, y2 - h)
            if pad_l or pad_t or pad_r or pad_b:
                frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
                x1 += pad_l
                x2 += pad_l
                y1 += pad_t
                y2 += pad_t

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if not ok:
                return None
            return buf.tobytes()
        except Exception:
            return None

    async def _classify_face_crop_cloud(self, client: httpx.AsyncClient, crop_bytes: bytes, filename: str | None = None) -> list[tuple[str, float]]:
        """Classify a cropped face with the cloud endpoint and return ranked probabilities."""
        if not self.settings.cloud_inference_url:
            return []
        files = {"file": (filename or "face_crop.jpg", crop_bytes, "image/jpeg")}
        response = await client.post(self.settings.cloud_inference_url, files=files)
        response.raise_for_status()
        payload = response.json()
        if "detections" in payload and payload.get("detections"):
            det = self._normalize_detection(payload["detections"][0])
            return [(e.emotion.value if hasattr(e.emotion, "value") else str(e.emotion), float(e.confidence)) for e in det.emotions]
        return self._rank_from_payload(payload)

    def _get_local_emotion_model(self):
        """Lazy-load local ViT emotion model as fallback when cloud endpoint is unavailable."""
        if self._local_emotion_model is not None and self._local_emotion_transform is not None:
            return self._local_emotion_model, self._local_emotion_transform
        import torch
        from torchvision import transforms as T
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        model_dir = Path(__file__).resolve().parents[3] / "models"
        model_path = model_dir / "vit_b16_age5_plus_age10_calibrated_best.pth"
        if not model_path.exists():
            model_path = model_dir / "vit_b16_age5_plus_age10_fixed_best.pth"
        if not model_path.exists():
            model_path = model_dir / "vit_b16_age5_best.pth"
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(EMOTIONS))
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._local_emotion_model = model
        self._local_emotion_transform = transform
        print(f"Loaded local emotion model: {model_path}")
        return model, transform

    def _load_video_label_priors(self) -> dict[str, dict[str, Any]]:
        """Load user-labeled per-video priors from suggested_calibration_labels.csv.

        This is intentionally video-scoped: it only affects videos whose debug hash appears
        in the label file, so it won't degrade unrelated normal videos.
        """
        if self._video_label_priors is not None:
            return self._video_label_priors
        import collections
        path = Path.cwd() / "data" / "video_debug_crops_review" / "suggested_calibration_labels.csv"
        fixes = {"digust": "disgust", "disguist": "disgust", "neytral": "neutral", "netural": "neutral"}
        priors: dict[str, dict[str, Any]] = {}
        if not path.exists():
            self._video_label_priors = priors
            return priors
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    vid = (row.get("video_id") or "").strip()
                    label = (row.get("true_label") or "").strip().lower()
                    label = fixes.get(label, label)
                    if not vid or label not in EMOTIONS:
                        continue
                    frame = int(float(row.get("frame_index") or 0))
                    info = priors.setdefault(vid, {"counts": collections.Counter(), "frames": []})
                    info["counts"][label] += 1
                    info["frames"].append((frame, label))
            for vid, info in priors.items():
                total = sum(info["counts"].values()) or 1
                info["distribution"] = {emo: info["counts"].get(emo, 0) / total for emo in EMOTIONS}
                info["total"] = total
            print(f"Loaded video label priors for {len(priors)} videos: {path}")
        except Exception as exc:
            print(f"Video prior load error: {type(exc).__name__}: {str(exc)[:160]}")
        self._video_label_priors = priors
        return priors

    def _apply_video_prior_to_ranked(self, ranked: list[tuple[str, float]], video_id: str, frame_idx: int) -> list[tuple[str, float]]:
        # Disabled: per-video priors only help labeled videos and do not improve global
        # generalization, so they are inappropriate for the final system.
        return ranked
        priors = self._load_video_label_priors()
        info = priors.get(video_id)
        if not info:
            return ranked
        probs = {label: float(prob) for label, prob in ranked if label in EMOTIONS}
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        # Stronger local prior if a labeled sample is temporally near the current frame.
        nearest = None
        for f, label in info.get("frames", []):
            dist = abs(int(frame_idx) - int(f))
            if nearest is None or dist < nearest[0]:
                nearest = (dist, label)
        prior_dist = dict(info.get("distribution") or {})
        alpha = 0.0
        if nearest and nearest[0] <= 24:
            # About 0.8s at 30fps / several sampled frames: likely the same expression segment.
            prior_dist = {emo: 0.0 for emo in EMOTIONS}
            prior_dist[nearest[1]] = 1.0
            alpha = 0.55
        else:
            dominant_label, dominant_ratio = max(prior_dist.items(), key=lambda kv: kv[1]) if prior_dist else ("", 0.0)
            if info.get("total", 0) >= 8 and dominant_ratio >= 0.62:
                alpha = 0.22
            elif info.get("total", 0) >= 12:
                alpha = 0.10
        if alpha <= 0:
            return ranked
        blended = [(emo, (1 - alpha) * probs.get(emo, 0.0) + alpha * prior_dist.get(emo, 0.0)) for emo in EMOTIONS]
        blended.sort(key=lambda item: item[1], reverse=True)
        return blended

    def _get_emotion_calibrator(self):
        """Lazy-load optional correction layer trained from real video debug crops."""
        if self._emotion_calibrator is not None:
            return self._emotion_calibrator, self._emotion_calibrator_schema
        path = Path.cwd() / "data" / "video_debug_crops_review" / "emotion_calibrator.joblib"
        if not path.exists():
            return None, None
        try:
            import joblib
            obj = joblib.load(path)
            self._emotion_calibrator = obj.get("model")
            self._emotion_calibrator_schema = obj.get("feature_schema")
            print(f"Loaded emotion calibrator: {path}")
            return self._emotion_calibrator, self._emotion_calibrator_schema
        except Exception as exc:
            print(f"Emotion calibrator load error: {type(exc).__name__}: {str(exc)[:160]}")
            return None, None

    def _calibrate_ranked_emotions(self, ranked: list[tuple[str, float]], visual_cues: dict[str, Any] | None = None) -> list[tuple[str, float]]:
        """Apply optional correction layer to raw model probabilities."""
        # Disabled by default: the current calibration set is dominated by failure cases,
        # which improved those clips but degraded normal happy/neutral videos. Keep the
        # code for later once we add balanced normal samples, but do not apply it now.
        return ranked
        model, _schema = self._get_emotion_calibrator()
        if model is None:
            return ranked
        try:
            probs = {label: float(prob) for label, prob in ranked if label in EMOTIONS}
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            visual_cues = visual_cues or {}
            cue_keys = ['skin_ratio','dark_mouth_ratio','saturated_ratio','upper_dark_ratio','mouth_open','lower_face_occluded','cry_like']
            pred = max(probs.items(), key=lambda kv: kv[1])[0] if probs else "neutral"
            sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            top1 = sorted_probs[0][1] if sorted_probs else 0.0
            top2 = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
            top_gap = top1 - top2
            negative_mass = sum(probs.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
            mouth_open = bool(visual_cues.get("mouth_open"))
            lower_face_occluded = bool(visual_cues.get("lower_face_occluded"))
            cry_like = bool(visual_cues.get("cry_like"))
            surprise_prob = probs.get("surprise", 0.0)

            # Gate the calibrator to avoid harming normal cases. The calibration set is small
            # and intentionally biased toward failure videos, so apply it only to suspicious
            # patterns observed in those failures.
            suspicious = (
                (pred == "happy" and (negative_mass > 0.18 or lower_face_occluded or cry_like))
                or (pred == "anger" and (mouth_open or surprise_prob > 0.08))
                or (pred == "neutral" and negative_mass > 0.35)
                or (top_gap < 0.10 and negative_mass > 0.22)
            )
            if not suspicious:
                return ranked

            x = [float(probs.get(e, 0.0)) for e in EMOTIONS]
            x += [float(visual_cues.get(k, 0.0)) for k in cue_keys]
            x += [1.0 if pred == e else 0.0 for e in EMOTIONS]
            pred_probs = model.predict_proba([x])[0]
            classes = list(model.classes_)
            calibrated = {emo: 0.0 for emo in EMOTIONS}
            for cls, prob in zip(classes, pred_probs):
                if cls in calibrated:
                    calibrated[cls] = float(prob)
            # Blend with original probabilities to avoid overfitting the small calibration set.
            alpha = 0.45
            blended = [(emo, alpha * calibrated.get(emo, 0.0) + (1 - alpha) * probs.get(emo, 0.0)) for emo in EMOTIONS]
            blended.sort(key=lambda item: item[1], reverse=True)
            return blended
        except Exception as exc:
            print(f"Emotion calibration error: {type(exc).__name__}: {str(exc)[:160]}")
            return ranked

    def _extract_face_crop_cues(self, crop_bytes: bytes) -> dict[str, float | bool]:
        """Extract cheap visual cues to correct systematic emotion confusions.

        These are not a replacement for the model; they only act as conservative priors for
        common failure cases: hand-over-mouth disgust, crying/fear faces misread as happy,
        and open-mouth surprise misread as anger.
        """
        try:
            import cv2
            import numpy as np
            img = Image.open(BytesIO(crop_bytes)).convert("RGB").resize((224, 224))
            rgb = np.array(img)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            # Lower central face: mouth/nose/hand occlusion zone.
            lower = hsv[112:196, 56:168]
            lower_ycc = ycc[112:196, 56:168]
            H, S, V = lower[:, :, 0], lower[:, :, 1], lower[:, :, 2]
            Y, Cr, Cb = lower_ycc[:, :, 0], lower_ycc[:, :, 1], lower_ycc[:, :, 2]
            skin = (((H <= 25) | (H >= 170)) & (S >= 18) & (S <= 180) & (V >= 45)) | ((Cr >= 125) & (Cr <= 180) & (Cb >= 80) & (Cb <= 140) & (Y >= 35))
            skin_ratio = float(np.mean(skin))
            dark_mouth_ratio = float(np.mean((V < 82) & (S > 35)))
            saturated_ratio = float(np.mean((S > 150) & (V > 80)))
            # Upper eye/brow region: closed/crying eyes often produce dark horizontal strokes.
            upper = hsv[58:112, 48:176]
            upper_dark_ratio = float(np.mean((upper[:, :, 2] < 95) & (upper[:, :, 1] > 25)))
            mouth_open = dark_mouth_ratio > 0.075
            lower_face_occluded = skin_ratio > 0.48 and dark_mouth_ratio < 0.045
            cry_like = upper_dark_ratio > 0.16 and dark_mouth_ratio > 0.045
            return {
                "skin_ratio": skin_ratio,
                "dark_mouth_ratio": dark_mouth_ratio,
                "saturated_ratio": saturated_ratio,
                "upper_dark_ratio": upper_dark_ratio,
                "mouth_open": bool(mouth_open),
                "lower_face_occluded": bool(lower_face_occluded),
                "cry_like": bool(cry_like),
            }
        except Exception:
            return {}

    def _classify_face_crop_local(self, crop_bytes: bytes) -> list[tuple[str, float]]:
        """Classify a cropped face locally on CPU as a fallback."""
        ranked, _emb = self._classify_and_embed_face_crop_local(crop_bytes)
        return ranked

    def _classify_and_embed_face_crop_local(
        self, crop_bytes: bytes
    ) -> tuple[list[tuple[str, float]], "np.ndarray | None"]:
        """Classify a face crop with the local ViT AND extract the 768-dim
        normalised CLS descriptor (pre-classifier) for anonymous re-ID.

        The identity descriptor comes from ``settings.reid_backbone``:
        ``imagenet`` uses a separate emotion-agnostic ViT (more stable across
        expressions), ``emotion`` reuses the emotion ViT's CLS (single pass).
        """
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            model, transform = self._get_local_emotion_model()
            img = Image.open(BytesIO(crop_bytes)).convert("RGB")
            x = transform(img).unsqueeze(0)
            with torch.no_grad():
                x = model._process_input(x)
                cls = model.class_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = model.encoder(x)
                feat = F.normalize(x[:, 0], dim=1)
                logits = model.heads(x[:, 0])
                probs = F.softmax(logits, dim=1)[0]
            ranked = [(EMOTIONS[i], float(probs[i])) for i in range(len(EMOTIONS))]
            ranked.sort(key=lambda item: item[1], reverse=True)
            if (getattr(self.settings, "reid_backbone", "emotion") or "emotion") == "imagenet":
                emb = self._embed_face_crop_reid(crop_bytes)
            else:
                emb = feat[0].cpu().numpy().astype(np.float32)
            return ranked, emb
        except Exception as exc:
            print(f"Local crop classify/embed error: {type(exc).__name__}: {str(exc)[:180]}")
            return [], None

    def _get_reid_model(self):
        """Lazy ImageNet-pretrained ViT-B/16 used only for identity descriptors."""
        if self._reid_model is not None or self._reid_backbone_failed:
            return self._reid_model
        try:
            import torch
            from torchvision import transforms as T
            from torchvision.models import ViT_B_16_Weights, vit_b_16
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            model.eval()
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._reid_model = model
            self._reid_transform = transform
            print("Loaded ImageNet ViT for identity descriptors")
        except Exception as exc:
            self._reid_backbone_failed = True
            print(f"Re-ID ImageNet backbone unavailable ({type(exc).__name__}: {str(exc)[:140]}); falling back to emotion CLS")
        return self._reid_model

    def _embed_face_crop_reid(self, crop_bytes: bytes) -> "np.ndarray | None":
        """768-dim normalised descriptor from the emotion-agnostic ViT."""
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            model = self._get_reid_model()
            if model is None:
                return None
            img = Image.open(BytesIO(crop_bytes)).convert("RGB")
            x = self._reid_transform(img).unsqueeze(0)
            with torch.no_grad():
                x = model._process_input(x)
                cls = model.class_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = model.encoder(x)
                feat = F.normalize(x[:, 0], dim=1)
            return feat[0].cpu().numpy().astype(np.float32)
        except Exception as exc:
            print(f"Re-ID embed error: {type(exc).__name__}: {str(exc)[:140]}")
            return None

    def _track_and_build_detections(
        self,
        face_bboxes: list[list[int]],
        image_bytes: bytes,
        filename: str | None,
        image_size: tuple[int, int],
        track_map: dict[str, dict[str, Any]],
        track_prev_emo: dict[str, str | None],
        track_counter: int,
        completed_decisions: dict[str, dict] | None = None,
        completed_thumbs: dict[str, bytes] | None = None,
    ) -> tuple[list[FaceDetection], dict[str, dict[str, Any]], dict[str, str | None], int]:
        """Track faces and emit bbox every sampled frame.

        Emotion is not recomputed for display here every frame. Raw per-frame emotion
        samples are buffered, while the displayed emotion is a lock that predict_video
        refreshes every ~1.5s. This keeps boxes moving while labels update discretely.

        ``completed_decisions`` receives the re-ID decision of any track that dies
        here (left the frame), so the caller can still persist it at video end.
        ``completed_thumbs`` likewise receives the best face crop of a departed
        track for the session-only claim thumbnail buffer.
        """
        import copy

        def _emo_value(v) -> str:
            return v.value if hasattr(v, "value") else str(v)

        def _center(box: list[int]) -> tuple[float, float]:
            x, y, bw, bh = box
            return x + bw / 2, y + bh / 2

        def _iou(a: list[int], b: list[int]) -> float:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0

        def _predict_bbox(track: dict[str, Any]) -> list[int]:
            box = list(track.get("bbox", [0, 0, 0, 0]))
            vx, vy = track.get("velocity", [0.0, 0.0])
            x, y, bw, bh = box
            img_w, img_h = image_size
            nx = int(max(0, min(max(0, img_w - bw), x + vx)))
            ny = int(max(0, min(max(0, img_h - bh), y + vy)))
            return [nx, ny, int(bw), int(bh)]

        def _clone_with_bbox(det: FaceDetection, bbox: list[int], face_conf: float) -> FaceDetection:
            data = det.model_dump()
            data["bbox"] = [int(v) for v in bbox]
            data["face_confidence"] = max(0.0, min(1.0, face_conf))
            return FaceDetection(**data)

        def _apply_display_emotion(det: FaceDetection, tid: str) -> FaceDetection:
            display_state = getattr(self, "_current_track_display_emotion", {})
            state = display_state.get(tid)
            if not state:
                return det
            emo = state.get("emotion")
            conf = float(state.get("confidence", det.primary_confidence))
            emos = state.get("emotions")
            neg = float(state.get("negative_score", det.negative_score))
            if emo in EMOTIONS:
                det.primary_emotion = EmotionName(emo)
                det.primary_confidence = max(0.0, min(1.0, conf))
                if emos:
                    det.emotions = [EmotionScore(**e.model_dump()) if hasattr(e, "model_dump") else EmotionScore(**e) for e in emos]
                else:
                    det.emotions = [EmotionScore(emotion=EmotionName(emo), confidence=det.primary_confidence, label_cn=LABEL_CN.get(emo))]
                det.negative_score = max(0.0, min(1.0, neg))
            return det

        # Normalize legacy track format if present.
        normalized_tracks: dict[str, dict[str, Any]] = {}
        for tid, v in track_map.items():
            if isinstance(v, dict):
                normalized_tracks[tid] = v
            elif isinstance(v, list) and len(v) >= 4:
                cx, cy, hits, misses = v[:4]
                normalized_tracks[tid] = {
                    "center": [float(cx), float(cy)],
                    "bbox": [int(cx - 40), int(cy - 40), 80, 80],
                    "velocity": [0.0, 0.0],
                    "hits": int(hits),
                    "misses": int(misses),
                    "last_det": None,
                }
        track_map = normalized_tracks

        detections: list[FaceDetection] = []
        matched_tracks: set[str] = set()
        assignments: list[tuple[str, list[int]]] = []
        sorted_faces = sorted(face_bboxes, key=lambda b: b[0])

        # Match detections to existing tracks.
        for raw_box in sorted_faces:
            box = [int(v) for v in raw_box]
            cx, cy = _center(box)
            best_id = None
            best_score = -1e9

            for tid, track in track_map.items():
                if tid in matched_tracks:
                    continue
                pred_box = _predict_bbox(track)
                pcx, pcy = _center(pred_box)
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                max_dist = max(80.0, image_size[0] * 0.15, box[2] * 0.8) # STricter matching distance
                
                size_ratio = (box[2] * box[3]) / max(1, pred_box[2] * pred_box[3])
                if size_ratio < 0.6 or size_ratio > 1.8: # Stricter size change limit
                    continue
                
                if dist > max_dist and _iou(box, pred_box) < 0.15: # Stricter IoU requirement if far
                    continue
                    
                score = _iou(box, pred_box) * 3.0 - dist / max_dist
                if score > best_score:
                    best_score = score
                    best_id = tid

            if best_id is None:
                track_counter += 1
                best_id = f"person-{track_counter}"
                track_prev_emo[best_id] = None
                track_map[best_id] = {
                    "bbox": box,
                    "center": [cx, cy],
                    "velocity": [0.0, 0.0],
                    "hits": 0,
                    "misses": 0,
                    "last_det": None,
                }
            else:
                old = track_map[best_id]
                old_cx, old_cy = old.get("center", [cx, cy])
                old_vx, old_vy = old.get("velocity", [0.0, 0.0])
                new_vx = old_vx * 0.55 + (cx - old_cx) * 0.45
                new_vy = old_vy * 0.55 + (cy - old_cy) * 0.45
                old.update({"bbox": box, "center": [cx, cy], "velocity": [new_vx, new_vy]})

            track_map[best_id]["hits"] = int(track_map[best_id].get("hits", 0)) + 1
            track_map[best_id]["misses"] = 0
            matched_tracks.add(best_id)
            assignments.append((best_id, box))

        # Only confirmed tracks are rendered. Balanced setting: ~0.8s at 5fps before showing
        # a new box. This keeps most one-frame false positives out while reducing missed faces.
        confirmed_assignments = [(tid, box) for tid, box in assignments if int(track_map[tid].get("hits", 0)) >= 4]
        
        # Also filter out boxes that are entirely in the extreme corners/edges if they are new tracks
        filtered_assignments = []
        w, h = image_size
        for tid, box in confirmed_assignments:
             x, y, bw, bh = box
             hits = int(track_map[tid].get("hits", 0))
             
             # If it's a relatively new track (hits < 10) and it's tiny and in the background, drop it
             if hits < 10:
                 if bw * bh < w * h * 0.02: # Less than 2% of screen
                     continue
                 
             filtered_assignments.append((tid, box))
             
        assigned_ids = [tid for tid, _ in filtered_assignments]
        assigned_bboxes = [box for _, box in filtered_assignments]
        
        prev_list = [track_prev_emo.get(tid) for tid in assigned_ids]
        built = self._build_detections(assigned_bboxes, image_bytes, filename, image_size, prev_emotions=prev_list)

        raw_buffer = getattr(self, "_current_track_raw_emotion_buffer", None)
        for tid, det in zip(assigned_ids, built):
            det.track_id = tid
            raw_emo = _emo_value(det.primary_emotion)
            if raw_buffer is not None:
                raw_buffer[tid].append({
                    "emotion": raw_emo,
                    "confidence": float(det.primary_confidence),
                    "emotions": [EmotionScore(**e.model_dump()) for e in det.emotions],
                    "negative_score": float(det.negative_score),
                    "face_confidence": float(det.face_confidence),
                })

            det = _apply_display_emotion(det, tid)
            hits = int(track_map[tid].get("hits", 0))
            det.face_confidence = min(0.98, 0.55 + hits * 0.08)
            track_prev_emo[tid] = _emo_value(det.primary_emotion)
            track_map[tid]["last_det"] = copy.deepcopy(det)
            detections.append(det)

        # Short hold for missed tracks. Keep it short to avoid stale boxes remaining
        # after a person leaves the frame.
        for tid, track in list(track_map.items()):
            if tid in matched_tracks:
                continue
            misses = int(track.get("misses", 0)) + 1
            track["misses"] = misses
            track["hits"] = max(0, int(track.get("hits", 0)) - 1)

            if misses > 2:
                # Preserve the re-ID decision and best thumbnail of a departed
                # track so the caller can persist them at video end.
                dec = track.get("identity_decision")
                if dec is not None and completed_decisions is not None:
                    completed_decisions[tid] = dec
                thumb = track.get("thumb")
                if thumb and completed_thumbs is not None:
                    completed_thumbs[tid] = thumb
                track_map.pop(tid, None)
                track_prev_emo.pop(tid, None)
                continue
                
            # Don't hold unconfirmed tracks
            if int(track.get("hits", 0)) < 4:
                 continue

            last_det = track.get("last_det")
            if isinstance(last_det, FaceDetection):
                pred_box = _predict_bbox(track)
                pcx, pcy = _center(pred_box)
                track["bbox"] = pred_box
                track["center"] = [pcx, pcy]
                det = _clone_with_bbox(last_det, pred_box, last_det.face_confidence * (0.55 ** misses))
                det = _apply_display_emotion(det, tid)
                detections.append(det)

        detections.sort(key=lambda d: d.bbox[0])
        return detections, track_map, track_prev_emo, track_counter
    async def predict_video(self, video_bytes: bytes, filename: str | None = None) -> VideoInferenceResponse:
        import cv2
        import os
        import tempfile
        from collections import Counter, defaultdict

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename or "video.mp4").suffix)
        tmp.write(video_bytes)
        tmp_path = tmp.name
        tmp.close()

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / max(fps, 1)

        mode = self.settings.inference_mode.lower()
        frames_out: list[FrameResult] = []
        all_primary: list[str] = []
        frame_idx = 0
        processed = 0
        raw_bbox_frames = 0
        rendered_bbox_frames = 0
        max_raw_bboxes = 0
        max_rendered_bboxes = 0

        track_map: dict[str, dict[str, Any]] = {}
        track_counter = 0
        track_prev_emo: dict[str, str | None] = {}
        request_id = str(uuid.uuid4())
        completed_decisions: dict[str, dict] = {}
        completed_thumbs: dict[str, bytes] = {}

        # BBox is emitted every sampled frame; emotion label is refreshed every 1.5s.
        sample_step = max(1, int(fps / 5))
        emotion_window_sec = 1.5
        emotion_window_start = 0.0
        self._current_track_raw_emotion_buffer = defaultdict(list)
        self._current_track_display_emotion: dict[str, dict[str, Any]] = {}
        cloud_client: httpx.AsyncClient | None = None
        cloud_crop_calls = 0
        cloud_crop_errors = 0
        cloud_last_error = ""
        local_crop_calls = 0
        local_crop_errors = 0
        cloud_classify_interval_sec = 0.6
        debug_export_dir: Path | None = None
        debug_export_count = 0
        debug_name = hashlib.sha256(video_bytes[:1024] + (filename or "").encode("utf-8")).hexdigest()[:10]
        if mode == "cloud":
            debug_export_dir = Path.cwd() / "data" / "video_debug_crops" / debug_name
            debug_export_dir.mkdir(parents=True, exist_ok=True)

        def _aggregate_emotion_window() -> None:
            raw_window = getattr(self, "_current_track_raw_emotion_buffer", None)
            if raw_window is None:
                return
            for tid, samples in list(raw_window.items()):
                if not samples:
                    continue
                scores: dict[str, float] = defaultdict(float)
                sample_by_emo: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for sample in samples:
                    face_weight = max(0.35, min(1.0, float(sample.get("face_confidence", 0.75))))
                    ranked_probs = sample.get("ranked") or []
                    if ranked_probs:
                        # Use the full probability distribution instead of only the top-1 label.
                        # This is more stable for close classes such as fear/surprise/sad.
                        for label, prob in ranked_probs:
                            if label in EMOTIONS:
                                p = max(0.0, min(1.0, float(prob)))
                                scores[label] += max(0.001, p) * face_weight
                        top_label = ranked_probs[0][0] if ranked_probs and ranked_probs[0][0] in EMOTIONS else None
                        if top_label:
                            sample_by_emo[top_label].append(sample)
                        continue

                    emo = sample.get("emotion")
                    if emo not in EMOTIONS:
                        continue
                    conf = float(sample.get("confidence", 0.0))
                    scores[emo] += max(0.01, conf) * face_weight
                    sample_by_emo[emo].append(sample)
                if not scores:
                    continue
                total_score = sum(scores.values()) or 1.0
                normalized_scores = {emo: val / total_score for emo, val in scores.items()}

                # Visual-cue hard correction is disabled. It helped a few failure clips but
                # also falsely suppressed genuine happy videos. We still export cues for future
                # balanced calibration, but do not alter probabilities here.
                ranked = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
                best_emo, best_norm_score = ranked[0]
                best_score = best_norm_score * total_score

                # Low-confidence suppression: diagnostic/demo-balanced mode.
                # If the fused top-1 emotion is negative, let it pass through. The previous
                # neutral/previous fallback was masking real negative expressions.
                top1_score = normalized_scores.get(best_emo, 0.0)
                top2_score = normalized_scores.get(ranked[1][0], 0.0) if len(ranked) > 1 else 0.0
                top_gap = top1_score - top2_score
                negative_mass = sum(normalized_scores.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
                neutral_mass = normalized_scores.get("neutral", 0.0)
                happy_mass = normalized_scores.get("happy", 0.0)
                positive_mass = neutral_mass + happy_mass
                low_confidence = best_emo not in NEGATIVE_EMOTIONS and (top1_score < 0.26 or top_gap < 0.02)
                weak_negative = False

                # Hysteresis: avoid label flicker, but do not block transitions into negative
                # emotions. For negative top-1 predictions the switch margin is zero.
                prev_state = self._current_track_display_emotion.get(tid, {})
                prev_emo = prev_state.get("emotion")
                switch_margin = 0.0 if best_emo in NEGATIVE_EMOTIONS else 0.06

                if low_confidence and weak_negative:
                    if prev_emo in EMOTIONS and prev_emo not in NEGATIVE_EMOTIONS:
                        best_emo = prev_emo
                    else:
                        best_emo = "neutral"
                    best_score = normalized_scores.get(best_emo, 0.0) * total_score
                elif prev_emo in EMOTIONS and prev_emo != best_emo:
                    prev_score = normalized_scores.get(prev_emo, 0.0)
                    if normalized_scores.get(best_emo, 0.0) <= prev_score + switch_margin:
                        best_emo = prev_emo
                        best_score = scores.get(best_emo, prev_score * total_score)

                best_samples = sample_by_emo.get(best_emo) or samples
                best_conf = min(1.0, max(0.0, best_score / total_score))
                # Prefer the most confident sample's top-3 distribution as display detail.
                best_sample = max(best_samples, key=lambda s: float(s.get("confidence", 0.0)))
                self._current_track_display_emotion[tid] = {
                    "emotion": best_emo,
                    "confidence": best_conf,
                    "emotions": best_sample.get("emotions") or [EmotionScore(emotion=EmotionName(best_emo), confidence=best_conf, label_cn=LABEL_CN.get(best_emo))],
                    "negative_score": max(float(s.get("negative_score", 0.0)) for s in best_samples),
                    "switch_margin": switch_margin,
                    "low_confidence_suppressed": bool(low_confidence and weak_negative),
                    "top_gap": float(top_gap),
                    "negative_mass": float(negative_mass),
                }
            raw_window.clear()

        try:
            if mode == "cloud" and self.settings.cloud_inference_url:
                cloud_client = httpx.AsyncClient(timeout=self.settings.cloud_timeout_seconds)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % sample_step != 0:
                    continue

                processed += 1
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                image_bytes = buffer.tobytes()
                image_size = (frame.shape[1], frame.shape[0])
                ts = frame_idx / max(fps, 1)

                # Face detection + tracking is evaluated every sampled frame.
                face_bboxes, _ = self._detect_faces(image_bytes)
                if face_bboxes:
                    raw_bbox_frames += 1
                    max_raw_bboxes = max(max_raw_bboxes, len(face_bboxes))
                dets, track_map, track_prev_emo, track_counter = self._track_and_build_detections(
                    face_bboxes,
                    image_bytes,
                    filename,
                    image_size,
                    track_map,
                    track_prev_emo,
                    track_counter,
                    completed_decisions,
                    completed_thumbs,
                )

                use_cloud_classifier = mode == "cloud" and cloud_client is not None
                if dets and (use_cloud_classifier or mode == "local"):
                    # Classify confirmed face crops with the real emotion model
                    # (cloud endpoint, or the on-device ViT in local mode).
                    # BBox remains local/realtime; emotion samples are refreshed at a lower rate
                    # and cached per track to avoid excessive calls.
                    for d in dets:
                        track = track_map.get(d.track_id, {})
                        last_ts = float(track.get("last_cloud_emotion_ts", -999.0))
                        ranked = track.get("last_cloud_ranked")
                        if ts - last_ts >= cloud_classify_interval_sec:
                            crop_bytes = self._crop_face_jpeg_from_frame(frame, d.bbox)
                            if crop_bytes:
                                # Keep the best-quality crop of this track as the
                                # session-only claim thumbnail (RAM buffer, claim-to-burn).
                                face_conf = float(getattr(d, "face_confidence", 0.0))
                                if "thumb" not in track or face_conf >= float(track.get("thumb_conf", -1.0)):
                                    track["thumb"] = crop_bytes
                                    track["thumb_conf"] = face_conf
                                visual_cues = self._extract_face_crop_cues(crop_bytes)
                                if visual_cues:
                                    track["last_visual_cues"] = visual_cues
                                if mode == "local":
                                    local_crop_calls += 1
                                    try:
                                        ranked, crop_emb = self._classify_and_embed_face_crop_local(crop_bytes)
                                    except Exception as exc:
                                        local_crop_errors += 1
                                        ranked, crop_emb = None, None
                                        print(f"Local crop classify error: {type(exc).__name__}: {str(exc)[:180]}")
                                    # Collect a few spaced-out descriptors per track for re-ID.
                                    if crop_emb is not None:
                                        emb_list = track.setdefault("embeddings", [])
                                        last_emb_ts = float(track.get("last_embed_ts", -999.0))
                                        limit = int(getattr(self.settings, "reid_track_sample_limit", 8))
                                        if len(emb_list) < limit and ts - last_emb_ts >= 0.75:
                                            emb_list.append(crop_emb)
                                            track["last_embed_ts"] = ts
                                            # Decide identity once a track has a few
                                            # descriptors; alerts from this point on
                                            # carry the final pseudonym.
                                            if "identity_decision" not in track and len(emb_list) >= 3:
                                                track["identity_decision"] = self._resolve_track_identity(
                                                    d.track_id, emb_list, request_id
                                                )
                                else:
                                    cloud_crop_calls += 1
                                    try:
                                        ranked = await self._classify_face_crop_cloud(cloud_client, crop_bytes, f"{filename or 'video'}-{d.track_id}-{frame_idx}.jpg")
                                    except Exception as exc:
                                        cloud_crop_errors += 1
                                        cloud_last_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                                        print(f"Cloud crop classify error: {cloud_last_error}")
                                        # Cloud endpoint may be offline/404. Fall back to local ViT so video
                                        # emotion is still real model output instead of mock/hash labels.
                                        try:
                                            ranked = self._classify_face_crop_local(crop_bytes)
                                            local_crop_calls += 1
                                        except Exception:
                                            local_crop_errors += 1
                                            ranked = None
                                if ranked:
                                    visual_cues_for_cal = track.get("last_visual_cues") or {}
                                    ranked = self._calibrate_ranked_emotions(ranked, visual_cues_for_cal)
                                    ranked = self._apply_video_prior_to_ranked(ranked, debug_name, frame_idx)
                                    track["last_cloud_emotion_ts"] = ts
                                    track["last_cloud_ranked"] = ranked
                        if ranked:
                            self._apply_ranked_emotions_to_detection(d, ranked)
                            track_prev_emo[d.track_id] = d.primary_emotion.value if hasattr(d.primary_emotion, "value") else str(d.primary_emotion)
                            # Carry the resolved pseudonym onto the live detection
                            # once this track's identity has been decided.
                            dec = track.get("identity_decision")
                            if dec:
                                d.anonymous_id = dec["anon_code"]
                            raw_buffer = getattr(self, "_current_track_raw_emotion_buffer", None)
                            if raw_buffer is not None and raw_buffer.get(d.track_id):
                                raw_buffer[d.track_id][-1] = {
                                    "emotion": track_prev_emo[d.track_id],
                                    "confidence": float(d.primary_confidence),
                                    "emotions": [EmotionScore(**e.model_dump()) for e in d.emotions],
                                    "negative_score": float(d.negative_score),
                                    "face_confidence": float(d.face_confidence),
                                    "ranked": [(label, float(prob)) for label, prob in ranked if label in EMOTIONS],
                                    "visual_cues": track.get("last_visual_cues") or {},
                                }
                            if debug_export_dir is not None and crop_bytes and debug_export_count < 300:
                                stem = f"f{frame_idx:06d}_{d.track_id}_{track_prev_emo[d.track_id]}_{float(d.primary_confidence):.3f}"
                                img_path = debug_export_dir / f"{stem}.jpg"
                                meta_path = debug_export_dir / f"{stem}.json"
                                try:
                                    img_path.write_bytes(crop_bytes)
                                    meta_path.write_text(json.dumps({
                                        "frame_index": frame_idx,
                                        "timestamp_sec": ts,
                                        "track_id": d.track_id,
                                        "bbox": d.bbox,
                                        "primary_emotion": track_prev_emo[d.track_id],
                                        "primary_confidence": float(d.primary_confidence),
                                        "negative_score": float(d.negative_score),
                                        "ranked": [(label, float(prob)) for label, prob in ranked if label in EMOTIONS],
                                        "visual_cues": track.get("last_visual_cues") or {},
                                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                                    debug_export_count += 1
                                except Exception as exc:
                                    print(f"Debug crop export error: {type(exc).__name__}: {str(exc)[:120]}")

                if mode == "replay" and dets:
                    base = self._predict_replay(filename, image_size=image_size)
                    if base.detections:
                        base_det = base.detections[0]
                        for d in dets:
                            d.emotions = [EmotionScore(**e.model_dump()) for e in base_det.emotions]
                            d.primary_emotion = base_det.primary_emotion
                            d.primary_confidence = base_det.primary_confidence
                            d.negative_score = base_det.negative_score

                # Refresh display emotion every 1.5s, but keep emitting bbox every sampled frame.
                if processed == 1 or ts - emotion_window_start >= emotion_window_sec:
                    _aggregate_emotion_window()
                    emotion_window_start = ts
                    # Re-apply newly locked emotion to current detections before output.
                    locked = self._current_track_display_emotion
                    for d in dets:
                        state = locked.get(d.track_id)
                        if state and state.get("emotion") in EMOTIONS:
                            emo = state["emotion"]
                            d.primary_emotion = EmotionName(emo)
                            d.primary_confidence = max(0.0, min(1.0, float(state.get("confidence", d.primary_confidence))))
                            d.negative_score = max(0.0, min(1.0, float(state.get("negative_score", d.negative_score))))
                            emos = state.get("emotions")
                            if emos:
                                d.emotions = [EmotionScore(**e.model_dump()) if hasattr(e, "model_dump") else EmotionScore(**e) for e in emos]

                if dets:
                    rendered_bbox_frames += 1
                    max_rendered_bboxes = max(max_rendered_bboxes, len(dets))
                fr = FrameResult(frame_index=frame_idx, timestamp_sec=round(ts, 2), detections=dets)
                frames_out.append(fr)
                for d in dets:
                    all_primary.append(d.primary_emotion.value if hasattr(d.primary_emotion, "value") else d.primary_emotion)
        finally:
            cap.release()
            os.unlink(tmp_path)
            # Keep attributes present after a run; deleting them can race with async/error paths
            # and cause AttributeError on the next access. Reset instead.
            self._current_track_raw_emotion_buffer = defaultdict(list)
            self._current_track_display_emotion = {}
            if cloud_client is not None:
                await cloud_client.aclose()

        counts = Counter(all_primary)
        total_detected = len(all_primary)
        agg = {
            "total_frames": total,
            "processed_frames": processed,
            "detected_faces": total_detected,
            "dominant_emotion": counts.most_common(1)[0][0] if counts else "none",
            "emotion_distribution": {k: {"count": v, "pct": round(v / max(total_detected, 1) * 100, 1)} for k, v in counts.most_common()},
            "smoothing": {
                "bbox_output": "per_sampled_frame",
                "emotion_window_sec": emotion_window_sec,
                "sample_step": sample_step,
                "emotion_output": "locked_every_1.5s",
                "bbox_output": "per_sampled_frame",
                "missed_face_hold_frames": 2,
                "bbox_source": "local_mediapipe_verified_tracker",
                "raw_bbox_frames": raw_bbox_frames,
                "rendered_bbox_frames": rendered_bbox_frames,
                "max_raw_bboxes": max_raw_bboxes,
                "max_rendered_bboxes": max_rendered_bboxes,
                "verifier": "skin_texture_saturation_v3_balanced",
                "emotion_source": ("cloud_face_crop_with_local_fallback" if mode == "cloud"
                                   else ("local_vit_b16" if mode == "local" else mode)),
                "cloud_crop_calls": cloud_crop_calls,
                "cloud_crop_errors": cloud_crop_errors,
                "cloud_last_error": cloud_last_error,
                "local_crop_calls": local_crop_calls,
                "local_crop_errors": local_crop_errors,
                "cloud_classify_interval_sec": cloud_classify_interval_sec,
                "emotion_switch_margin": "0_when_switching_to_negative_else_0.06",
                "low_confidence_policy": "negative_top1_pass_through_non_negative_only_top1_lt_0.26_or_gap_lt_0.02",
                "emotion_fusion": "full_probability_weighted_by_face_confidence",
                "crop_strategy": "square_face_centered_224_v2",
                "visual_cue_correction": "disabled_export_only",
                "emotion_calibrator": "disabled_pending_balanced_labels",
                "video_prior": "disabled_not_generalizable",
                "debug_export_dir": str(debug_export_dir) if debug_export_dir else "",
                "debug_export_count": debug_export_count,
            },
        }

        if debug_export_dir is not None:
            try:
                final_summary = {
                    "filename": filename or "",
                    "mode": mode,
                    "duration_sec": round(duration, 1),
                    "processed_frames": processed,
                    "aggregate": agg,
                    "frame_emotions": [
                        {
                            "frame_index": fr.frame_index,
                            "timestamp_sec": fr.timestamp_sec,
                            "detections": [
                                {
                                    "track_id": d.track_id,
                                    "primary_emotion": d.primary_emotion.value if hasattr(d.primary_emotion, "value") else str(d.primary_emotion),
                                    "primary_confidence": float(d.primary_confidence),
                                    "negative_score": float(d.negative_score),
                                    "bbox": d.bbox,
                                    "top_emotions": [
                                        {
                                            "emotion": e.emotion.value if hasattr(e.emotion, "value") else str(e.emotion),
                                            "confidence": float(e.confidence),
                                        }
                                        for e in d.emotions
                                    ],
                                }
                                for d in fr.detections
                            ],
                        }
                        for fr in frames_out
                    ],
                }
                (debug_export_dir / "final_response_summary.json").write_text(
                    json.dumps(final_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"Final video summary export error: {type(exc).__name__}: {str(exc)[:160]}")

        # Anonymous re-ID: one decision per physical face track (not per frame).
        # Departed tracks keep their decision in completed_decisions; surviving
        # tracks resolve here (fallback when too few descriptors were seen).
        track_identity_decisions = []
        for tid, decision in completed_decisions.items():
            track_identity_decisions.append(decision)
        for tid, track in track_map.items():
            decision = track.get("identity_decision")
            if decision is None:
                decision = self._resolve_track_identity(tid, track.get("embeddings") or [], request_id)
            if decision:
                track_identity_decisions.append(decision)
        if track_identity_decisions:
            decision_by_track = {d["track_id"]: d for d in track_identity_decisions}
            for fr in frames_out:
                for d in fr.detections:
                    dec = decision_by_track.get(d.track_id)
                    if dec:
                        d.anonymous_id = dec["anon_code"]

        # Register session-only claim thumbnails (RAM buffer, claim-to-burn).
        thumbs = {tid: track.get("thumb") for tid, track in track_map.items() if track.get("thumb")}
        thumbs.update(completed_thumbs)
        if thumbs and track_identity_decisions:
            decision_by_track = {d["track_id"]: d for d in track_identity_decisions}
            try:
                from app.thumb_buffer import thumb_buffer
                for tid, jpeg in thumbs.items():
                    dec = decision_by_track.get(tid)
                    if dec and jpeg:
                        thumb_buffer.put(request_id, tid, dec["anon_code"], jpeg)
            except Exception as exc:
                print(f"Thumb buffer registration error: {type(exc).__name__}: {str(exc)[:120]}")

        return VideoInferenceResponse(
            request_id=request_id,
            filename=filename or "",
            mode=mode,
            fps=float(fps),
            total_frames=total,
            processed_frames=processed,
            duration_sec=round(duration, 1),
            frames=frames_out,
            aggregate=agg,
            track_identity_decisions=track_identity_decisions,
        )














