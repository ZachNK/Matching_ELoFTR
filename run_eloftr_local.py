# /workspace/run_eloftr_local.py
from __future__ import annotations
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from PIL import Image, UnidentifiedImageError
import torch, numpy as np, matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time, sys, re, json

# ======================== 설정 =========================
START_DIR = Path("/workspace/_datasets")  # 시작 탐색 루트
SAVE_DIR  = Path("/workspace/matchAnything_eloftr/SAVED/")
RESULTS_DIR = SAVE_DIR / "results"
LOGS_DIR    = SAVE_DIR / "logs"
for d in (RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.2
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# ======================================================

def now(): return time.perf_counter()
def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXT

def open_image_strict(p: Path) -> Image.Image:
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.load()
            return im.copy()
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"[ERROR] 이미지가 아님 또는 손상됨: {p} ({e})")

def print_help():
    print(
        "명령어:\n"
        "  ls                 현재 폴더 내용 표시 (번호 부여)\n"
        "  ls images          현재 폴더의 이미지 파일만 표시 (번호 부여)\n"
        "  cd <번호|#번호>    직전 ls에서 표시된 번호의 디렉터리로 이동\n"
        "  cd..               상위 폴더로 이동\n"
        "  pick <번호|#번호>  직전 ls에서 표시된 번호의 이미지 파일 선택\n"
        "  pwd                현재 경로 표시\n"
        "  help               이 도움말\n"
        "  quit               종료\n"
    )

def _scan_dir(cur: Path, images_only: bool = False) -> list[Path]:
    try:
        items = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        print("[ERR] 권한이 없습니다.")
        return []
    if images_only:
        items = [p for p in items if is_image_file(p)]
    return items

def list_dir(cur: Path, images_only: bool, last_listing: dict):
    items = _scan_dir(cur, images_only=images_only)
    if not items:
        print("(비어있음)")
        last_listing.clear()
        return
    for idx, p in enumerate(items, start=1):
        tag = "[D]" if p.is_dir() else "[F]"
        print(f"[#{idx:>2}] {tag} {p.name}")
    last_listing["dir"] = cur
    last_listing["images_only"] = images_only
    last_listing["items"] = items

def _parse_index_token(token: str) -> int|None:
    m = re.fullmatch(r'#?(\d+)', token.strip())
    return int(m.group(1)) if m else None

def _take_from_listing_by_index(idx: int, last_listing: dict) -> Path|None:
    if not last_listing or "items" not in last_listing:
        print("[ERR] 먼저 ls로 목록을 띄운 뒤 번호를 선택하세요.")
        return None
    items = last_listing.get("items", [])
    if not (1 <= idx <= len(items)):
        print(f"[ERR] 번호 범위(1~{len(items)})를 벗어났습니다.")
        return None
    return items[idx - 1]

def navigator_pick(base: Path) -> Path:
    cur = base.resolve()
    start_root = base.resolve()
    print(f"\n파일 선택을 시작합니다. 시작 경로: {cur}")
    print_help()

    last_listing: dict = {}

    while True:
        try:
            cmd = input(f"(pick) {cur}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n사용자 중단.")
            sys.exit(1)

        if not cmd:
            continue
        parts = cmd.split()
        op = parts[0].lower()

        if op == "ls":
            images_only = len(parts) > 1 and parts[1].lower() == "images"
            list_dir(cur, images_only=images_only, last_listing=last_listing)

        elif op == "pwd":
            print(cur)

        elif op == "cd..":
            parent = cur.parent
            if parent.as_posix().startswith(start_root.as_posix()) and parent != cur:
                cur = parent
                last_listing.clear()
            else:
                print("[WARN] 시작 경로보다 위로는 이동할 수 없습니다.")

        elif op == "cd":
            if len(parts) < 2:
                print("[ERR] 사용법: cd <번호|#번호>")
                continue
            idx = _parse_index_token(parts[1])
            if idx is None:
                print("[ERR] cd는 번호만 허용합니다. (예: cd 1)")
                continue
            target = _take_from_listing_by_index(idx, last_listing)
            if target is None:
                continue
            if not target.is_dir():
                print("[ERR] 디렉토리가 아닙니다. (번호가 파일을 가리킴)")
                continue
            if not target.as_posix().startswith(start_root.as_posix()):
                print("[ERR] 시작 루트 밖으로는 이동할 수 없습니다.")
                continue
            cur = target.resolve()
            last_listing.clear()

        elif op == "pick":
            if len(parts) < 2:
                print("[ERR] 사용법: pick <번호|#번호>")
                continue
            idx = _parse_index_token(parts[1])
            if idx is None:
                print("[ERR] pick은 번호만 허용합니다. (예: pick 2)")
                continue
            target = _take_from_listing_by_index(idx, last_listing)
            if target is None:
                continue
            if not target.exists():
                print("[ERR] 파일이 존재하지 않습니다."); continue
            if not target.is_file():
                print("[ERR] 파일이 아닙니다. (번호가 디렉터리를 가리킨 것 같아요)"); continue
            if not is_image_file(target):
                print(f"[ERR] 이미지 파일 아님: {target.name} (허용 확장자: {sorted(VALID_EXT)})"); continue
            try:
                _ = open_image_strict(target)
            except RuntimeError as e:
                print(e); print("[FATAL] 이미지 파일이 아닙니다. 프로그램을 종료합니다.")
                sys.exit(1)
            print(f"[OK] 선택됨: {target}")
            return target

        elif op == "help":
            print_help()

        elif op == "quit":
            print("종료합니다.")
            sys.exit(0)

        else:
            print("[ERR] 알 수 없는 명령. 'help'를 입력하세요.")

def main():
    # 1) 이미지 두 장 선택
    print("=== 첫 번째 이미지 선택 ===")
    p1 = navigator_pick(START_DIR)
    print("\n=== 두 번째 이미지 선택 ===")
    p2 = navigator_pick(START_DIR)

    timings = {}
    t0 = now()
    img1 = open_image_strict(p1)
    img2 = open_image_strict(p2)
    images = [img1, img2]
    timings["load_images_s"] = now() - t0

    # 2) 모델/프로세서
    t1 = now()
    processor = AutoImageProcessor.from_pretrained("zju-community/matchanything_eloftr", use_fast=True)
    model = AutoModelForKeypointMatching.from_pretrained("zju-community/matchanything_eloftr")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    timings["load_proc_model_s"] = now() - t1

    # 3) 전처리
    t2 = now()
    inputs = processor(images, return_tensors="pt").to(device)
    sync_if_cuda()
    timings["preprocess_s"] = now() - t2

    # 4) 추론
    t3 = now()
    with torch.no_grad():
        outputs = model(**inputs)
    sync_if_cuda()
    timings["inference_s"] = now() - t3

    # 5) 후처리
    t4 = now()
    image_sizes = [[(im.height, im.width) for im in images]]
    pp_all = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=THRESHOLD)
    sync_if_cuda()
    timings["postprocess_s"] = now() - t4

    # 매칭 값 출력(옵션)
    for i, output in enumerate(pp_all):
        print(f"For the image pair {i}")
        k0_cpu = output["keypoints0"].detach().cpu()
        k1_cpu = output["keypoints1"].detach().cpu()
        s_cpu  = output["matching_scores"].detach().cpu()
        for keypoint0, keypoint1, matching_score in zip(k0_cpu, k1_cpu, s_cpu):
            print(
                f"Keypoint at {keypoint0.numpy()}  <->  {keypoint1.numpy()}  score={float(matching_score.item())}"
            )

    # 시각화용 numpy
    pp = pp_all[0]
    k0 = pp["keypoints0"].detach().cpu().numpy()   # (N,2)
    k1 = pp["keypoints1"].detach().cpu().numpy()   # (N,2)
    scores = pp["matching_scores"].detach().cpu().numpy()  # (N,)

    # 6) 결과 이미지 저장 (results/)
    t5 = now()
    w1, h1 = img1.width, img1.height
    w2, h2 = img2.width, img2.height
    H, W = max(h1, h2), w1 + w2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = np.array(img1.convert("RGB"))
    canvas[:h2, w1:w1+w2] = np.array(img2.convert("RGB"))
    k1_shift = k1.copy(); k1_shift[:, 0] += w1

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas); plt.axis("off")
    for (x0, y0), (x1, y1), sc in zip(k0, k1_shift, scores):
        plt.scatter([x0, x1], [y0, y1], s=10)
        plt.plot([x0, x1], [y0, y1], linewidth=max(0.5, 2.0 * float(sc)))
    tstamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    png_path = RESULTS_DIR / f"match_vis_{tstamp}.png"
    plt.tight_layout(); plt.savefig(png_path.as_posix(), dpi=150); plt.close()
    timings["visualize_save_s"] = now() - t5

    # 7) 기하 재구성 (RANSAC): F, H, rectification H1/H2 + 인라이어 수
    # pts는 float32로
    pts1 = k0.astype(np.float32)
    pts2 = k1.astype(np.float32)

    # Fundamental matrix with RANSAC
    F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.999)
    num_inliers_F = int(maskF.sum()) if maskF is not None else 0

    # Homography with RANSAC
    H, maskH = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)
    # 스테레오 비보정 정렬용 H1/H2 (가능할 때만)
    H1 = H2 = None
    try:
        if F is not None and maskF is not None and num_inliers_F >= 8:
            in1 = pts1[maskF.ravel() == 1]
            in2 = pts2[maskF.ravel() == 1]
            ok, H1_tmp, H2_tmp = cv2.stereoRectifyUncalibrated(in1, in2, F, imgSize=(w1, h1))
            if ok:
                H1, H2 = H1_tmp, H2_tmp
    except Exception:
        H1 = H2 = None

    # 8) 로그(JSON) 저장 (logs/)
    matches_stats = {
        "num_raw_matches": int(len(scores)),
        "num_ransac_matches": num_inliers_F  # Fundamental RANSAC 인라이어 수
    }

    match_info = {
        "match_conf": {
            "output": "matches-eloftr",
            "model": {
                "name": "eloftr",
                "model_name": "zju-community/matchanything_eloftr",
                "max_keypoints": None,         # 알 수 없어 None
                "match_threshold": THRESHOLD
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": None,            # 프로세서 내부 리사이즈 미상
                "dfactor": None,
                "width": w1,                   # 첫 번째 이미지 기준
                "height": h1,
                "force_resize": False
            },
            "max_error": 1,    # RANSAC reproj 임계(픽셀) 의미로 기록
            "cell_size": 1
        },
        "extractor_conf": None
    }

    def _to_list(x):
        return x.tolist() if x is not None else None

    geom_info = {
        "Fundamental": _to_list(F) if F is not None else None,
        "Homography":  _to_list(H) if H is not None else None,
        "H1":          _to_list(H1) if H1 is not None else None,
        "H2":          _to_list(H2) if H2 is not None else None
    }

    log_obj = {
        "matches_statistics": matches_stats,
        "match_info": match_info,
        "geom_info": geom_info,
        "paths": {
            "img1": p1.as_posix(),
            "img2": p2.as_posix(),
            "result_png": png_path.as_posix()
        },
        "timings": {
            **{k: float(v) for k, v in timings.items()},
            "total_pipeline_s": float(sum(timings.values()))
        },
        "device": device
    }

    json_path = LOGS_DIR / f"match_log_{tstamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_obj, f, ensure_ascii=False, indent=2)

    # 9) 콘솔 요약
    summary = {
        "device": device,
        "threshold": THRESHOLD,
        "img1": p1.as_posix(),
        "img2": p2.as_posix(),
        **matches_stats,
        **timings,
        "total_pipeline_s": sum(timings.values()),
        "png_path": png_path.as_posix(),
        "json_log": json_path.as_posix()
    }
    print("\n=== Summary ===")
    for k in ["device","threshold","img1","img2","num_raw_matches","num_ransac_matches",
              "load_images_s","load_proc_model_s","preprocess_s","inference_s","postprocess_s","visualize_save_s","total_pipeline_s","png_path","json_log"]:
        print(f"{k}: {summary[k]}")

    print(f"\nSaved image: {png_path}")
    print(f"Saved json : {json_path}")

if __name__ == "__main__":
    if not START_DIR.exists() or not START_DIR.is_dir():
        print(f"[FATAL] START_DIR가 올바르지 않습니다: {START_DIR}")
        sys.exit(1)
    main()
