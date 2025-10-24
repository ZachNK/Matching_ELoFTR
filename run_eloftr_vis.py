# /workspace/run_eloftr_vis.py
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from transformers.image_utils import load_image
import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import time, os, sys

# ---------------- cfg ----------------
SAVE_DIR = Path("/workspace/ma_eloftr/SAVED")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.2  # 매칭 점수 임계값

# ---------------- timers helpers ----------------
def now():
    return time.perf_counter()

def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

timings = {}

# ---------------- 1) load images ----------------
t0 = now()
img1 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg")
img2 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg")
images = [img1, img2]
timings["load_images_s"] = now() - t0

# ---------------- 2) 프로세서/모델 ----------------
t1 = now()
processor = AutoImageProcessor.from_pretrained("zju-community/matchanything_eloftr")
model = AutoModelForKeypointMatching.from_pretrained("zju-community/matchanything_eloftr")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
timings["load_proc_model_s"] = now() - t1

# ---------------- 3) 전처리 ----------------
t2 = now()
inputs = processor(images, return_tensors="pt").to(device)
sync_if_cuda()
timings["preprocess_s"] = now() - t2

# ---------------- 4) 추론 ----------------
with torch.no_grad():
    outputs = model(**inputs)
    outputs = model(**inputs)
    
t3 = now()
with torch.no_grad():
    outputs = model(**inputs)
sync_if_cuda()
timings["inference_s"] = now() - t3

# ---------------- 5) 후처리 (매칭 좌표/점수 출력) ----------------
t4 = now()
image_sizes = [[(im.height, im.width) for im in images]]
pp_all = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=THRESHOLD)
sync_if_cuda()
timings["postprocess_s"] = now() - t4

# 콘솔 출력: 프린트 전에 텐서를 CPU로 옮기기(.cpu())
for i, output in enumerate(pp_all):
    print(f"For the image pair {i}")
    k0_cpu = output["keypoints0"].detach().cpu()
    k1_cpu = output["keypoints1"].detach().cpu()
    s_cpu  = output["matching_scores"].detach().cpu()
    for keypoint0, keypoint1, matching_score in zip(k0_cpu, k1_cpu, s_cpu):
        print(
            f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with "
            f"keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {float(matching_score.item())}."
        )

# 첫 번째 페어를 그림으로 시각화
pp = pp_all[0]
# 시각화용도도 미리 CPU NumPy로 변환
k0 = pp["keypoints0"].detach().cpu().numpy()
k1 = pp["keypoints1"].detach().cpu().numpy()
scores = pp["matching_scores"].cpu().numpy()

# ---------------- 6) 시각화 ----------------
t5 = now()
w1, h1 = img1.width, img1.height
w2, h2 = img2.width, img2.height
H, W = max(h1, h2), w1 + w2
canvas = np.zeros((H, W, 3), dtype=np.uint8)
canvas[:h1, :w1] = np.array(img1.convert("RGB"))
canvas[:h2, w1:w1+w2] = np.array(img2.convert("RGB"))

k1_shift = k1.copy()
k1_shift[:, 0] += w1

plt.figure(figsize=(16, 8))
plt.imshow(canvas)
plt.axis("off")
for (x0, y0), (x1, y1), sc in zip(k0, k1_shift, scores):
    plt.scatter([x0, x1], [y0, y1], s=10)
    plt.plot([x0, x1], [y0, y1], linewidth=max(0.5, 2.0 * float(sc)))

tstamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
png_path = SAVE_DIR / f"match_vis_{tstamp}.png"
plt.tight_layout()
plt.savefig(png_path.as_posix(), dpi=150)
plt.close()
timings["visualize_save_s"] = now() - t5

# ---------------- 7) 요약 + 로그 ----------------
counts = {
    "num_matches": int(len(scores)),
    "num_kpts_img0": int(len(k0)),
    "num_kpts_img1": int(len(k1)),
}
summary = {
    "device": device,
    "threshold": THRESHOLD,
    "png_path": png_path.as_posix(),
    **counts,
    **timings,
    "total_pipeline_s": timings["load_images_s"]
                         + timings["load_proc_model_s"]
                         + timings["preprocess_s"]
                         + timings["inference_s"]
                         + timings["postprocess_s"]
                         + timings["visualize_save_s"],
}

# 콘솔 요약
print("\n=== Summary ===")
for k in ["device","threshold","num_matches","num_kpts_img0","num_kpts_img1",
          "load_images_s","load_proc_model_s","preprocess_s","inference_s","postprocess_s","visualize_save_s","total_pipeline_s","png_path"]:
    print(f"{k}: {summary[k]}")

# 로그 저장
log_path = SAVE_DIR / f"match_log_{tstamp}.log"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("MatchAnything-EloFTR run log\n")
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")
    f.write("\n-- Raw matches (keypoints & scores) --\n")
    for keypoint0, keypoint1, matching_score in zip(k0, k1, scores):
        f.write(f"kpt0 {keypoint0}  <->  kpt1 {keypoint1}  score={float(matching_score)}\n")

print(f"\nSaved image: {png_path}")
print(f"Saved log  : {log_path}")
