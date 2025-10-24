<!-- 

테스트 자동 실행 + 결과 호스트에 생성

docker run --rm --gpus all --mount type=bind,source="$((Get-Location).Path)",target=/code -w /code  vincentqin/image-matching-webui:latest bash -lc 'set -e; pip install -q spaces; pip install -e .; python -m tests.test_basic; echo "--- done ---"; ls -al experiments || true' 

(선택) 웹 UI로 띄우기
docker run -it --gpus all -p 7860:7860 --mount type=bind,source="$((Get-Location).Path)",target=/code -w /code vincentqin/image-matching-webui:latest bash -lc 'pip install -q spaces; pip install -e .; python app.py --server_name 0.0.0.0 --server_port 7860'

→ 브라우저에서 http://localhost:7860 접속.

*** 디버깅 이슈
Successfully installed aiofiles-23.2.1 gradio-5.4.0 gradio-client-1.4.2 imcui-0.0.0 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-ml-py-13.580.82 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.9.86 nvidia-nvtx-cu12-12.1.105 pycolmap-0.6.1 pydantic-2.10.6 pydantic-core-2.27.2 pynvml-13.0.1 python-multipart-0.0.12 safetensors-0.6.2 tomlkit-0.12.0 torch-2.2.2 torchvision-0.17.2 triton-2.2.0 websockets-12.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
/opt/conda/lib/python3.11/site-packages/torch/cuda/__init__.py:54: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/code/tests/test_basic.py", line 110, in <module>
    test_one()
  File "/code/tests/test_basic.py", line 70, in test_one
    api = ImageMatchingAPI(conf=conf, device=DEVICE)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/code/imcui/api/core.py", line 57, in __init__
    self._init_models()
  File "/code/imcui/api/core.py", line 106, in _init_models
    self.extractor = get_feature_model(self.conf["feature"])
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/code/imcui/ui/utils.py", line 188, in get_feature_model
    Model = dynamic_load(extractors, conf["model"]["name"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/code/imcui/hloc/utils/base_model.py", line 48, in dynamic_load
    module = __import__(module_path, fromlist=[""])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/code/imcui/hloc/extractors/superpoint.py", line 11, in <module>
    from SuperGluePretrainedNetwork.models import superpoint  # noqa E402
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'SuperGluePretrainedNetwork'

에러가 나온다면

→ 필요한 서브모듈 설치(권장) 
A) 한줄로

docker run --rm --gpus all --mount type=bind,source="$((Get-Location).Path)",target=/code -w /code vincentqin/image-matching-webui:latest bash -lc 'set -e command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y git) mkdir -p third_party [ -d third_party/SuperGluePretrainedNetwork ] || git clone https://github.com/magicleap/SuperGluePretrainedNetwork third_party/SuperGluePretrainedNetwork pip install -q spaces pip install -e third_party/SuperGluePretrainedNetwork pip install -e . python -m tests.test_basic .

B) 순차적으로:SuperPoint 그대로 쓰는 방법(서브모듈 설치)

0) 0) (호스트) 서브모듈 가져오기
mkdir third_party

git clone https://github.com/magicleap/SuperGluePretrainedNetwork .\third_party\SuperGluePretrainedNetwork

1) 공통 마운트 경로 변수
$W = $((Get-Location).Path)

2) 컨테이너에서 필요한 패키지 설치 (짧게 나눠서)
docker run --rm --gpus all --mount type=bind,source="$W",target=/code -w /code vincentqin/image-matching-webui:latest bash -lc "pip install -q spaces"

docker run --rm --gpus all --mount type=bind,source="$W",target=/code -w /code vincentqin/image-matching-webui:latest bash -lc "pip install -e third_party/SuperGluePretrainedNetwork"

docker run --rm --gpus all --mount type=bind,source="$W",target=/code -w /code vincentqin/image-matching-webui:latest bash -lc "pip install -e ."

3) 테스트 실행
docker run --rm --gpus all --mount type=bind,source="$W",target=/code -w /code vincentqin/image-matching-webui:latest bash -lc "python -m tests.test_basic"

4) 결과 확인(호스트, PowerShell)
Get-ChildItem .\experiments -Recurse | Select-Object -First 30

-->
