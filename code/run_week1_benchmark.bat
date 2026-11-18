@echo off
cd /d "%~dp0"
echo [1/2] Dang chay khao sat 5 bien the YOLO tren 1-shot va 5-shot (3 seeds)...
python step2_screen_models.py ^
  --config-dir ..\runs\configs ^
  --runs-dir ..\runs\models ^
  --results-dir ..\results ^
  --overlap-json ..\results\class_overlap.json ^
  --k-shots 1 5 ^
  --seeds 0 1 2

echo [2/2] Dang xuat 5 anh du doan tot nhat / 5 anh sai va kiem tra xuat ONNX cho Orange Pi 5 Pro...
python step4_cameraless_trap_demo.py ^
  --weights ..\runs\models\yolo11s_spatial_attn_k5_seed0\weights\last.pt ^
  --test-images ..\runs\processed_data\test\images ^
  --out-dir ..\results\qualitative_analysis ^
  --export-edge

echo [HOAN TAT] Xem ket qua tai thu muc: ..\results\model_stability_report.md
pause
