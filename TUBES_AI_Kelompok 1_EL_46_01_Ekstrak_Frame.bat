@echo off
setlocal enabledelayedexpansion

REM Folder output untuk setiap video
mkdir frames

REM Loop semua file .h264 di folder ini
for %%F in (*.h264) do (
    echo Memproses: %%F

    REM Buat subfolder berdasarkan nama file (tanpa ekstensi)
    set "filename=%%~nF"
    mkdir frames\!filename!

    REM Ekstrak 1 frame per detik
    ffmpeg -i "%%F" -vf "fps=5" "frames\!filename!\frame_%%04d.jpg"
)

echo Selesai semua file diproses.
pause
