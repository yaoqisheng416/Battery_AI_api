# -*- coding: utf-8 -*-
import os
import shutil
import sys
import tempfile
import threading
import zipfile
import logging
from datetime import datetime
from http.client import HTTPException
from pathlib import Path

import uvicorn

from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse

from schemas import Stage4Request, cbdGenerateRequest, fitParameterRequest, largeVolumeGenerateRequest, \
    localConditionsGenerateRequest

from core.task_manager import (
    create_task,
    get_task, TASK_STORE,
)

from tasks.stage4_task import run_stage4_task
from tasks.stage5_task import run_large_volume_generate_task, run_local_conditions_generate_task
from tasks.stage6_task import run_stage6_cbd_fit_task, run_stage6_cbd_generate_task

logger = logging.getLogger("cbd_w_fitting_service")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

app = FastAPI()

# =========================================================
# cors
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return JSONResponse(
        content={
            "message": "Welcome to the BA-API!",
            "docs": "Visit /docs for API documentation",
            "available_endpoints": ["/tasks", "/task/{task_id}"]
        },
        status_code=200
    )


@app.get("/tasks")
def get_all_tasks():
    return list(
        TASK_STORE.values()
    )


# =========================================================
# Create Stage4 generate_structure_from_condition Task API
# =========================================================
@app.post("/stage4/generate_structure_from_condition")
def create_stage4_task(
        request: Stage4Request
):
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:
        task_id = create_task(title="Stage4 条件可控的两相结构生成")

    #  打印日志（调试用）
    print(f"[Stage4] 任务ID: {task_id}")
    print(f"[Stage4] VAE路径: {request.vae_path}")
    print(f"[Stage4] LDM路径: {request.ldm_path}")
    print(f"[Stage4] Porosity: {request.porosity}")
    print(f"[Stage4] Tau Z: {request.tau_z}")
    print(f"[Stage4] Surface Area: {request.surface_area}")

    thread = threading.Thread(
        target=run_stage4_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Query Task API
# ============================================================
@app.get("/task/query/{task_id}")
def query_task(task_id: str):
    task = get_task(task_id)

    if task is None:
        return {
            "status": "not_found"
        }

    return task


# ============================================================
# Create Stage5 build_large_volume_conditions_from_real Task API
# ============================================================
@app.post("/stage5/local-conditions-generate")
def create_stage5_task(
        request: localConditionsGenerateRequest
):

    # ============================================
    # task_id
    # ============================================
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(
            title="Stage5 从真实体积构建local conditions"
        )

    # ============================================
    # start thread
    # ============================================
    thread = threading.Thread(
        target=run_local_conditions_generate_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    # ============================================
    # response
    # ============================================
    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Create Stage5 large-volume-generate Task API
# ============================================================
@app.post("/stage5/large-volume-generate")
def create_stage5_task(
        request: largeVolumeGenerateRequest
):
    # ============================================
    # task_id 处理
    # ============================================
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(
            title="Stage5 224³大体积生成"
        )

    # ============================================
    # start thread
    # ============================================
    thread = threading.Thread(
        target=run_large_volume_generate_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    # ============================================
    # response
    # ============================================
    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Create Stage6 cbd-generate Task API
# ============================================================
@app.post("/stage6/cbd-generate")
def create_stage6_task(
        request: cbdGenerateRequest
):
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(

            title="Stage6 CBD三相电极结构生成"
        )

    thread = threading.Thread(
        target=run_stage6_cbd_generate_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Create Stage6 fit_cbd_spreading_parameter Task API
# ============================================================
@app.post("/stage6/fit-cbd-spreading-parameter")
def create_stage6_task(
        request: fitParameterRequest
):
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(

            title="Stage6 CBD参数拟合"
        )

    thread = threading.Thread(
        target=run_stage6_cbd_fit_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    return {
        "task_id": task_id,
        "status": "queued",
    }


# 模型选择
def get_base_dir():
    """智能判断：打包后用exe目录，本地用脚本目录"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)  # 打包后exe目录
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 本地模式: 从脚本位置向上找项目根目录


@app.get("/health")
def health(SERVER_READY=True):
    return {
        "ready": SERVER_READY
    }


@app.get("/models/versions")
def get_model_versions():
    BASE_DIR = get_base_dir()

    # 智能切换路径
    if getattr(sys, 'frozen', False):
        vae_dir = os.path.join(BASE_DIR, "checkpoints")
        ldm_dir = os.path.join(BASE_DIR, "ldm_checkpoints")
        print(f"[打包模式] VAE: {vae_dir}")
        print(f"[打包模式] LDM: {ldm_dir}")
    else:
        vae_dir = os.path.join(BASE_DIR, "checkpoints")
        ldm_dir = os.path.join(BASE_DIR, "ldm_checkpoints")
        print(f"[本地模式] VAE: {vae_dir}")
        print(f"[本地模式] LDM: {ldm_dir}")

    # 扫描 VAE
    vae_models = []
    if os.path.exists(vae_dir):
        for file_name in sorted(os.listdir(vae_dir), key=lambda x: os.path.getmtime(os.path.join(vae_dir, x)),
                                reverse=True):
            if file_name.endswith(".ckpt"):
                vae_models.append({
                    "file_name": file_name,
                    "full_path": os.path.join(vae_dir, file_name),
                    "create_time": datetime.fromtimestamp(os.path.getmtime(os.path.join(vae_dir, file_name))).strftime(
                        "%Y-%m-%d %H:%M:%S"),
                })
        print(f"[OK] 找到 {len(vae_models)} 个VAE模型")
    else:
        print(f"[警告] VAE目录不存在：{vae_dir}")

    # 扫描 LDM
    ldm_models = []
    if os.path.exists(ldm_dir):
        for file_name in sorted(os.listdir(ldm_dir), key=lambda x: os.path.getmtime(os.path.join(ldm_dir, x)),
                                reverse=True):
            if file_name.endswith(".ckpt"):
                ldm_models.append({
                    "file_name": file_name,
                    "full_path": os.path.join(ldm_dir, file_name),
                    "create_time": datetime.fromtimestamp(os.path.getmtime(os.path.join(ldm_dir, file_name))).strftime(
                        "%Y-%m-%d %H:%M:%S"),
                })
        print(f"[OK] 找到 {len(ldm_models)} 个LDM模型")
    else:
        print(f"[警告] LDM目录不存在：{ldm_dir}")

    return {
        "vae_models": vae_models,
        "ldm_models": ldm_models,
        "base_dir": BASE_DIR,
        "mode": "frozen" if getattr(sys, 'frozen', False) else "dev",
    }


# ========================================================
#  上传文件到API的/workspace/tasks目录
# ========================================================
@app.post("/upload/files")
async def upload_best_two_phase_structure(
        file: UploadFile = File(...),
        title: str = Form(...),
):
    """
    上传文件到 Workspace（支持单个文件或 ZIP 压缩包）
    """
    # ============================================
    # 创建任务
    # ============================================
    task_id = create_task(title=title)

    # ============================================
    # workspace
    # ============================================
    task_root = os.path.join("workspace", "tasks", task_id)
    input_dir = os.path.join(task_root, "input")
    os.makedirs(input_dir, exist_ok=True)

    # ============================================
    # 保存文件
    # ============================================
    save_path = os.path.join(input_dir, file.filename)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # ============================================
    # ✅ 如果是 ZIP，自动解压并扁平化
    # ============================================
    extracted_dir = None
    unzip_message = None

    if file.filename.endswith(".zip"):
        try:
            extracted_dir = os.path.join(input_dir, "extracted")
            os.makedirs(extracted_dir, exist_ok=True)

            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                # ✅ 获取 ZIP 内的
                file_list = zip_ref.namelist()

                # ✅ 关键：如果只有一个顶层目录，直接解压其内容
                if len(file_list) == 1 and file_list[0].endswith('/'):
                    # 只有一个目录，解压其内容到 extracted_dir
                    zip_ref.extractall(extracted_dir)

                    # ✅ 删除空的顶层目录，将内容移到上一层
                    top_dir = os.path.join(extracted_dir, os.path.basename(file_list[0].rstrip('/')))
                    if os.path.exists(top_dir):
                        for item in os.listdir(top_dir):
                            shutil.move(
                                os.path.join(top_dir, item),
                                extracted_dir
                            )
                        os.rmdir(top_dir)
                else:
                    # 多个文件或复杂结构，直接解压
                    zip_ref.extractall(extracted_dir)

                    # ✅ 关键：如果解压后只有一个子目录，将其内容移到上一层
                    subdirs = [d for d in os.listdir(extracted_dir)
                               if os.path.isdir(os.path.join(extracted_dir, d))]

                    if len(subdirs) == 1:
                        subdir = os.path.join(extracted_dir, subdirs[0])
                        for item in os.listdir(subdir):
                            shutil.move(
                                os.path.join(subdir, item),
                                extracted_dir
                            )
                        os.rmdir(subdir)

            # ✅ 更新 save_path 为扁平化后的目录
            save_path = extracted_dir
            unzip_message = f"✅ ZIP 已自动解压并扁平化到: {extracted_dir}"

        except zipfile.BadZipFile:
            unzip_message = "⚠️ ZIP 文件格式错误，已保存原始文件"
        except Exception as e:
            unzip_message = f"⚠️ 解压失败: {str(e)}，已保存原始 ZIP 文件"

    # ============================================
    # ✅ 返回完整信息
    # ============================================
    return {
        "success": True,
        "task_id": task_id,
        "input_file": save_path,  # ✅ 扁平化后的路径
        "title": title,
        "filename": file.filename,
        "file_type": "zip" if file.filename.endswith(".zip") else "single",
        "unzip_message": unzip_message,
        "extracted_dir": extracted_dir,
    }


@app.get("/download/{task_id}/{file_name}")
def download_file(

        task_id: str,

        file_name: str,
):
    file_path = os.path.join(

        "workspace",
        "tasks",
        task_id,
        "output",
        file_name,
    )

    if not os.path.exists(file_path):
        return {
            "error": "file not found"
        }

    return FileResponse(

        file_path,

        filename=file_name,
    )


# 获取处理结果展示
@app.get("/task/results/list")
def list_task_results(task_id: str):
    """列出任务输出目录下的所有文件和文件夹"""

    logger.info(f"📥 请求：task_id={task_id}")

    # ✅ 构建标准化路径
    output_dir = f"workspace/tasks/{task_id}/output"
    output_dir = output_dir.replace("\\", "/")

    # ✅ 转换为绝对路径
    full_path = Path.cwd() / output_dir
    full_path = full_path.resolve()

    logger.info(f"🔍 路径: {full_path}")
    logger.info(f"🔍 存在: {full_path.exists()}")

    # ✅ 检查路径
    if not full_path.exists():
        logger.warning(f"⚠️ 路径不存在: {full_path}")
        return {"files": []}

    if not full_path.is_dir():
        logger.warning(f"⚠️ 不是目录: {full_path}")
        return {"files": []}

    files = []

    try:
        # ✅ 遍历文件和文件夹
        for root, dirs, files_list in os.walk(full_path):

            # =============================================
            # ✅ 修复1：遍历文件（缩进正确）
            # =============================================
            for f in files_list:
                full_file_path = Path(root) / f
                rel_path = full_file_path.relative_to(full_path)
                rel_path_str = str(rel_path).replace("\\", "/")  # ✅ 在循环内

                files.append({
                    "name": f,
                    "path": rel_path_str,
                    "is_dir": False,
                    "size": os.path.getsize(full_file_path),
                })

            # =============================================
            # ✅ 修复2：遍历文件夹（缩进正确）
            # =============================================
            for d in dirs:
                full_dir_path = Path(root) / d
                rel_path = full_dir_path.relative_to(full_path)
                rel_path_str = str(rel_path).replace("\\", "/") + "/"  # ✅ 在循环内

                files.append({
                    "name": d,
                    "path": rel_path_str,
                    "is_dir": True,
                    "size": 0,
                })

        logger.info(f"✅ 找到 {len(files)} 个文件/文件夹")
        return {"files": files}

    except Exception as e:
        logger.error(f"❌ 遍历失败: {str(e)}", exc_info=True)
        return {"files": []}


# 下载单个文件
@app.get("/download/file")
def download_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")

    if os.path.isdir(path):
        raise HTTPException(400, "path is directory, use zip download")

    return FileResponse(
        path,
        filename=os.path.basename(path)
    )


# 下载为zip包
@app.get("/download/dir")
def download_dir(path: str):
    """下载目录（ZIP）"""

    logger.info(f"📥 下载目录请求: {path}")

    # ✅ 修复1：统一使用 / 作为分隔符（跨平台）
    normalized_path = path.replace("\\", "/")

    # ✅ 修复2：构建绝对路径
    if os.path.isabs(normalized_path):
        full_path = Path(normalized_path)
    else:
        full_path = Path.cwd() / normalized_path

    # ✅ 修复3：标准化路径（解析符号链接、规范化分隔符）
    full_path = full_path.resolve()

    logger.info(f"🔍 标准化路径: {full_path}")
    logger.info(f"🔍 路径存在: {full_path.exists()}")
    logger.info(f"🔍 是否目录: {full_path.is_dir()}")

    if not full_path.exists():
        logger.error(f"❌ 路径不存在: {full_path}")
        raise HTTPException(404, f"路径不存在: {path}")

    if not full_path.is_dir():
        logger.error(f"❌ 不是目录: {full_path}")
        raise HTTPException(400, f"不是目录: {path}")

    tmp_zip = None
    try:
        tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        logger.info(f"📦 临时 ZIP 文件: {tmp_zip.name}")

        with zipfile.ZipFile(tmp_zip.name, "w") as z:
            file_count = 0
            for root, _, files in os.walk(full_path):
                for f in files:
                    full_file_path = Path(root) / f
                    arcname = full_file_path.relative_to(full_path)
                    z.write(full_file_path, arcname)
                    file_count += 1

            logger.info(f"✅ 打包完成: {file_count} 个文件")

        return FileResponse(
            tmp_zip.name,
            filename=full_path.name + ".zip",
            media_type="application/zip"
        )

    except Exception as e:
        logger.error(f"❌ 打包失败: {str(e)}", exc_info=True)
        raise HTTPException(500, f"打包失败: {str(e)}")

    finally:
        if tmp_zip and os.path.exists(tmp_zip.name):
            try:
                os.unlink(tmp_zip.name)
                logger.info(f"🧹 清理临时文件: {tmp_zip.name}")
            except:
                pass


# ============================================
#  修改入口：打包时用这个函数启动
# ============================================
def start_server():
    """启动后端 API（供 subprocess 调用）"""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
        log_config=None,
        access_log=False
    )


#  原来的入口（保留，方便单独测试）
if __name__ == "__main__":
    start_server()
