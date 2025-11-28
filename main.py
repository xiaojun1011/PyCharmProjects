import asyncio
import uuid
import time
from enum import Enum
from typing import Dict, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==========================================
# 1. 配置与数据模型
# ==========================================

app = FastAPI(title="AI Service Backend", version="1.0.0")

# 配置CORS，允许跨域，方便前端调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存数据库，用于存储异步任务的状态和结果
# 在生产环境中，这里应该替换为Redis或MySQL
tasks_db: Dict[str, dict] = {}


class TaskType(str, Enum):
    TRANSLATE_CN_TO_EN = "cn_to_en"
    TRANSLATE_EN_TO_CN = "en_to_cn"
    SUMMARIZE = "summarize"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRequest(BaseModel):
    task_type: TaskType = Field(..., description="任务类型")
    content: str = Field(..., description="需要处理的文本内容")


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class TaskResultResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[str] = None
    error: Optional[str] = None


# ==========================================
# 2. 模拟大模型服务层 (Pseudo Code Implementation)
# ==========================================

class MockLLMService:
    """
    模拟大模型服务的类。
    """
    async def process_async(self, task_type: TaskType, content: str) -> str:
        """
        模拟耗时的非流式生成过程
        """
        # 模拟模型"思考"的网络延迟
        await asyncio.sleep(2)

        # 此处实际调用大模型
        # response = openai.ChatCompletion.create(...)
        # return response.choices[0].message.content

        if task_type == TaskType.TRANSLATE_CN_TO_EN:
            return f"[AI Translate]: {content} -> (Translated to English)"
        elif task_type == TaskType.TRANSLATE_EN_TO_CN:
            return f"[AI 翻译]: {content} -> (翻译为中文)"
        elif task_type == TaskType.SUMMARIZE:
            return f"[AI Summary]: Key points extracted from {len(content)} chars..."
        return "Unknown task"

    async def process_stream(self, task_type: TaskType, content: str) -> AsyncGenerator[str, None]:
        """
        模拟流式生成过程 (Generator)
        """
        # 模拟生成的内容
        mock_response_text = ""
        if task_type == TaskType.TRANSLATE_CN_TO_EN:
            mock_response_text = "This is a streaming response translated from Chinese."
        elif task_type == TaskType.TRANSLATE_EN_TO_CN:
            mock_response_text = "这是从英文翻译过来的流式响应内容。"
        else:
            mock_response_text = "Here is the summary of the content you provided, generated token by token."

        tokens = list(mock_response_text)

        # 模拟逐字吐出
        for token in tokens:
            await asyncio.sleep(0.1)  # 模拟生成每个token的时间间隔
            yield token


llm_service = MockLLMService()


# ==========================================
# 3. 后台任务处理逻辑
# ==========================================

async def background_task_runner(task_id: str, task_type: TaskType, content: str):
    """
    实际执行后台任务的函数
    """
    try:
        # 更新状态为处理中
        tasks_db[task_id]["status"] = TaskStatus.PROCESSING

        # 调用大模型
        result = await llm_service.process_async(task_type, content)

        # 更新结果
        tasks_db[task_id]["result"] = result
        tasks_db[task_id]["status"] = TaskStatus.COMPLETED
    except Exception as e:
        tasks_db[task_id]["error"] = str(e)
        tasks_db[task_id]["status"] = TaskStatus.FAILED


# ==========================================
# 4. API 接口实现
# ==========================================

@app.get("/", tags=["Health"])
async def root():
    return {"message": "AI Backend is running"}


@app.get("/api/features", tags=["Features"])
async def get_features():
    """获取系统支持的功能列表"""
    return {
        "features": [
            {"code": "cn_to_en", "name": "中译英", "description": "将中文文本翻译成英文"},
            {"code": "en_to_cn", "name": "英译中", "description": "将英文文本翻译成中文"},
            {"code": "summarize", "name": "文本总结", "description": "提取文本的核心摘要"}
        ]
    }


@app.post("/api/task/submit", response_model=TaskResponse, tags=["Async Task"])
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """
    提交一个耗时任务，立即返回 Task ID。
    任务将在后台异步执行。
    """
    task_id = str(uuid.uuid4())

    # 初始化任务状态
    tasks_db[task_id] = {
        "id": task_id,
        "type": request.task_type,
        "status": TaskStatus.PENDING,
        "created_at": time.time(),
        "result": None,
        "error": None
    }

    # 添加到后台任务队列 (FastAPI 自动处理)
    background_tasks.add_task(
        background_task_runner,
        task_id,
        request.task_type,
        request.content
    )

    return {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "message": "Task submitted successfully. Please poll /api/task/{task_id} for results."
    }


@app.get("/api/task/{task_id}", response_model=TaskResultResponse, tags=["Async Task"])
async def get_task_result(task_id: str):
    """
    根据 Task ID 查询任务状态和结果。
    """
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task.get("result"),
        "error": task.get("error")
    }


@app.post("/api/stream", tags=["Streaming"])
async def stream_ai_response(request: TaskRequest):
    """
    建立流式连接，实时获取大模型生成的内容。
    """

    async def event_generator():
        # 1. 模拟开始
        # yield "data: [START]\n\n"

        # 2. 调用流式服务
        async for token in llm_service.process_stream(request.task_type, request.content):
            # 这里可以直接返回 token，也可以封装成 SSE 格式 (data: {...})
            # 为了演示简单，这里直接返回文本块
            yield token

        # 3. 模拟结束
        # yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"  # 如果是 SSE，通常使用 "text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
