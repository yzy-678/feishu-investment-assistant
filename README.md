# 飞书AI投资助手

飞书机器人驱动的智能投资研究系统。支持自动市场日报、盘中预警、股票分析、自选股管理。

## 架构总览

```
┌─────────────────────┐      ┌─────────────────────┐
│     飞书用户         │      │  GitHub Actions      │
│  · 消息问答          │      │  · 08:00 早报       │
│  · 命令控制          │      │  · 12:00 午间观察    │
│  · 自选股管理        │      │  · 16:00 收盘复盘    │
└─────────┬───────────┘      └─────────┬───────────┘
          │  Webhook                    │  CLI 调用
          ▼                             ▼
┌──────────────────────────────────────────────┐
│           Railway (FastAPI)                    │
│                                                │
│  /feishu/event  ← 飞书事件回调                  │
│  /health         → 健康检查                     │
│  /api/config     → 内部 API                    │
│                                                │
│  MessageHandler → Coordinator → Agent → AI    │
└──────────────────────────────────────────────┘
```

## 模块说明

| 模块 | 说明 |
|------|------|
| `bot/` | 飞书集成（Client / Handler / Router） |
| `agents/` | Agent 系统（Coordinator / Market / Report / Alert） |
| `ai/` | DeepSeek API 封装 |
| `market/` | 行情数据源（Mock / EastMoney） |
| `memory/` | 对话记忆（短期 + 摘要） |
| `watchlist/` | 自选股管理 |
| `config/` | 配置管理（环境变量 + SQLite） |
| `db/` | 数据库层 |
| `scheduler/` | 后台调度 + CLI 任务 |

## 快速开始

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key 和飞书应用凭证
```

### 2. 安装

```bash
pip install -r requirements.txt
```

### 3. 本地启动

```bash
python -m src
# 或 uvicorn src.main:app --reload
```

### 4. 测试

```bash
pytest tests/ -v
```

## 飞书配置

1. 在 [飞书开发者后台](https://open.feishu.cn/app) 创建应用
2. 启用「机器人」能力
3. 开启「事件订阅」，配置请求 URL：`https://your-app.railway.app/feishu/event`
4. 添加事件 `im.message.receive_v1`
5. 发布应用

## 部署

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

1. 点击上方按钮
2. 连接 GitHub 仓库
3. 配置环境变量
4. 部署完成

### GitHub Actions 定时任务

配置以下 Secrets：

| Secret | 说明 |
|--------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `FEISHU_APP_ID` | 飞书应用 App ID |
| `FEISHU_APP_SECRET` | 飞书应用 App Secret |
| `ADMIN_USER_OPEN_ID` | 管理员 open_id |
| `RAILWAY_TOKEN` | Railway 部署 Token |

## API 端点

| 路径 | 方法 | 说明 |
|------|------|------|
| `/feishu/event` | POST | 飞书事件回调 |
| `/health` | GET | 健康检查 |
| `/api/config` | GET | 运行配置 |

## 飞书命令

### 系统控制
- `启动` / `暂停` — 开关系统
- `状态` — 查看当前配置
- `切换A股` / `切换港股` / `切换美股`
- `扫描频率 30` — 设置盘中扫描间隔

### 自选股管理
- `添加自选 000001`
- `删除自选 000001`
- `我的自选` / `自选股`
- `清空自选`

### 市场问答
- `今天市场怎么样` — 大盘概览
- `分析 平安银行` — 个股分析
- `银行板块怎么样` — 板块分析
- `今天主线是什么` — 市场主线
- `生成早报` / `午报` / `收盘复盘` — 日报

## 本地测试

```bash
# 启动服务
./scripts/run_local.sh

# 模拟飞书事件
curl -X POST http://localhost:8000/feishu/event \
  -H "Content-Type: application/json" \
  -d '{"type":"url_verification","challenge":"test"}'

# 模拟定时任务
python -m src.scheduler.tasks morning
python -m src.scheduler.tasks noon
python -m src.scheduler.tasks closing
```

## 项目结构

```
src/
├── main.py                  # FastAPI 应用入口
├── config/                  # 配置管理（env + SQLite）
├── db/                      # 数据库层
├── bot/                     # 飞书集成
├── ai/                      # DeepSeek API
├── market/                  # 行情数据源
├── agents/                  # Agent 系统
│   ├── coordinator.py       # 消息路由
│   ├── market_agent.py      # 市场问答
│   ├── report_agent.py      # 日报生成
│   └── alert_agent.py       # 预警监控
├── watchlist/               # 自选股管理
├── memory/                  # 对话记忆
├── scheduler/               # 定时任务
└── reports/                 # 报告模板
```
